// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformDialectStrategiesGPU.h"

#include <numeric>
#include <type_traits>
#include <utility>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformDialectStrategies.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ConfigExtractPart;
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::
    MapNestedForeachThreadToGpuThreadsOp;
using iree_compiler::IREE::transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::MergeHandlesOp;
using transform::PrintOp;
using transform::SequenceOp;
using transform::SplitHandlesOp;
using transform::SplitReductionOp;
using transform::TileToForeachThreadOp;
using transform::VectorizeOp;
using transform_ext::AllDims;
using transform_ext::IsPermutation;
using transform_ext::m_StructuredOp;
using transform_ext::MatchCallbackOp;
using transform_ext::NumEqualsTo;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::ShapeKind;
using transform_ext::StructuredOpMatcher;
using transform_ext::TakeFirstOp;

/// Matches `args` within `targetH` and unpacks a number of handles `N`.
/// Assumes there are exactly `N` matched ops (but could be relaxed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto matchAndUnpack(ImplicitLocOpBuilder &b, Value targetH,
                    MatchingArgs... args) {
  Value matchedH = b.create<MatchOp>(targetH, args...);
  auto matchOp = b.create<SplitHandlesOp>(matchedH,
                                          /*numHandles=*/N);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i) a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

/// Matches a C++ callback previously registered under `callbackName` and
/// taking arguments `args`.
/// Unpacks a number of handles `N` (asserts there are exactly `N` matched ops
/// but this could be relaxed if needed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto unpackRegisteredMatchCallback(ImplicitLocOpBuilder &b,
                                   StringRef callbackName,
                                   MatchingArgs... args) {
  SmallVector<Type> matchedTypes(N, pdl::OperationType::get(b.getContext()));
  auto matchOp = b.create<MatchCallbackOp>(
      matchedTypes, callbackName, std::forward<decltype(args)>(args)...);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i) a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

namespace {

/// Compute good tile and vector sizes for the reduction dimension of a 1-D
/// reduction dimension for a TileReductionUsingForeachThreadOp strategy.
///
/// Dynamic case: use as many threads as allowed along threadIdx.x with vector
/// size of 1 (i.e. coalesced accesses).
/// This can be further refined with splitting or vector masking when
/// available.
///
/// Static case: perfectly tile by:
///   - 128 to obtain 32*k threads working on vector<4xf32> with k as high as
///   possible within the limits of maxNumThreadsToUse, when possible;
///   - 64 to obtain 32*k threads working on vector<2xf32> with k as high as
///   possible within the limits of maxNumThreadsToUse, when possible;
///   - reductionDimensionSize within the limits of maxNumThreadsToUse,
///   otherwise.
// TODO: refine even further based on mod 2 and mod 4 only + min
// canonicalizations.
// TODO: refine sizes based on the bitwidth of the elemental type.
class ReductionStrategyThreadDistributionSizes {
 public:
  explicit ReductionStrategyThreadDistributionSizes(
      int64_t reductionDimensionSize = 0,
      int64_t maxNumThreadsToUse = iree_compiler::kCudaMaxNumThreads)
      : reductionDimensionSize(reductionDimensionSize),
        maxNumThreadsToUse(maxNumThreadsToUse) {
    computeStrategy();
  }
  ReductionStrategyThreadDistributionSizes(
      const ReductionStrategyThreadDistributionSizes &) = default;

  ReductionStrategyThreadDistributionSizes &operator=(
      const ReductionStrategyThreadDistributionSizes &) = default;

  int64_t reductionTileSize;
  int64_t vectorTileSize;

 private:
  void computeStrategy();

  int64_t reductionDimensionSize;
  // TODO: Characterize shared memory consumption of this strategy and limit
  // accordingly for good occupancy.
  int64_t maxNumThreadsToUse;
};

static int64_t maxMultipleOf(int64_t val, int64_t multiple) {
  assert(val > 0 && "expected nonnegative quantity");
  assert(multiple > 0 && "expected nonnegative quantity");
  return (val / multiple) * multiple;
}
static int64_t nextMultipleOf(int64_t val, int64_t multiple) {
  assert(val > 0 && "expected nonnegative quantity");
  assert(multiple > 0 && "expected nonnegative quantity");
  return ((val + multiple - 1) / multiple) * multiple;
}

void ReductionStrategyThreadDistributionSizes::computeStrategy() {
  vectorTileSize = 1;
  reductionTileSize = maxNumThreadsToUse;
  if (reductionDimensionSize <= 0) return;

  // When we know the problem size is divisible by k, we know we want to
  // target vector<kxf32>. Over-provision to the next multiple of the warp
  // size fo ensure we tile to a multiple of the warp size to allow proper
  // distribution across warp shuffles.
  // TODO: refine with elemental type bitwidth and GPU memory transaction size.
  if (reductionDimensionSize % 4 == 0) {
    reductionTileSize = std::min(
        nextMultipleOf(reductionDimensionSize / 4,
                       iree_compiler::kCudaWarpSize),
        maxMultipleOf(maxNumThreadsToUse, iree_compiler::kCudaWarpSize));
    vectorTileSize = 4;
  } else if (reductionDimensionSize % 2 == 0) {
    reductionTileSize = std::min(
        nextMultipleOf(reductionDimensionSize / 2,
                       iree_compiler::kCudaWarpSize),
        maxMultipleOf(maxNumThreadsToUse, iree_compiler::kCudaWarpSize));
    vectorTileSize = 2;
  } else {
    reductionTileSize = std::min(
        nextMultipleOf(reductionDimensionSize, iree_compiler::kCudaWarpSize),
        maxMultipleOf(maxNumThreadsToUse, iree_compiler::kCudaWarpSize));
    vectorTileSize = 1;
  }
}

/// Structure to hold the parameters related to GPU reduction strategy.
struct GPUReductionStrategyInfos {
  explicit GPUReductionStrategyInfos(MLIRContext *context, int64_t rank,
                                     int64_t reductionDimensionSize)
      : context(context),
        rank(rank),
        reductionDimensionSize(reductionDimensionSize) {
    auto blockX =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimX);
    auto blockY =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimY);
    auto blockZ =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimZ);
    allBlockAttrs = SmallVector<Attribute>{blockX, blockY, blockZ};

    if (reductionDimensionSize >= 4) {
      setThreadDistributionSizes(4);
    } else if (reductionDimensionSize >= 2) {
      setThreadDistributionSizes(2);
    } else {
      threadDistributionSizes =
          ReductionStrategyThreadDistributionSizes(reductionDimensionSize);
    }
  }

  void setThreadDistributionSizes(int64_t baseSize) {
    splitPoint = reductionDimensionSize - reductionDimensionSize % baseSize;
    threadDistributionSizes =
        ReductionStrategyThreadDistributionSizes(splitPoint);
    if (reductionDimensionSize % baseSize != 0) {
      secondaryThreadDistributionSizes =
          ReductionStrategyThreadDistributionSizes(reductionDimensionSize %
                                                   baseSize);
    }
  }

  /// Constructor quantities.
  MLIRContext *context;
  int64_t rank;
  int64_t reductionDimensionSize;
  ReductionStrategyThreadDistributionSizes threadDistributionSizes;
  std::optional<ReductionStrategyThreadDistributionSizes>
      secondaryThreadDistributionSizes = std::nullopt;

  /// Derived quantities.
  SmallVector<Attribute> allBlockAttrs;
  int64_t splitPoint;
  // Tile sizes for the workgroup / determines grid size.
  SmallVector<int64_t> workgroupTileSizes;
  // Launch bounds for the workgroups / block size.
  std::array<int64_t, 3> workgroupSize;
};
}  // namespace

static std::pair<Value, Value> createReductionStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value maybeLeadingH, Value fillH, Value reductionH,
    Value maybeTrailingH, const GPUReductionStrategyInfos &infos) {
  auto pdlOperation = pdl::OperationType::get(b.getContext());
  auto fusionTargetSelector = b.create<TakeFirstOp>(
      pdlOperation, pdlOperation, ArrayRef<Value>{maybeTrailingH, reductionH});
  Value fusionTargetH = fusionTargetSelector.getFirst();
  Value fusionGroupH = fusionTargetSelector.getRest();
  ArrayRef<Attribute> allBlocksRef(infos.allBlockAttrs);
  iree_compiler::TileToForeachThreadAndFuseAndDistributeResult tileResult =
      iree_compiler::
          buildTileFuseDistToForeachThreadAndWorgroupCountWithTileSizes(
              /*builder=*/b,
              /*rootH=*/fusionTargetH,
              /*opsToFuseH=*/fusionGroupH,
              /*tileSizes=*/
              getAsOpFoldResult(b.getI64ArrayAttr(infos.workgroupTileSizes)),
              /*threadDimMapping=*/
              b.getArrayAttr(allBlocksRef.take_front(infos.rank - 1)));
  Value foreachThreadH =
      b.create<FuseIntoContainingOp>(fillH, tileResult.foreachThreadH);
  foreachThreadH =
      b.create<FuseIntoContainingOp>(maybeLeadingH, foreachThreadH);
  auto gridReductionSelector = b.create<TakeFirstOp>(
      pdlOperation, pdlOperation,
      ValueRange{tileResult.resultingFusedOpsHandles.front(),
                 tileResult.tiledOpH});

  return std::make_pair(gridReductionSelector.getFirst(),
                        gridReductionSelector.getRest());
}

static void createReductionStrategyReductionTiling(
    ImplicitLocOpBuilder &b, Value reductionH, int rank,
    const ReductionStrategyThreadDistributionSizes &sizes) {
  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);
  auto threadY = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimY);

  // Split the reduction into a parallel and combiner part, then tile the
  // parallel part and map it to a full warp so it works on vectors.
  SmallVector<int64_t> leadingParallelDims(rank - 1, 0);
  SmallVector<int64_t> numThreads = leadingParallelDims;
  numThreads.push_back(sizes.reductionTileSize);
  SmallVector<int64_t> tileSizes = leadingParallelDims;
  tileSizes.push_back(sizes.vectorTileSize);
  auto tileReduction = b.create<transform::TileReductionUsingForeachThreadOp>(
      /*target=*/reductionH,
      /*numThreads=*/numThreads,
      /*tileSizes=*/tileSizes,
      /*threadDimMapping=*/b.getArrayAttr(threadX));
  Value blockParallelForeachThreadOp = tileReduction.getForeachThreadOp();
  Value blockParallelFillH = tileReduction.getFillOp();
  Value blockCombinerOpH = tileReduction.getCombiningLinalgOp();

  // Fuse the fill and pointwise to privatize them.
  blockParallelFillH = b.create<FuseIntoContainingOp>(
      blockParallelFillH, blockParallelForeachThreadOp);

  // Map the combiner reduction to one thread along y so it can be mapped
  // further via predication.
  iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
      b, blockCombinerOpH, {}, getAsOpFoldResult(b.getI64ArrayAttr({1})),
      b.getArrayAttr(threadY));
}

static void createReductionStrategyThreadDistribution(
    ImplicitLocOpBuilder &b, Value gridReductionH, Value maybeTiledTrailingH,
    const GPUReductionStrategyInfos &infos) {
  Value mainReductionH = gridReductionH;
  Value leftoverReductionH = nullptr;
  if (infos.secondaryThreadDistributionSizes) {
    auto pdlOperation = pdl::OperationType::get(b.getContext());
    auto split = b.create<transform::SplitOp>(
        pdlOperation, pdlOperation, gridReductionH,
        b.getI64IntegerAttr(infos.rank - 1), Value(),
        b.getI64IntegerAttr(infos.splitPoint));
    mainReductionH = split.getFirst();
    leftoverReductionH = split.getSecond();
  }
  createReductionStrategyReductionTiling(b, mainReductionH, infos.rank,
                                         infos.threadDistributionSizes);
  if (leftoverReductionH) {
    createReductionStrategyReductionTiling(
        b, leftoverReductionH, infos.rank,
        *infos.secondaryThreadDistributionSizes);
  }

  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);
  // auto threadY = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
  //                                                     mlir::gpu::Threads::DimY);

  // // Split the reduction into a parallel and combiner part, then tile the
  // // parallel part and map it to a full warp so it works on vectors.
  // SmallVector<int64_t> leadingParallelDims(infos.rank - 1, 0);
  // SmallVector<int64_t> numThreads = leadingParallelDims;
  // numThreads.push_back(infos.threadDistributionSizes.reductionTileSize);
  // SmallVector<int64_t> tileSizes = leadingParallelDims;
  // tileSizes.push_back(infos.threadDistributionSizes.vectorTileSize);
  // auto tileReduction =
  // b.create<transform::TileReductionUsingForeachThreadOp>(
  //     /*target=*/gridReductionH,
  //     /*numThreads=*/numThreads,
  //     /*tileSizes=*/tileSizes,
  //     /*threadDimMapping=*/b.getArrayAttr(threadX));
  // Value blockParallelForeachThreadOp = tileReduction.getForeachThreadOp();
  // Value blockParallelFillH = tileReduction.getFillOp();
  // Value blockCombinerOpH = tileReduction.getCombiningLinalgOp();

  // // Fuse the fill and pointwise to privatize them.
  // blockParallelFillH = b.create<FuseIntoContainingOp>(
  //     blockParallelFillH, blockParallelForeachThreadOp);

  // // Map the combiner reduction to one thread along y so it can be mapped
  // // further via predication.
  // iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
  //     b, blockCombinerOpH, {}, getAsOpFoldResult(b.getI64ArrayAttr({1})),
  //     b.getArrayAttr(threadY));

  // Map the potential maybeTiledTrailingH.
  if (maybeTiledTrailingH) {
    assert(infos.rank >= 1);
    SmallVector<int64_t> trailingTileSizes(infos.rank - 1, 0);
    if (infos.rank > 1) trailingTileSizes.back() = infos.workgroupSize[0];

    auto res = iree_compiler::buildTileFuseToScfFor(
        b, maybeTiledTrailingH, {},
        getAsOpFoldResult(b.getI64ArrayAttr(trailingTileSizes)));
    iree_compiler::buildTileFuseDistToForeachThreadWithNumThreads(
        b, res.tiledOpH, {},
        getAsOpFoldResult(b.getI64ArrayAttr(trailingTileSizes)),
        b.getArrayAttr({threadX}));
  }
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
static void createReductionCudaStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const GPUReductionStrategyInfos &infos) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [maybeLeadingH, fillH, reductionH, maybeTrailingH] =
      unpackRegisteredMatchCallback<4>(
          b, "reduction", transform::FailurePropagationMode::Propagate,
          variantH);

  // Step 2. Use tiling to introduce a single-iteration loop mapped to a
  // single block/workgroup. Keep everything fused.
  auto [gridReductionH, maybeTiledTrailingH] =
      createReductionStrategyBlockDistribution(
          b, maybeLeadingH, fillH, reductionH, maybeTrailingH, infos);

  // Step 3. Split the reduction and tile the pieces to ensure vector
  // load/stores and mapping to a single warp with shuffles.
  createReductionStrategyThreadDistribution(b, gridReductionH,
                                            maybeTiledTrailingH, infos);

  // Step 4. Bufferize and drop HAL decriptor from memref ops.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);
  variantH = iree_compiler::buildBufferize(b, variantH, /*targetGpu=*/true);

  // Step 5. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH =
      iree_compiler::buildMapToBlockAndThreads(b, funcH, infos.workgroupSize);

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  iree_compiler::buildDistributeVectors(b, variantH, funcH);
}

static FailureOr<GPUReductionStrategyInfos> matchGPUReduction(
    linalg::LinalgOp op) {
  StructuredOpMatcher reduction, fill, leading, trailing;
  transform_ext::MatchedReductionCaptures captures;
  makeReductionMatcher(reduction, fill, leading, trailing, captures);
  if (!matchPattern(op, reduction)) return failure();

  //
  // !!We must match exactly all payload ops when the dispatch is pre-formed!!
  //
  int64_t mustMatchNumPayloadOps =
      transform_ext::getNumPayloadOpsThatWeMustMatch(
          op->getParentOfType<func::FuncOp>());
  int64_t numMatchedOps = 2;  // Mandatory operations.
  if (leading.getCaptured()) ++numMatchedOps;
  if (trailing.getCaptured()) ++numMatchedOps;
  if (numMatchedOps != mustMatchNumPayloadOps) {
    LLVM_DEBUG({
      DBGS() << "Failed to match " << mustMatchNumPayloadOps
             << " payload ops, matched " << numMatchedOps << " instead\n";
    });
    return failure();
  }

  GPUReductionStrategyInfos info(op->getContext(), captures.rank,
                                 captures.reductionDimensionSize);

  // Tile all the parallel dimensions to 1 and create many blocks.
  int64_t numParallelLoops = captures.rank - 1;
  info.workgroupTileSizes.append(numParallelLoops, 1);
  // Tile and distribute the reduction across `reductionTileSize` threads.
  info.workgroupSize = {
      info.secondaryThreadDistributionSizes
          ? std::max(info.threadDistributionSizes.reductionTileSize,
                     info.secondaryThreadDistributionSizes->reductionTileSize)
          : info.threadDistributionSizes.reductionTileSize,
      1, 1};
  return info;
}

LogicalResult iree_compiler::matchAndSetGPUReductionTransformStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
  // 1. Match.
  FailureOr<GPUReductionStrategyInfos> maybeInfos = matchGPUReduction(op);
  if (failed(maybeInfos)) return failure();
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return createReductionCudaStrategy(b, variant, *maybeInfos);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, strategyBuilder);
  return success();
}
