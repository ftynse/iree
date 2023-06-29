// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_

#include <functional>
#include <numeric>
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/AbstractGemmLikeStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/CopyMapping.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

class MatmulStrategy : public AbstractGemmLikeStrategy {
public:
  MatmulStrategy(MLIRContext *context,
                 const transform_ext::MatchedMatmulCaptures &captures,
                 const GPUModel &gpuModel)
      : AbstractGemmLikeStrategy(gpuModel), ctx(context), captures(captures) {
    initDefaultValues(gpuModel);
  }

  MatmulStrategy(const MatmulStrategy &) = default;
  MatmulStrategy &operator=(const MatmulStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedMatmulCaptures captures;

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
  void initDefaultValues(const GPUModel &gpuModel) override;

  LogicalResult validate(const GPUModel &gpuModel) const override;

  int64_t m() const override {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[0];
  }
  int64_t n() const override {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[1];
  }
  int64_t k() const override {
    assert(captures.matmulOpSizes.size() == 3 && "need 3 sizes");
    return captures.matmulOpSizes[2];
  }

  int64_t blockTileM() const override {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[0];
  }
  int64_t blockTileN() const override {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[1];
  }

  int64_t numThreadsX() const override {
    assert(numThreads.size() >= 2 && "need at least 2 thread sizes");
    return numThreads[0];
  }
  int64_t numThreadsY() const override {
    assert(numThreads.size() >= 2 && "need at least 2 thread sizes");
    return numThreads[1];
  }
  int64_t numWarpsX() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[0];
  }
  int64_t numWarpsY() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[1];
  }

  Type getLhsElementalType() const override { return captures.lhsElementType; }
  Type getRhsElementalType() const override { return captures.rhsElementType; }
  Type getResElementalType() const override {
    return captures.outputElementType;
  }

  MappingInfo getBlockMapping() const override {
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/{blockTileM(), blockTileN()},
                       /*threadMapping=*/{blockY(ctx), blockX(ctx)},
                       /*vectorSize=*/std::nullopt};
  }

  auto map(ArrayRef<AffineExpr> exprs) const {
    return AffineMap::get(captures.rank(), 0, exprs, ctx);
  };

  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    int64_t alignment;
    SmallVector<int64_t> copySizes;
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    if (captures.lhsIndexing() == map({m, k})) {
      alignment = this->k();
      copySizes = SmallVector<int64_t>{blockTileM(), reductionTileSize};
    } else if (captures.lhsIndexing() == map({k, m})) {
      alignment = this->m();
      copySizes = SmallVector<int64_t>{reductionTileSize, blockTileM()};
    } else {
      captures.lhsIndexing().dump();
      llvm_unreachable("unsupported lhs indexing");
    }
    return CopyMapping::getMappingInfo(
        ctx,
        /*totalNumThreads=*/totalNumThreads(),
        /*alignment=*/alignment,
        /*copySizes=*/copySizes,
        /*favorPredication=*/false,
        /*elementalBitWidth=*/lhsElementalBitWidth());
  }
  LogicalResult validateLhsCopyMapping() const override {
    MappingInfo mapping = lhsCopyMapping();
    // It is fine to use fewer threads to copy the LHS.
    if (totalNumThreads() < mapping.numThreads[0] * mapping.numThreads[1]) {
      llvm::errs() << "too many threads used for transferring lhs: "
                   << mapping.numThreads[0] << " * " << mapping.numThreads[1]
                   << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    int64_t alignment;
    SmallVector<int64_t> copySizes;
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    if (captures.rhsIndexing() == map({k, n})) {
      alignment = this->n();
      copySizes = SmallVector<int64_t>{reductionTileSize, blockTileN()};
    } else if (captures.rhsIndexing() == map({n, k})) {
      alignment = this->k();
      copySizes = SmallVector<int64_t>{blockTileN(), reductionTileSize};
    } else {
      captures.rhsIndexing().dump();
      llvm_unreachable("unsupported rhs indexing");
    }
    return CopyMapping::getMappingInfo(
        ctx,
        /*totalNumThreads=*/totalNumThreads(),
        /*alignment=*/alignment,
        /*copySizes=*/copySizes,
        /*favorPredication=*/false,
        /*elementalBitWidth=*/rhsElementalBitWidth());
  }
  LogicalResult validateRhsCopyMapping() const override {
    MappingInfo mapping = rhsCopyMapping();
    // It is fine to use fewer threads to copy the RHS.
    if (totalNumThreads() < mapping.numThreads[0] * mapping.numThreads[1]) {
      llvm::errs() << "too many threads used for transferring rhs: "
                   << mapping.numThreads[0] << " * " << mapping.numThreads[1]
                   << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    int64_t alignment;
    SmallVector<int64_t> copySizes;
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    if (captures.resIndexing() == map({m, n})) {
      alignment = this->n();
      copySizes = SmallVector<int64_t>{blockTileM(), blockTileN()};
    } else if (captures.resIndexing() == map({n, m})) {
      alignment = this->m();
      copySizes = SmallVector<int64_t>{blockTileN(), blockTileM()};
    } else {
      captures.resIndexing().dump();
      llvm_unreachable("unsupported res indexing");
    }
    return CopyMapping::getMappingInfo(
        ctx,
        /*totalNumThreads=*/totalNumThreads(),
        /*alignment=*/alignment,
        /*copySizes=*/copySizes,
        /*favorPredication=*/false,
        /*elementalBitWidth=*/resElementalBitWidth());
  }

  LogicalResult validateResCopyMapping() const override {
    MappingInfo mapping = resCopyMapping();
    // It is fine to use fewer threads to copy the RES.
    if (totalNumThreads() < mapping.numThreads[0] * mapping.numThreads[1]) {
      llvm::errs() << "too many threads used for transferring res: "
                   << mapping.numThreads[0] << " * " << mapping.numThreads[1]
                   << " > " << totalNumThreads() << "\n";
      return failure();
    }
    return success();
  }

  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    if (useFma) {
      // When using FMA we don't need to map to warps, instead just match what
      // the copy does.
      // return CopyMapping::getMappingInfo(ctx, totalNumThreads(),
      //                                    /*alignment=*/n(),
      //                                    {blockTileM(), blockTileN()});
      return MappingInfo{/*numThreads=*/{numThreadsY(), numThreadsX()},
                         /*tileSizes=*/{},
                         /*threadMapping=*/{threadY(ctx), threadX(ctx)},
                         /*vectorSize=*/std::nullopt};
    }
    return MappingInfo{/*numThreads=*/{numWarpsY(), numWarpsX()},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpY(ctx), warpX(ctx)},
                       /*vectorSize=*/std::nullopt};
  }

  void print(llvm::raw_ostream &os) const override;
  LLVM_DUMP_METHOD void dump() const override;
};

class BatchMatmulStrategy : public MatmulStrategy {
public:
  // TOOD: captures
  BatchMatmulStrategy(MLIRContext *context, const GPUModel &gpuModel,
                      const transform_ext::MatchedMatmulCaptures &captures)
      : MatmulStrategy(context, captures, gpuModel) {
    initDefaultValues(gpuModel);
  }

  void initDefaultValues(const GPUModel &gpuModel) override {
    MatmulStrategy::initDefaultValues(gpuModel);

    paddingDimensions = {0, 1, 2, 3};
    packingDimensions = {1, 1, 1, 1};
  }

  LogicalResult validate(const GPUModel &gpuModel) const override {
    if (failed(MatmulStrategy::validate(gpuModel)))
      return failure();

    if (batch() < blockTileBatch()) {
      return emitError(UnknownLoc::get(ctx))
             << "batch( " << batch() << ") <  blockTileBatch("
             << blockTileBatch()
             << ") this is at risk of not vectorizing and is NYI";
    }

    // TODO: not sure this is the right place
    if (captures.batches().size() != 1 || captures.batches().back() != 0)
      return failure();

    if (blockTileSizes.size() < 3)
      return failure();

    if (numWarps.size() < 3)
      return failure();

    if (numThreads.size() < 3)
      return failure();

    // TODO: debug output
    if (!useFma)
      return failure();

    return success();
  }

  int64_t batch() const { return captures.matmulOpSizes[0]; }
  int64_t m() const override { return captures.matmulOpSizes[1]; }
  int64_t n() const override { return captures.matmulOpSizes[2]; }
  int64_t k() const override { return captures.matmulOpSizes[3]; }

  int64_t blockTileBatch() const { return blockTileSizes[0]; }
  int64_t blockTileM() const override { return blockTileSizes[1]; }
  int64_t blockTileN() const override { return blockTileSizes[2]; }

  int64_t numThreadsX() const override { return numThreads[0]; }
  int64_t numThreadsY() const override { return numThreads[1]; }
  int64_t numThreadsZ() const { return numThreads[2]; }

  int64_t numWarpsX() const override { return numWarps[0]; }
  int64_t numWarpsY() const override { return numWarps[1]; }
  int64_t numWarpsZ() const { return numWarps[2]; }

  MappingInfo getBlockMapping() const override {
    // TODO: assumes batch dimension to be the outermost, relax and use captures
    // instead.
    return MappingInfo{
        /*numThreads=*/
        {},
        /*tileSizes=*/{blockTileBatch(), blockTileM(), blockTileN()},
        /*threadMapping=*/{blockZ(ctx), blockY(ctx), blockX(ctx)},
        /*vectorSize=*/std::nullopt};
  }

  // TODO: generalize to transpositions
  MappingInfo lhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(), k(),
        {blockTileBatch(), blockTileM(), reductionTileSize},
        /*favorPredication=*/false,
        captures.lhsElementType.getIntOrFloatBitWidth());
  }

  MappingInfo rhsCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(), n(),
        {blockTileBatch(), reductionTileSize, blockTileN()},
        /*favorPredication=*/false,
        captures.rhsElementType.getIntOrFloatBitWidth());
  }

  MappingInfo resCopyMapping() const override {
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(), n(),
        {blockTileBatch(), blockTileM(), blockTileN()},
        /*favorPredication=*/false,
        captures.outputElementType.getIntOrFloatBitWidth());
  }

  LogicalResult validateCopyMapping(const MappingInfo &mapping,
                                    StringRef name) const {
    int64_t threadsUsed =
        std::accumulate(mapping.numThreads.begin(), mapping.numThreads.end(), 1,
                        std::multiplies<int64_t>());
    if (totalNumThreads() < threadsUsed) {
      InFlightDiagnostic diag = emitError(UnknownLoc::get(ctx))
                                << "too many threads used for transferring "
                                << name;

      std::string str;
      llvm::raw_string_ostream os(str);
      llvm::interleave(mapping.numThreads, os, " * ");
      os << " >= " << totalNumThreads();
      diag.attachNote() << os.str();
      return diag;
    }

    return success();
  }

  LogicalResult validateLhsCopyMapping() const override {
    return validateCopyMapping(lhsCopyMapping(), "lhs");
  }

  LogicalResult validateRhsCopyMapping() const override {
    return validateCopyMapping(rhsCopyMapping(), "rhs");
  }

  LogicalResult validateResCopyMapping() const override {
    return validateCopyMapping(resCopyMapping(), "result");
  }

  MappingInfo computeMapping() const override {
    assert(useFma && "only fma is currently supported");

    return MappingInfo{{numThreadsZ(), numThreadsY(), numThreadsY()},
                       {},
                       {threadZ(ctx), threadY(ctx), threadX(ctx)},
                       std::nullopt};
  }
};

} // namespace gpu
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_MATMUL_STRATEGY_H_
