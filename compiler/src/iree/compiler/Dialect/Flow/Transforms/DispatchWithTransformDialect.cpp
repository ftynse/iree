// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterPassBase.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "dispatch-with-transform"
#define DBGS() llvm::dbgs() << DEBUG_TYPE

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Builds transform IR forming dispatch regions for reductions.
void buildReductionDispatch(ImplicitLocOpBuilder &builder, Value reductionH,
                            Value fillH, Value leadingH, Value trailingH,
                            bool emitRemarkOnMatch = false) {
  auto pdlOperation = pdl::OperationType::get(builder.getContext());
  if (emitRemarkOnMatch) {
    builder.create<transform_ext::EmitRemarkOp>(
        reductionH, "dispatch matched reduction: reduction op");
    builder.create<transform_ext::EmitRemarkOp>(
        fillH, "dispatch matched reduction: fill op");
    builder.create<transform_ext::EmitRemarkOp>(
        leadingH, "dispatch matched reduction: leading elementwise op");
    builder.create<transform_ext::EmitRemarkOp>(
        trailingH, "dispatch matched reduction: trailing elementwise op");
  }

  auto [firstH, restH] =
      buildSelectFirstNonEmpty(builder, trailingH, reductionH);
  Value regionH = builder.create<transform_dialect::WrapInDispatchRegionOp>(
      pdlOperation, firstH);
  SmallVector<Value> handlesToMerge({leadingH, fillH});
  handlesToMerge.push_back(restH);
  Value mergedHandlesH = builder.create<transform::MergeHandlesOp>(
      handlesToMerge, /*deduplicate=*/false);
  regionH =
      builder.create<transform_dialect::MovePrecedingOpIntoDispatchRegionOp>(
          mergedHandlesH, regionH);
  builder.create<transform_dialect::RegionToWorkgroupsOp>(pdlOperation,
                                                          regionH);
}

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for dispatch region
/// formation. This needs to be its own pass because the registration mechanism
/// and ops available are different than for other interpreters.
struct DispatchWithTransformDialect
    : public transform::TransformInterpreterPassBase<
          DispatchWithTransformDialect, DispatchWithTransformDialectBase> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    IREE::Flow::FlowDialect,
                    AffineDialect,
                    arith::ArithDialect,
                    linalg::LinalgDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect
    >();
    // clang-format on
  }

  DispatchWithTransformDialect(StringRef transformFileName = StringRef(),
                               StringRef debugPayloadRootTag = StringRef(),
                               StringRef debugTransformRootTag = StringRef(),
                               bool debugEmitRemarkOnMatch = false) {
    this->transformFileName = transformFileName.str();
    this->debugPayloadRootTag = debugPayloadRootTag.str();
    this->debugTransformRootTag = debugTransformRootTag.str();
    this->debugEmitRemarkOnMatch = debugEmitRemarkOnMatch;
  }
  DispatchWithTransformDialect(const DispatchWithTransformDialect &pass)
      : TransformInterpreterPassBase(pass) {
    this->transformFileName = pass.transformFileName;
    this->debugPayloadRootTag = pass.debugPayloadRootTag;
    this->debugTransformRootTag = pass.debugTransformRootTag;
    this->debugEmitRemarkOnMatch = pass.debugEmitRemarkOnMatch;
  }

  /// Adds N arguments with `!pdl.operation` type to the given sequence
  /// operation and returns them as a tuple.
  template <int N>
  auto addArgumentsAndUnpack(Location loc, transform::SequenceOp sequence) {
    auto arguments = sequence.getBodyBlock()->addArguments(
        SmallVector<Type>(N, pdl::OperationType::get(loc->getContext())),
        SmallVector<Location>(N, loc));
    std::array<Value, N> unpacked;
    for (int i = 0; i < N; ++i) unpacked[i] = *std::next(arguments.begin(), i);
    return std::tuple_cat(unpacked);
  }

  /// Adds a mapping entry to be associated with an argument of a top-level
  /// transform operation pointing to the payload operation captured by the
  /// matcher.
  void addMappedOps(
      const transform_ext::StructuredOpMatcher &matcher,
      SmallVectorImpl<ArrayRef<transform::MappedValue>> &topLevelMapping,
      SmallVectorImpl<transform::MappedValue> &topLevelMappingStorage) {
    assert(matcher.getCaptured() && "expected to capture");
    topLevelMappingStorage.push_back(matcher.getCaptured());
    topLevelMapping.push_back(ArrayRef(topLevelMappingStorage).take_back());
  }

  /// Adds a mapping entry to be associated with an argument of a top-level
  /// transform operation pointing to the payload operation captured by the
  /// matcher if any, an empty entry otherwise.
  void addOptionalMappedOps(
      const transform_ext::StructuredOpMatcher &matcher,
      SmallVectorImpl<ArrayRef<transform::MappedValue>> &topLevelMapping,
      SmallVectorImpl<transform::MappedValue> &topLevelMappingStorage) {
    if (matcher.getCaptured()) {
      addMappedOps(matcher, topLevelMapping, topLevelMappingStorage);
    } else {
      topLevelMapping.emplace_back();
    }
  }

  /// If `patterRoot` is the root operation of a reduction structure, returns a
  /// module containing transform dialect that places that reduction into a
  /// dispatch region. Otherwise returns nullptr.
  OwningOpRef<ModuleOp> constructReductionMatchAndDispatch(
      Location initialLoc, Operation *patternRoot,
      SmallVectorImpl<ArrayRef<transform::MappedValue>> &topLevelMapping,
      SmallVectorImpl<transform::MappedValue> &topLevelMappingStorage) {
    LLVM_DEBUG(DBGS() << "matching reduction: ");
    transform_ext::StructuredOpMatcher reduction, fill, leading, trailing;
    transform_ext::MatchedReductionCaptures captures;
    transform_ext::makeReductionMatcher(reduction, fill, leading, trailing,
                                        captures);
    if (!matchPattern(patternRoot, reduction)) {
      LLVM_DEBUG(llvm::dbgs() << "failed");
      return nullptr;
    }
    LLVM_DEBUG(llvm::dbgs() << "succeeded");

    auto mod = ModuleOp::create(initialLoc);
    OpBuilder b(initialLoc.getContext());
    b.setInsertionPointToEnd(mod.getBody());

    // TODO: change upstream SequenceOp builder callback to handle extra
    // arguments.
    auto sequence = b.create<transform::SequenceOp>(
        initialLoc, TypeRange(), transform::FailurePropagationMode::Propagate,
        b.getType<transform::AnyOpType>(), [](OpBuilder &, Location, Value) {});
    auto [reductionH, fillH, leadingH, trailingH] =
        addArgumentsAndUnpack<4>(initialLoc, sequence);

    {
      b.setInsertionPointToStart(sequence.getBodyBlock());
      ImplicitLocOpBuilder builder(initialLoc, b);
      buildReductionDispatch(builder, leadingH, fillH, reductionH, trailingH);
      b.create<transform::YieldOp>(initialLoc);
    }

    // Bind matched operations to sequence arguments.
    addOptionalMappedOps(leading, topLevelMapping, topLevelMappingStorage);
    addMappedOps(fill, topLevelMapping, topLevelMappingStorage);
    addMappedOps(reduction, topLevelMapping, topLevelMappingStorage);
    addOptionalMappedOps(trailing, topLevelMapping, topLevelMappingStorage);

    return mod;
  }

  void runOnOperation() override {
    // Collect (top-level) Linalg ops at which matchers can be rooted.
    SmallVector<linalg::LinalgOp> possiblePatternRoots;
    getOperation().walk<WalkOrder::PreOrder>([&](linalg::LinalgOp linalgOp) {
      possiblePatternRoots.push_back(linalgOp);
      return WalkResult::skip();
    });

    for (linalg::LinalgOp linalgOp : possiblePatternRoots) {
      // If an external transform module was parsed, use it. Otherwise,
      // match different cases in order of priority and construct the
      // corresponding transform dialect IR if matched. Note that we
      // shouldn't mutate the actual shared transform module at this point,
      // so create a new one if necessary.
      // TODO: figure out a better location story.
      std::shared_ptr<OwningOpRef<ModuleOp>> transformModule(
          getSharedTransformModule());
      SmallVector<ArrayRef<transform::MappedValue>> topLevelMapping;
      SmallVector<transform::MappedValue> topLevelMappingStorage;
      if (!transform::detail::hasSharedTransformModuleImpl(transformModule))
        transformModule = std::make_shared<OwningOpRef<ModuleOp>>(
            constructReductionMatchAndDispatch(linalgOp.getLoc(), linalgOp,
                                               topLevelMapping,
                                               topLevelMappingStorage));

      // If nothing matched on the current op and there was no external
      // input, advance to the next op without attempting to transform.
      if (!transform::detail::hasSharedTransformModuleImpl(transformModule))
        continue;

      if (failed(mlir::verify(transformModule.get()->get())))
        return signalPassFailure();

      if (failed(transform::detail::interpreterBaseRunOnOperationImpl(
              linalgOp, topLevelMapping, getArgument(), transformModule,
              transformFileName, debugPayloadRootTag, debugTransformRootTag)))
        return signalPassFailure();
    }
  }

 private:
  Statistic numDispatches{this, "number of dispatches",
                          "Number of Flow dispatches created"};
};
}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDispatchWithTransformDialect(StringRef transformFileName,
                                   StringRef debugPayloadRootTag,
                                   StringRef debugTransformRootTag,
                                   bool debugEmitRemarkOnMatch) {
  return std::make_unique<DispatchWithTransformDialect>(
      transformFileName, debugPayloadRootTag, debugTransformRootTag,
      debugEmitRemarkOnMatch);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
