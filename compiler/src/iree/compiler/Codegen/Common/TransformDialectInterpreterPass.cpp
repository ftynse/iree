// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterUtils.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"

#define DEBUG_TYPE "iree-transform-dialect-interpreter"
#define DEBUG_TYPE_DUMP_STDERR "iree-transform-dialect-dump-repro"
#define DEBUG_TYPE_DUMP_FILE "iree-transform-dialect-save-repro"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for codegen.
/// This needs to be its own pass because the registration mechanism and ops
/// available are different than for other interpreters.
class TransformDialectInterpreterPass
    : public iree_compiler::TransformDialectInterpreterBase<
          TransformDialectInterpreterPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: this is only necessary to make registry subset happy when running
    // the lowering to LLVM. The lowering should be changed to stop using the
    // nested pass manager and this will go away.

    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    mlir::iree_compiler::IREE::Flow::FlowDialect,
                    arith::ArithDialect,
                    AffineDialect,
                    bufferization::BufferizationDialect,
                    func::FuncDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    linalg::transform::LinalgTransformDialect,
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect,
                    vector::VectorDialect
        // clang-format on
        >();

    // TODO: these should be registered by the extension instead, but there is
    // no support for it in core currently.
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);

    registry.addExtensions<
        mlir::iree_compiler::IREE::LinalgExt::LinalgExtTransformOpsExtension,
        transform_ext::StructuredTransformOpsExtension>();
    iree_compiler::registerTransformDialectCommonExtension(registry);
    iree_compiler::registerTransformDialectFlowExtension(registry);
    iree_compiler::registerTransformDialectLLVMCPUExtension(registry);
    iree_compiler::registerTransformDialectLLVMGPUExtension(registry);
    linalg::registerTransformDialectExtension(registry);
  }

  TransformDialectInterpreterPass(StringRef transformFileName = StringRef(),
                                  StringRef payloadRootTag = StringRef(),
                                  StringRef transformRootTag = StringRef()) {
    this->transformFileName = transformFileName.str();
    this->payloadRootTag = payloadRootTag.str();
    this->transformRootTag = transformRootTag.str();
  }
  TransformDialectInterpreterPass(const TransformDialectInterpreterPass &pass) {
    this->transformFileName = pass.transformFileName;
    this->payloadRootTag = pass.payloadRootTag;
    this->transformRootTag = pass.transformRootTag;
    // TODO: if we really don't like shared_ptr, we could also clone the
    // transformModule here.
    sharedTransformModule = pass.sharedTransformModule;
  }

  LogicalResult initialize(MLIRContext *context) override {
    OwningOpRef<ModuleOp> module;
    if (failed(transform::parseTransformModuleFromFile(
            context, transformFileName, module)))
      return failure();

    sharedTransformModule =
        std::make_shared<OwningOpRef<ModuleOp>>(std::move(module));
    return success();
  }

  void runOnOperation() override {
    Operation *target = getOperation();
    bool parsedTransform = (sharedTransformModule && *sharedTransformModule);

    Region *transformRegion = nullptr;
    if (!parsedTransform) {
      Operation *topLevelTransform = findTopLevelTransform(target);
      if (!topLevelTransform) return signalPassFailure();
      transformRegion = topLevelTransform->getParentRegion();
    } else {
      transformRegion = &(*sharedTransformModule)->getRegion();
    }
    assert(transformRegion && "unexpected detached root transform op");

    Operation *payloadRoot = target;
    if (payloadRootTag.empty()) {
      target->setAttr(
          kTransformIreeTagAttrName,
          StringAttr::get(&getContext(), kTransformIreeTagPayloadRootValue));
    } else {
      payloadRoot = findOpWithTag(target, payloadRootTag);
      if (!payloadRoot) {
        target->emitError()
            << "couldn't find the root payload op with "
            << kTransformIreeTagAttrName << "=\""
            << kTransformIreeTagPayloadRootValue << "\" attribute";
        return signalPassFailure();
      }
    }

    if (transformRootTag.empty()) {
      transformRegion->getParentOp()->setAttr(
          kTransformIreeTagAttrName,
          StringAttr::get(&getContext(),
                          kTransformIreeTagTransformContainerValue));
    } else {
      Operation *transformRoot =
          findOpWithTag(transformRegion->getParentOp(),
                        kTransformIreeTagTransformContainerValue);
      if (!transformRoot) {
        transformRegion->getParentOp()->emitError()
            << "couldn't find the transform container op with "
            << kTransformIreeTagAttrName << "=\""
            << kTransformIreeTagTransformContainerValue << "\" attribute";
        return signalPassFailure();
      }
      if (transformRoot->getNumRegions() != 1 ||
          !transformRoot->getRegion(0).hasOneBlock()) {
        transformRoot->emitError() << "expected transform container op to have "
                                      "one single-block region";
        return signalPassFailure();
      }
      transformRegion = &transformRoot->getRegion(0);
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_STDERR, {
      Operation *root = getRootOperation(target);
      llvm::dbgs() << "=== Transform Interpreter Repro ===\n";
      printIreeOptReproCall(llvm::dbgs() << "cat <<EOF | ",
                            root->getName().getStringRef());
      printModuleForRepro(llvm::dbgs(), root, transformRegion->getParentOp());
      llvm::dbgs() << "\nEOF\n";
      llvm::dbgs() << "===================================\n";
    });

    DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_FILE, {
      saveReproToTempFile(llvm::dbgs(), target, transformRegion->getParentOp());
    });

    if (payloadRootTag.empty()) {
      target->removeAttr(kTransformIreeTagAttrName);
    }
    if (transformRootTag.empty()) {
      transformRegion->getParentOp()->removeAttr(
          kTransformIreeTagTransformContainerValue);
    }

    // TODO: lift this assertion.
    assert(transformRegion->getBlocks().size() == 1 &&
           "expected single-region block");
    if (failed(transform::applyTransformsInRegion(*transformRegion,
                                                  payloadRoot))) {
      payloadRoot->emitOpError() << "transform dialect interpreter failed";
      return signalPassFailure();
    }
  }

 private:
  /// Finds the single top-level transform operation with `root` as ancestor.
  /// Reports an error if there is more than one such operation and returns the
  /// first one found. Reports an error returns nullptr if no such operation
  /// found.
  Operation *findTopLevelTransform(Operation *root) {
    transform::TransformOpInterface topLevelTransform = nullptr;
    WalkResult walkResult = root->walk<WalkOrder::PreOrder>(
        [&](transform::TransformOpInterface transformOp) {
          if (!topLevelTransform) {
            topLevelTransform = transformOp;
            return WalkResult::skip();
          }
          auto diag = transformOp.emitError()
                      << "more than one top-level transform op";
          diag.attachNote(topLevelTransform.getLoc())
              << "previous top-level transform op";
          return WalkResult::interrupt();
        });
    if (walkResult.wasInterrupted()) return nullptr;
    if (!topLevelTransform) {
      auto diag = root->emitError()
                  << "could not find a nested top-level transform op";
      diag.attachNote() << "use the '" << transformFileName.getArgStr()
                        << "' option to provide transform as external file";
      return nullptr;
    }
    return topLevelTransform;
  }

  /// Name of the attribute used for targeting the transform dialect interpreter
  /// at specific operations.
  constexpr static llvm::StringLiteral kTransformIreeTagAttrName =
      "transform.iree_tag";
  /// Value of the attribute indicating the root payload operation.
  constexpr static llvm::StringLiteral kTransformIreeTagPayloadRootValue =
      "iree_payload_root";
  /// Value of the attribute indicating the container of transform operations
  /// (containing the top-level transform operation).
  constexpr static llvm::StringLiteral
      kTransformIreeTagTransformContainerValue = "iree_transform_container";

  /// Finds an operation nested in `root` that has the transform dialect tag
  /// attribute with the value specified as `tag`. Assumes only one operation
  /// may have the tag. Returns nullptr if there is no such operation.
  static Operation *findOpWithTag(Operation *root, StringRef tag) {
    Operation *found = nullptr;
    root->walk<WalkOrder::PreOrder>([tag, &found](Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>(kTransformIreeTagAttrName);
      if (!attr || attr.getValue() != tag) return WalkResult::advance();

      assert(found == nullptr && "more than one op with the same tag");
      found = op;

      // In debug mode, continue the traversal to see if the tag is not
      // duplicated.
#ifndef NDEBUG
      return WalkResult::advance();
#else
      return WalkResult::interrupt();
#endif  // NDEBUG
    });
    return found;
  }

  /// Returns the ancestor of `target` that doesn't have a parent.
  Operation *getRootOperation(Operation *target) {
    Operation *root = target;
    while (root->getParentOp()) root = root->getParentOp();
    return root;
  }

  /// Prints the CLI command running the repro with the current path.
  llvm::raw_ostream &printIreeOptReproCall(llvm::raw_ostream &os,
                                           StringRef rootOpName) {
    os << llvm::formatv(
        "iree-opt "
        "--pass-pipeline=\"{0}(iree-transform-dialect-interpreter{{{1}={2} "
        "{3}={4}})\"",
        rootOpName, payloadRootTag.getArgStr(),
        payloadRootTag.empty() ? StringRef(kTransformIreeTagPayloadRootValue)
                               : payloadRootTag,
        transformRootTag.getArgStr(),
        transformRootTag.empty()
            ? StringRef(kTransformIreeTagTransformContainerValue)
            : transformRootTag);
    return os;
  }

  /// Prints the module rooted at `root` to `os` and appends
  /// `transformContainer` if it is not nested in `root`.
  llvm::raw_ostream &printModuleForRepro(llvm::raw_ostream &os, Operation *root,
                                         Operation *transformContainer) {
    root->print(os);
    if (!root->isAncestor(transformContainer)) {
      transformContainer->print(os);
    }
    return os;
  }

  /// Saves the payload and the transform IR into a temporary file and reports
  /// the file name to `os`.
  void saveReproToTempFile(llvm::raw_ostream &os, Operation *target,
                           Operation *transformContainer) {
    using llvm::sys::fs::TempFile;
    Operation *root = getRootOperation(target);

    SmallVector<char, 128> tmpPath;
    llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, tmpPath);
    llvm::sys::path::append(tmpPath, "iree_transform_dialect_%%%%%%.mlir");
    llvm::Expected<TempFile> tempFile = TempFile::create(tmpPath);
    if (!tempFile) {
      os << "could not open temporary file to save the repro\n";
      return;
    }

    llvm::raw_fd_ostream fout(tempFile->FD, /*shouldClose=*/false);
    printModuleForRepro(fout, root, transformContainer);
    fout.flush();
    std::string filename = tempFile->TmpName;

    if (tempFile->keep()) {
      os << "could not preserve the temporary file with the repro\n";
      return;
    }

    os << "=== Transform Interpreter Repro ===\n";
    printIreeOptReproCall(os, root->getName().getStringRef())
        << " " << filename << "\n";
    os << "===================================\n";
  }

  // The parsed transform module to be used for transformations.
  // TODO: Figure a better way to build a transform module and transport it in
  // the proper places in the IR as it is transformed by IREE so that it is
  // available with better ownership semantics.
  // Note: we wrap the OwningOpRef to get the desired destruction mechanism.
  // Note: shared_ptr is not great but we know the sharedTransformModule is
  // readonly.
  // Alternatives comprise:
  //   1. no shared_ptr but copying the module with every pass clone that the
  //      OpPassManager decides to perform.
  //   2. lifting ownership of the parsed transform module higher up in the
  //      IREE stack. This may be only shift the problem as we have passes
  //      building pass managers in IREE.
  //   3. build better support to embed the transformation module in the
  //      input IR and transport it to the place of use in IREE. This is deemed
  //      too intrusive atm.
  //   4. (future) config/resources mechanism that is being proposed in core?
  std::shared_ptr<OwningOpRef<ModuleOp>> sharedTransformModule;
};
}  // namespace

namespace mlir {
namespace iree_compiler {
/// Create a Transform dialect interpreter pass.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName, llvm::StringRef payloadRootTag,
    llvm::StringRef transformRootTag) {
  return std::make_unique<TransformDialectInterpreterPass>(
      transformFileName, payloadRootTag, transformRootTag);
}
}  // namespace iree_compiler
}  // namespace mlir
