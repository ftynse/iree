// Compile:
//
// ./build/tools/iree-compile bmm_set_schedule.mlir \
//   -iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//   --iree-flow-enable-pad-handling \
//   --iree-codegen-llvmgpu-enable-transform-dialect-jit=1 \
//   --iree-codegen-llvmgpu-enable-transform-dialect-batch-matmul-strategy
//
// See the script and strategy:
//
// ./build/tools/iree-opt bmm_set_schedule.mlir  -iree-hal-target-backends=cuda \
//   --iree-hal-cuda-llvm-target-arch=sm_80  --iree-abi-transformation-pipeline \
//   --iree-flow-transformation-pipeline --iree-stream-transformation-pipeline \
//   -iree-hal-configuration-pipeline --iree-flow-enable-pad-handling |\
// ./build/tools/iree-opt
//   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))' \
//   --iree-codegen-llvmgpu-enable-transform-dialect-jit=1 \
//   --debug-only=iree-transform-builder \
//   --iree-codegen-llvmgpu-enable-transform-dialect-batch-matmul-strategy

!lhs = tensor<128x80x32xf32>
!rhs = tensor<128x32x320xf32>
!res = tensor<128x80x320xf32>

func.func @batch_matmul(%arg0: !lhs, %arg1: !rhs, %arg2: !res) -> !res {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : !res
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !res) -> !res
  %2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : !lhs, !rhs) outs(%1 : !res) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %3 = arith.mulf %arg3, %arg4 : f32
    %4 = arith.addf %arg5, %3 : f32
    linalg.yield %4 : f32
  } -> !res
  return %2 : !res
}

//transform.sequence failures(propagate) {
//^bb0(%arg0: !transform.any_op):
//  transform.iree.register_match_callbacks
//  %0:2 = transform.iree.match_callback failures(propagate) "batch_matmul"(%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//  transform.iree.emit_remark "fill" at %0#0 : !transform.any_op
//  transform.iree.emit_remark "matmul" at %0#1 : !transform.any_op
//}
