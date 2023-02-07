// RUN: iree-opt %s --iree-transform-dialect-interpreter --split-input-file --verify-diagnostics

!input_tensor_t = tensor<127x47x16x16xf32>
!weight_tensor_t = tensor<16x47x3x3xf32>
!output_tensor_t = tensor<127x16x14x14xf32>
func.func @conv_2d_nchw_fchw(%arg0: !input_tensor_t, %arg2: !output_tensor_t) 
  -> !output_tensor_t 
{
  %c0 = arith.constant dense<0.1> : !weight_tensor_t
  // expected-remark @below {{matched}}
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %c0: !input_tensor_t, !weight_tensor_t)
    outs(%arg2: !output_tensor_t) -> !output_tensor_t
  return %0 : !output_tensor_t
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  transform.iree.register_match_callbacks
  %conv, %0:6 = transform.iree.match_callback failures(propagate) "convolution"(%module_op) :
    (!pdl.operation) -> (!transform.op<"linalg.conv_2d_nchw_fchw">, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)

  transform.iree.emit_remark "matched" at %conv : !transform.op<"linalg.conv_2d_nchw_fchw">
  // expected-remark @below {{0 : i64}}
  transform.iree.emit_params_as_remark "batch", %0#0 : !transform.param<i64>
  // expected-remark @below {{4 : i64}}
  transform.iree.emit_params_as_remark "input channels", %0#1 : !transform.param<i64>
  // expected-remark @below {{1 : i64}}
  transform.iree.emit_params_as_remark "output channels", %0#2 : !transform.param<i64>
  // expected-remark @below {{2 : i64 3 : i64}}
  transform.iree.emit_params_as_remark "image", %0#3 : !transform.param<i64>
  // expected-remark @below {{5 : i64}}
  transform.iree.emit_params_as_remark "filter", %0#4 : !transform.param<i64>
  // expected-remark @below {{depth}}
  transform.iree.emit_params_as_remark "depth", %0#5 : !transform.param<i64>
}
