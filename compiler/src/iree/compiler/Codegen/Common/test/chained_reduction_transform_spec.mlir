// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks
  %0:7 = transform.iree.match_callback failures(propagate) "chained_reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
//  transform.iree.emit_remark "leading_1" at %0#0 : !pdl.operation
//  transform.iree.emit_remark "fill_1" at %0#1 : !pdl.operation
//  transform.iree.emit_remark "reduction_1" at %0#2 : !pdl.operation
//  transform.iree.emit_remark "middle" at %0#3 : !pdl.operation
//  transform.iree.emit_remark "fill_2" at %0#4 : !pdl.operation
//  transform.iree.emit_remark "reduction_2" at %0#5 : !pdl.operation
//  transform.iree.emit_remark "trailing_2" at %0#6 : !pdl.operation

  %first, %rest = transform.iree.take_first %0#6, %0#5 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  %foreach_thread_op, %tiled_op = transform.structured.tile_to_foreach_thread_op %first   num_threads [] tile_sizes [1,1](mapping = [#gpu.block<x>, #gpu.block<y>])
  %1 = transform.structured.fuse_into_containing_op %rest into %foreach_thread_op
  %2 = transform.structured.fuse_into_containing_op %0#4 into %foreach_thread_op
  %3 = transform.structured.fuse_into_containing_op %0#3 into %foreach_thread_op
  %4 = transform.structured.fuse_into_containing_op %0#2 into %foreach_thread_op
  %5 = transform.structured.fuse_into_containing_op %0#1 into %foreach_thread_op
  %leading = transform.structured.fuse_into_containing_op %0#0 into %foreach_thread_op

  %first_0, %rest_1 = transform.iree.take_first %1, %tiled_op : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
  %foreach_thread_op_2, %fill_op, %split_linalg_op, %combining_linalg_op = 
    transform.structured.tile_reduction_using_foreach_thread %first_0 by 
    num_threads = [0, 0, 256], tile_sizes = [0, 0, 4], mapping = [#gpu.thread<x>]
  transform.structured.fuse_into_containing_op %fill_op into %foreach_thread_op_2

  %foreach_thread_op_2p, %fill_op_p, %split_linalg_op_p, %combining_linalg_op_p = 
    transform.structured.tile_reduction_using_foreach_thread %4 by 
    num_threads = [0, 0, 256], tile_sizes = [0, 0, 4], mapping = [#gpu.thread<x>]
  transform.structured.fuse_into_containing_op %fill_op_p into %foreach_thread_op_2p

  %foreach_thread_second, %fill_op_second, %split_linalg_op_second, %combining_linalg_op_second =
    transform.structured.tile_reduction_using_foreach_thread %combining_linalg_op by
    num_threads = [0, 0, 64], tile_sizes = [0, 0, 4], mapping = [#gpu.thread<x>]
  transform.structured.fuse_into_containing_op %fill_op_second into %foreach_thread_second
  transform.structured.tile_to_foreach_thread_op %combining_linalg_op_second
    num_threads [] tile_sizes[1](mapping = [#gpu.thread<y>])

  %foreach_thread_p_second, %fill_op_p_second, %split_linalg_op_p_second, %combining_linalg_op_p_second =
    transform.structured.tile_reduction_using_foreach_thread %combining_linalg_op_p by
    num_threads = [0, 0, 64], tile_sizes = [0, 0, 4], mapping = [#gpu.thread<x>]
  transform.structured.fuse_into_containing_op %fill_op_p_second into %foreach_thread_p_second
  transform.structured.tile_to_foreach_thread_op %combining_linalg_op_p_second
    num_threads [] tile_sizes[1](mapping = [#gpu.thread<y>])

  %tiled_middle, %loop_middle = transform.structured.tile %3[0, 0, 256]
  transform.structured.tile_to_foreach_thread_op %tiled_middle
    num_threads [0, 0, 256] tile_sizes [](mapping = [#gpu.thread<x>])

  %tiled_trailing, %loop_trailing = transform.structured.tile %rest_1[0, 0, 256]
  transform.structured.tile_to_foreach_thread_op %tiled_trailing
    num_threads [0, 0, 256] tile_sizes [](mapping = [#gpu.thread<x>])
  
  %6 = transform.structured.match ops{["func.func"]} in %arg0
  %7 = transform.iree.apply_patterns %6 {rank_reducing}
  %8 = transform.structured.vectorize %7
  %9 = transform.structured.match ops{["func.func"]} in %arg0
  %10 = transform.iree.apply_patterns %9 {fold_reassociative_reshapes}
  %11 = transform.iree.eliminate_empty_tensors %arg0
  %12 = transform.iree.bufferize {target_gpu} %11
  %13 = transform.structured.match ops{["func.func"]} in %12
  %14 = transform.iree.erase_hal_descriptor_type_from_memref %13
  %15 = transform.structured.match ops{["func.func"]} in %12
  %16 = transform.iree.foreach_thread_to_workgroup %15
  %17 = transform.iree.map_nested_foreach_thread_to_gpu_threads %16 {workgroup_size = [256, 1, 1]}
  %18 = transform.iree.apply_patterns %17 {fold_memref_aliases, rank_reducing}
  %19 = transform.structured.match ops{["scf.if"]} in %18
  sequence %12 : !pdl.operation failures(suppress) {
  ^bb0(%arg1: !pdl.operation):
    %20 = transform.iree.vector.to_warp_execute_on_lane_0 %19 {warp_size = 32 : i64}
  }
  transform.iree.vector.warp_distribute %18
}
