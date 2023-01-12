// RUN: iree-opt %s

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  transform.iree.register_match_callbacks
  %0:7 = transform.iree.match_callback failures(propagate) "chained_reduction"(%arg0)
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
  %region = transform.iree.wrap_in_dispatch_region %0#6 { generateWorkload = false}
  %region2 = transform.iree.move_preceding_op_into_dispatch_region %0#5 into %region
  %region3 = transform.iree.move_preceding_op_into_dispatch_region %0#4 into %region2
  %region4 = transform.iree.move_preceding_op_into_dispatch_region %0#3 into %region3
  %region5 = transform.iree.move_preceding_op_into_dispatch_region %0#2 into %region4
  %region6 = transform.iree.move_preceding_op_into_dispatch_region %0#1 into %region5
  %region7 = transform.iree.move_preceding_op_into_dispatch_region %0#0 into %region6
  transform.iree.region_to_workgroups %region7
}

