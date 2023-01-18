// RUN: iree-opt %s

// Dispatch softmax.
//
// This is quite hacky because it uses the obsolete approach with matching a
// fixed number of Linalg operations in a function.
// TODO: use the proper matcher (callback) for softmax.
//
// In order to work at a function level instead of the individual op level,
// this gets the closest isolated parent, which is a function, and checks
// whether it already has "flow.dispatch.tensor.store" as an indication that
// dispatch region formation has already happened. It then attempts to
// split that into 0 handles to trigger a silenceable error when a store
// was matched. The inner sequence propagates silenceable errors to stop
// further transformation. The additional outer sequence suppressed them
// so this intended failure-to-split doesn't get reported to the user and
// stop the pass.
transform.sequence failures(suppress) {
^bb0(%root: !transform.any_op):
  transform.sequence  %root : !transform.any_op failures(propagate) {
  ^bb1(%op: !transform.any_op):
    %variant_op = transform.get_closest_isolated_parent %op : (!transform.any_op) -> !transform.any_op

    %flow = transform.structured.match ops{["flow.dispatch.tensor.store"]} in %variant_op
    transform.split_handle %flow : (!transform.any_op) -> ()
    // transform.print %flow : !transform.any_op

    %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
      in %variant_op

    %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum, %div =
    transform.split_handle %ops
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op,
                              !transform.any_op, !transform.any_op, !transform.any_op)

    /// This must be used with the custom dispatch region formation
    /// because IREE's does not fuse the 6 ops softmax version even with
    /// --iree-flow-enable-aggressive-fusion.
    %region_op = transform.iree.wrap_in_dispatch_region %div { generateWorkload = false }

    %non_div = transform.merge_handles %input_max_fill, %input_max, %exps_sum_fill, %exps, %exps_sum
      : !transform.any_op
    %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_div into %region_op

    %empty = transform.structured.match ops{["tensor.empty"]} in %variant_op
    %region_op_3 = transform.iree.move_preceding_op_into_dispatch_region %empty into %region_op_2
    transform.iree.region_to_workgroups %region_op_3
  }
}
