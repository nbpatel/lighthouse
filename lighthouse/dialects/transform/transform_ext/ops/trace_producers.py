from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure
from collections import deque

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.ops.filter_reduction_ops import (
    is_reduction_op,
)
from lighthouse.utils.mlir import defining_op


class TraceProducersOp(TransformExtensionDialect.Operation, name="trace_producers"):
    """
    Collect all ops in the SSA producer chain of the target op.

    Args:
        target: Handle to a single op.
        stop_at_reductions: Do not walk PAST a reduction op. The reduction itself is still
            returned, but its own operands are not visited. Off by default, i.e. the walk
            covers the whole producer graph.
    Returns:
        Handles to all producer ops, ordered closest-first.
    """

    target: ext.Operand[transform.AnyOpType]
    stop_at_reductions: ir.IntegerAttr
    ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "TraceProducersOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure

            leaf = target_ops[0]
            stop_at_reductions = bool(ir.IntegerAttr(op.stop_at_reductions).value)
            # Walk the SSA producer graph via operand -> owner edges.
            # Use BFS to guarantee closest-first ordering by graph distance.
            producers: list[ir.Operation] = []
            visited: set[ir.Operation] = set()
            worklist = deque()

            for operand in leaf.operands:
                owner_op = defining_op(operand)
                if owner_op is not None and owner_op not in visited:
                    visited.add(owner_op)
                    worklist.append(owner_op)

            while worklist:
                producer = worklist.popleft()
                producers.append(producer)

                # A reduction is a BARRIER when asked for: the ops feeding it compute the
                # reduced value, so they belong to the reduction's own loop, not to the loop
                # of whatever consumes the reduction's result. Walking past it hands callers
                # producers that have no use in the container they are fusing into.
                if stop_at_reductions and is_reduction_op(producer):
                    continue

                for operand in producer.operands:
                    owner_op = defining_op(operand)
                    if owner_op is not None and owner_op not in visited:
                        visited.add(owner_op)
                        worklist.append(owner_op)

            results.set_ops(op.ops, producers)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "TraceProducersOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def trace_producers(
    target: ir.Value[transform.AnyOpType],
    stop_at_reductions: bool = False,
) -> ir.Value:
    """
    snake_case wrapper to create a TraceProducersOp.

    Args:
        target: Handle to a single op.
        stop_at_reductions: Do not walk PAST a reduction op (the reduction itself is still
            returned). Use this when the producers are about to be fused into a loop that
            consumes the reduction's RESULT: everything on the reduction's input side has no
            use in that loop, and `fuse_into_containing_op` fails with "could not find next
            producer to fuse into container" if it is handed one.
    Returns:
        Handles to all producer ops, ordered closest-first.
    """
    return TraceProducersOp(
        target=target,
        stop_at_reductions=ir.IntegerAttr.get(
            ir.IntegerType.get_signless(1), int(stop_at_reductions)
        ),
    ).ops
