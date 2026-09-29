"""Generic XeGPU schedule for a torch-mlir-generated payload.

This is the schedule half of the torch-mlir Llama-3 flow, and the counterpart to the
hand-written `llama3_schedule.py`. The difference: `llama3_schedule.py` is told what each
op is, via a `kinds` list maintained alongside the hand-written payload. Here nothing is
told to us -- `classify_payload` derives the whole plan by walking the imported IR, so any
torch-mlir payload can be scheduled.

Structure of the flow (mirrors the nanoGPT payload/schedule/driver split):
    payload   <- torch-mlir, from the PyTorch model in llama3_torch_model.py
    schedule  <- this file
    driver    <- llama3_torch_mlir_check.py (CPU oracle) / torch_mlir_ffn_gpu.py (GPU)

How the schedule works:

1. `classify_payload` walks the payload function and returns an ordered plan, one entry
   per GPU kernel. Each entry names its op class and the ops it owns (`members`).
   Op classes are derived, not declared:
     matmul / batch_matmul   -- named contraction ops
     contraction             -- a contraction torch-mlir emitted as a linalg.generic
     reduction               -- reduces a single input (RMSNorm row sum, softmax max/sum)
     elementwise             -- every iterator parallel
   A reduction ITERATOR is not enough to tell a reduction from a contraction; see
   `_classify_generic`.

2. Ops are GROUPED into kernels by dataflow: an elementwise op joins the preceding entry
   when it consumes that entry's result. This is what puts a cast/activation in its
   producer's kernel (Inductor's F1/F2/F4 shape) and what makes RMSNorm one kernel (which
   for a reduction is mandatory, not an optimization -- its reduced intermediate cannot be
   a kernel output).

   NOTE: we deliberately do NOT run the global `linalg-fuse-elementwise-ops` pass.
   It does reduce op count, but it absorbs neighbours INTO contractions and rewrites them
   as generics -- on the full Llama block that turned 8 `linalg.matmul` + 2
   `linalg.batch_matmul` into 4 matmul + 6 generic contractions, breaking the DPAS path
   (which matches `linalg.matmul` / `xegpu.dpas`). Fusing through tiling instead is
   layout-aware and preserves op identity.

3. Each group is tiled into one work-group `forall`, in TOPOLOGICAL order, through its
   LAST member, with `fuse_producers=True`. Every part of that is load-bearing:
     - tiling through the last member pulls the group's earlier ops in (so the epilogue
       lands in the producer's kernel);
     - `fuse_producers` also pulls in each contraction's `linalg.fill` accumulator --
       leaving it outside the forall causes a GPU page fault;
     - topological order means an op's producers already sit inside foralls by the time we
       reach it, so they cannot be re-absorbed. Without that ordering, tiling one matmul's
       leaf consumer drags a second matmul into the same forall, producing a 2-dpas kernel
       the single-dpas annotation cannot handle.

4. One shared tail for the whole module (it is op-class-agnostic: it processes every
   `scf.forall`), then per-kernel XeGPU layout anchors. Three wrinkles:
     - the tail is spelled out rather than taken from
       `vectorize_bufferize_and_outline_gpu_func`, so that a reduction can have
       `promote-buffers-to-stack` inserted after bufferization -- without it a spilled
       accumulator becomes a global `gpu.alloc` that lowers to an unassignable scattered
       `xegpu.store`;
     - vectorization is scoped to the foralls, i.e. to the kernels, and must not run over
       the host code around them (see `_vectorize_kernels_only`);
     - vector->xegpu runs per kernel, because only a reduction may have its allocas moved
       to shared local memory; doing that to an elementwise kernel creates `store_matrix`
       paths that fail to lower.

5. A payload may compute its result in PIECES -- RoPE writes a rotated half at a time --
   which reads as `tensor.empty` -> `insert_slice` -> ... -> the output. Empty-tensor
   elimination runs BEFORE tiling to point each piece's producer straight at its slice of
   the output argument; see the loop in `generic_schedule`. `linalg.transpose` and
   `tensor.concat` are the two remaining forms of real data movement with no kernel of
   their own (concat is decomposed to `insert_slice`; transpose is still unhandled).
"""

from mlir import ir
from mlir.dialects import linalg
from mlir.dialects import memref
from mlir.dialects import tensor
from mlir.dialects import transform
from mlir.dialects import vector
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform import bufferization as bufferization_transform
from mlir.dialects.transform import memref as memref_transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import tensor as tensor_transform
from mlir.dialects.transform import vector as vector_transform
from mlir.dialects.transform import xegpu as xegpu_transform
import lighthouse.transform as lh_transform

from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu import XeGPUParameterSelector
from lighthouse.schedule.xegpu.lowering_common import (
    get_payload_func,
    convert_allocs_to_gpu,
    convert_to_gpu_launch,
    outline_gpu_function,
)
from lighthouse.schedule.xegpu.mlp_schedule import xegpu_wg_annotation_for_mlp_layer
from lighthouse.schedule.xegpu.elemwise_schedule import (
    xegpu_wg_annotation_for_elemwise_layer,
)

# The fused-attention XeGPU layouts, reused from the hand schedule in this same directory
# instead of being copied: it is the only place that knows the Q/K/V/K^T/out anchor layouts
# for the flash loop, and duplicating ~70 lines of layout constants would rot.
from llama3_schedule import xegpu_fa_annotation

# Default elementwise geometry. wg_n is deliberately bounded: a row-only tile leaves the
# full column width as one per-subgroup vector, which at FFN width (8192) overruns the
# register file and faults.
EW_PARAMS = {
    "wg_m": 128,
    "wg_n": 256,
    "sg_m": 32,
    "sg_n": 32,
    "load_m": 8,
    "load_n": 16,
}
# Reduction geometry. wg_m/sg_m carry the row split (wg_n=sg_n=1) so the shared tail
# derives (wg_rows/sg_rows)*NB_WORKITEMS threads; rss is the per-subgroup reduction step.
# Mirrors the hand driver's ln_params.
RED_PARAMS = {"wg_m": 64, "sg_m": 8, "wg_n": 1, "sg_n": 1, "rss": 16}
# Fused-attention geometry, for the `d_head == 64` case (Llama-3's head dim). These are the
# published-good values from `kernel_bench.py`'s attention branch, not invented here.
#
# The wg_m/sg_m/wg_n/sg_n quartet is NOT redundant with wg_rows/sg_rows: the shared tail's
# `outline_gpu_function` derives the thread count as
# (wg_m/sg_m)*(wg_n/sg_n)*NB_WORKITEMS, and (128/16)*(1/1)*16 = 128 reproduces exactly the
# `num_subgroups * subgroup_size` the library attention schedule sets by hand. `wg_rows` and
# `sg_rows` are what `xegpu_fa_annotation` reads, and `n_head` is its name for d_head.
FA_PARAMS = {
    "wg_m": 128,
    "sg_m": 16,
    "wg_n": 1,
    "sg_n": 1,
    "wg_rows": 128,
    "sg_rows": 16,
    "n_head": 64,  # d_head; the hand annotation's key name
    "inner_loop_tile_size": 64,
    "causal": False,
}

# Rounds of empty-tensor elimination to run before tiling. One round peels one level of an
# `insert_slice` chain, so this needs to cover the longest slice-assembly in the payload:
# RoPE assembles its result from 2 halves, and GQA/attention reshapes are no deeper.
_EMPTY_ELIM_ROUNDS = 4

# linalg ops that become their own kernel. linalg.fill is excluded on purpose: it is an
# accumulator that must be fused into its consumer.
_TILED_OPS = {
    "linalg.matmul": "matmul",
    "linalg.batch_matmul": "batch_matmul",
    "linalg.transpose": "transpose",
    "linalg.generic": None,  # decided by _classify_generic
}
_CONTRACTION_KINDS = {"matmul", "batch_matmul", "contraction"}

# Metadata-only ops: they reshape or slice a tensor without computing anything, so they
# get no kernel of their own -- but they DO carry dataflow. torch-mlir puts them between a
# projection and its RoPE (`(x@wq).view(T,H,hs)` then half-split slices), so a grouping
# pass that ignores them loses the producer/consumer link and splits what should be one
# kernel. NOTE `linalg.transpose` is deliberately absent: it is REAL data movement, so it
# gets its own kernel (kind "transpose") rather than being treated as a view.
_VIEW_OPS = {
    "tensor.expand_shape",
    "tensor.collapse_shape",
    "tensor.extract_slice",
    "tensor.concat",
    "tensor.insert_slice",
    "tensor.cast",
}


def _classify_generic(op) -> str:
    """Classify a linalg.generic as elementwise / reduction / contraction.

    A reduction iterator alone does not tell these apart, and getting it wrong sends a
    matmul down the RMSNorm path. torch-mlir frequently emits a projection as a generic
    rather than a `linalg.matmul` -- `(x @ wq).view(T,H,hs)` folds the head reshape into
    the contraction, giving a generic with a reduction iterator over a (T,H,hs) result.

    Discriminator: a contraction reduces over a dimension shared by TWO OR MORE inputs
    (it multiplies them together); a true reduction reduces a single input. DPS ops have
    one init per result, so the input count is `operands - results`.
    """
    if "reduction" not in str(op.attributes["iterator_types"]):
        return "elementwise"
    n_inputs = len(op.operands) - len(op.results)
    return "contraction" if n_inputs >= 2 else "reduction"


def get_payload_func_op(mod: ir.Module, func_name: str = "main"):
    """The payload `func.func` as an IR op, for the passes that run in Python.

    `get_payload_func` returns a transform HANDLE; the rewrites that the transform dialect
    has no op for (see `fuse_rope_halves`, `replace_transpose_kernels_with_views`) walk the
    IR directly and need the op itself.
    """
    for op in mod.body.operations:
        if op.operation.name == "func.func" and func_name in str(
            op.attributes["sym_name"]
        ):
            return op
    raise ValueError(f"no func.func {func_name!r}")


def _identity_maps(op) -> bool:
    """True when every one of `op`'s indexing maps is the rank-preserving identity.

    That is what makes an op inlinable into another op's body: one element in, one element
    out, at the same index, so two such ops over the same shape share an iteration space
    exactly rather than merely having the same extents.
    """
    maps = ir.ArrayAttr(op.attributes["indexing_maps"])
    if not len(maps):
        return False
    rank = ir.AffineMapAttr(maps[0]).value.n_dims
    identity = ir.AffineMap.get_identity(rank)
    return all(ir.AffineMapAttr(m).value == identity for m in maps)


def _is_inlinable_elementwise(op, shape) -> bool:
    """A single-result `linalg.generic` that is pure elementwise over `shape`."""
    if op.operation.name != "linalg.generic" or len(op.results) != 1:
        return False
    if list(ir.ShapedType(op.results[0].type).shape) != list(shape):
        return False
    if "reduction" in str(op.attributes["iterator_types"]):
        return False
    if not _identity_maps(op):
        return False
    # An accumulating body (one that reads its `outs` element) cannot be inlined: the
    # merged op supplies a fresh destination, so `%out` would no longer mean the same thing.
    body = op.regions[0].blocks[0]
    return len(list(body.arguments[-1].uses)) == 0


def _elementwise_tree(root, shape):
    """Collect the elementwise generics that compute `root`, plus the values they read.

    Returns `(members, leaves)` with `members` in topological order (producers first) and
    `leaves` the distinct values entering the tree from outside, or None if `root` is not
    produced by an inlinable elementwise generic. An op is only taken as a member when ALL
    of its uses are inside the tree -- otherwise something else reads it, so it has to stay
    a real op and is treated as a leaf instead.
    """
    if not isinstance(root, ir.OpResult):
        return None
    if not _is_inlinable_elementwise(root.owner, shape):
        return None

    # Gather candidates first; membership needs the whole set to test "all uses inside".
    candidates, order, stack = set(), [], [root.owner]
    while stack:
        op = stack.pop()
        if op in candidates:
            continue
        candidates.add(op)
        order.append(op)
        n_in = len(op.operands) - len(op.results)
        for operand in list(op.operands)[:n_in]:
            if isinstance(operand, ir.OpResult) and _is_inlinable_elementwise(
                operand.owner, shape
            ):
                stack.append(operand.owner)

    members = [
        op
        for op in candidates
        if op == root.owner
        or all(u.owner in candidates for v in op.results for u in v.uses)
    ]
    member_set = set(members)
    # Topological order: an op may only be emitted once everything it reads is emitted.
    ordered, emitted = [], set()
    while len(ordered) < len(member_set):
        progressed = False
        for op in members:
            if op in emitted:
                continue
            n_in = len(op.operands) - len(op.results)
            if all(
                not (isinstance(o, ir.OpResult) and o.owner in member_set)
                or o.owner in emitted
                for o in list(op.operands)[:n_in]
            ):
                ordered.append(op)
                emitted.add(op)
                progressed = True
        if not progressed:
            return None  # a cycle cannot happen in SSA; bail rather than loop
    leaves = []
    for op in ordered:
        n_in = len(op.operands) - len(op.results)
        for operand in list(op.operands)[:n_in]:
            produced_here = isinstance(operand, ir.OpResult) and operand.owner in member_set
            if not produced_here and operand not in leaves:
                leaves.append(operand)
    return ordered, leaves


def _inline_elementwise_body(op, operand_scalars, scalars):
    """Clone `op`'s body into the current insertion point, returning the scalar it yields.

    `operand_scalars` are the scalars standing in for the op's tensor inputs, in order;
    `scalars` maps an already-inlined member's RESULT to the scalar that computed it. The
    bindings have no value-remapping clone, so each body op is recreated by name with
    rewritten operands -- fine here because an inlinable body holds only scalar arithmetic.
    """
    body = op.regions[0].blocks[0]
    local = dict(scalars)
    for i, scalar in enumerate(operand_scalars):
        local[body.arguments[i]] = scalar
    for inner in body.operations:
        if inner.operation.name == "linalg.yield":
            return local.get(inner.operands[0], inner.operands[0])
        # Indexing the attribute map gives NamedAttribute; ITERATING it gives bare names.
        attrs = {
            (na := inner.attributes[i]).name: na.attr
            for i in range(len(inner.attributes))
        }
        clone = ir.Operation.create(
            inner.operation.name,
            results=[r.type for r in inner.results],
            operands=[local.get(o, o) for o in inner.operands],
            attributes=attrs,
        )
        for old, new in zip(inner.results, clone.results):
            local[old] = new
    raise ValueError(f"{op.operation.name} body has no linalg.yield")


def _static_unit_slice(ins):
    """`(offsets, sizes)` of a slice op as ints if fully static with unit strides, else None."""
    kdyn = ir.ShapedType.get_dynamic_size()

    def ints(name):
        # These are DenseI64ArrayAttr (`array<i64: 0, 0>`), not ArrayAttr.
        return [int(v) for v in ir.DenseI64ArrayAttr(ins.attributes[name])]

    offsets, sizes, strides = (
        ints("static_offsets"),
        ints("static_sizes"),
        ints("static_strides"),
    )
    if any(v == kdyn for v in offsets + sizes) or any(st != 1 for st in strides):
        return None
    return offsets, sizes


def _sibling_slice_chain(op, block_ops):
    """Match a chain of two or more `tensor.insert_slice`s that write static, unit-stride,
    pairwise-DISJOINT, equally shaped slices of one destination. `op` is the candidate LAST
    insert; the chain is followed backwards through each insert's destination. Returns
    `(chain, shape)` with `chain` in program order (the first insert's destination is the
    common base, each next one's destination is the previous one's result), or None.

    RoPE's two half-writes are the instance that motivated this, but nothing here is
    RoPE-specific: any rank, any number of siblings, any slice positions.
    """
    if op.operation.name != "tensor.insert_slice":
        return None
    chain, cur = [op], op
    while True:
        dest = cur.operands[1]
        if not isinstance(dest, ir.OpResult):
            break
        prev = dest.owner
        if prev.operation.name != "tensor.insert_slice" or prev not in block_ops:
            break
        # An intermediate may only be read by the next insert and by `tensor.extract_slice`s
        # -- after empty-tensor elimination there is one of those per later sibling,
        # supplying its `outs` (that is the false serialization this rewrite exists to
        # break). Anything else reading the half-assembled tensor would observe a value the
        # merge does not preserve.
        if any(
            u.owner != cur and u.owner.operation.name != "tensor.extract_slice"
            for u in dest.uses
        ):
            break
        chain.append(prev)
        cur = prev
    chain.reverse()
    if len(chain) < 2:
        return None
    boxes = [_static_unit_slice(ins) for ins in chain]
    if any(b is None for b in boxes):
        return None
    shape = boxes[0][1]
    if any(b[1] != shape for b in boxes):
        return None
    rank = len(shape)
    if any(ir.ShapedType(ins.operands[0].type).rank != rank for ins in chain):
        return None
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            oi, oj = boxes[i][0], boxes[j][0]
            if not any(
                oi[d] + shape[d] <= oj[d] or oj[d] + shape[d] <= oi[d] for d in range(rank)
            ):
                return None  # overlapping: the later write wins, a merge would not keep that
    return chain, shape


def fuse_sibling_slice_writers(mod: ir.Module, func_name: str = "main") -> int:
    """Merge sibling elementwise chains that write disjoint slices of one destination into ONE
    multi-result `linalg.generic`. The motivating instance is RoPE, whose two half-chains this
    turns into the hand payload's `Builder.rope` form; the matcher itself is shape-agnostic.

    The RoPE story, kept for the record:

    `_rope` computes `out[:, :half] = lo*cos - hi*sin` and `out[:, half:] = hi*cos + lo*sin`.
    torch-mlir emits that as two independent elementwise chains writing DISJOINT COLUMN
    SLICES of one destination, which costs 2 kernels per RoPE against Inductor's 1 -- the
    last item of the kernel-count gap, and the one no payload spelling fixes (`cat` becomes
    `tensor.concat`, which is data movement with no tiling interface; `flip` becomes a
    `tensor.extract` GATHER, which does not block-load; `stack` becomes `concat` again --
    all three measured, all worse).

    GROUPING cannot fix it either, which is why it outlived the MERGE rule: a group is tiled
    through its LAST member with `fuse_producers`, and the lower half is not a producer of
    the upper one, so it would be left outside every kernel. Nor can the two tiled `forall`s
    be fused as siblings: the upper half's `shared_outs` chains through the lower half's
    `tensor.insert_slice`, so `transform.loop.fuse_sibling` rejects the pair on dominance --
    a FALSE dependence, since the two write disjoint columns.

    So build the form the HAND payload already uses (`Builder.rope`): ONE generic with TWO
    results, each going to its own destination slice. Both halves read the same leaves over
    the same `(rows, half)` iteration space, so the merged op is ordinary elementwise and
    tiles like any other -- no reversal map anywhere, hence no gather.

    Runs on the tensor-level IR AFTER empty-tensor elimination (so the destination is already
    rooted at the real output buffer) and BEFORE `classify_payload`, so the plan simply sees
    one op and needs no special case. Returns the number of RoPEs fused.

    **MUST BE PAIRED WITH `redirect_staged_destination_copies`** (the driver flag does both).
    One-shot bufferization will not write both results in place: it cannot prove that
    `%dst[0, 0][rows, half]` and `%dst[0, half][rows, half]` are disjoint SUBSETS of one
    buffer, so it keeps the first in place and stages the second through an alloc bracketed by
    host `memref.copy`s -- host accesses over device memory, which fault (masked by libocloc as
    "longjmp causes uninitialized stack frame"). That companion pass undoes the staging after
    bufferization. The hand payload never hits this because it writes its two strided
    destination views at MEMREF level and never asks the analysis to prove anything.

    MEASURED DEAD ENDS, do not retry:
      * `transform.loop.fuse_sibling` on the two TILED foralls. Bounds match, but `isOpSibling`
        rejects the pair -- the upper half's `shared_outs` chains through the lower half's
        `tensor.insert_slice`, so a user of the target's result is not dominated by the source.
        That dependence is FALSE (disjoint columns) but the check is structural.
      * letting `classify_payload` group all 6 RoPE ops, which it does on its own once
        elimination runs before classification (it reports a tidy `members=6` and "tiled OK").
        That plan is INVALID: a group is tiled through its LAST member with `fuse_producers`
        and the lower half is not a producer of the upper one, so half 0 is left outside every
        kernel and the run dies in `Runner` with "Failure while creating the ExecutionEngine".
      * a rank-3 `(rows, 2, half)` single-result form, which would suit bufferization (one
        destination) and needs no reversal map -- the half is selected by `linalg.index(1)` +
        `arith.select`, how Inductor's Triton spells it. It moves the problem into XeGPU: the
        tile stays rank 3 (`d1` extent 2, so unit-extent folding has nothing to fold) and a 3-D
        `xegpu.tensor_desc` is not distributable; tiling `d1` to 1 puts the unit dim in the
        MIDDLE (the store-clipping case `_transpose_block_axis` exists to avoid), and blocking
        `d1` alone leaves 2 work-groups.
    """
    func = get_payload_func_op(mod, func_name)
    block = func.regions[0].blocks[0]
    fused = 0
    # Candidates are visited LAST-insert-first so a chain is taken whole rather than as an
    # earlier sub-chain that would leave its tail sibling unmerged.
    for op in reversed(list(block.operations)):
        block_ops = set(block.operations)
        if op not in block_ops:
            continue  # erased by an earlier merge
        matched = _sibling_slice_chain(op, block_ops)
        if matched is None:
            continue
        chain, shape = matched
        trees = [_elementwise_tree(ins.operands[0], shape) for ins in chain]
        if any(t is None for t in trees):
            continue
        member_sets = [set(members) for members, _ in trees]
        if any(
            member_sets[i] & member_sets[j]
            for i in range(len(trees))
            for j in range(i + 1, len(trees))
        ):
            continue  # shared work: merging would duplicate it, so leave it alone
        roots = [members[-1] for members, _ in trees]
        result_type = roots[0].results[0].type
        if any(r.results[0].type != result_type for r in roots):
            continue
        elem = ir.ShapedType(result_type).element_type
        leaves = []
        for _, tree_leaves in trees:
            leaves.extend(v for v in tree_leaves if v not in leaves)
        last = chain[-1]
        n_out = len(chain)
        # Build before the LAST insert, not the first: a later sibling's own leaf
        # `extract_slice`s sit between the inserts, so anchoring on the first insert would
        # put the merged op above operands it reads ("operand #4 does not dominate this
        # use"). The earlier inserts are then moved down past it, below.
        with ir.InsertionPoint(last), ir.Location.unknown():
            # The `outs` must be the destination's own SLICES, not fresh `tensor.empty`s.
            # With empties, empty-tensor elimination redirects only the FIRST one onto the
            # output argument -- a later insert's destination is an earlier insert's RESULT,
            # and elimination cannot see through that -- so that sibling writes a temporary
            # that a HOST `insert_slice` then copies over device memory, which faults
            # ("longjmp causes uninitialized stack frame").
            dest_base = chain[0].operands[1]
            outs = [
                tensor.ExtractSliceOp(
                    result_type,
                    dest_base,
                    [],
                    [],
                    [],
                    static_offsets=_static_unit_slice(ins)[0],
                    static_sizes=shape,
                    static_strides=[1] * len(shape),
                ).result
                for ins in chain
            ]
            identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(shape)))
            merged = linalg.GenericOp(
                result_tensors=[result_type] * n_out,
                inputs=leaves,
                outputs=outs,
                indexing_maps=ir.ArrayAttr.get([identity] * (len(leaves) + n_out)),
                iterator_types=roots[0].attributes["iterator_types"],
            )
            body = merged.regions[0].blocks.append(
                *([elem] * (len(leaves) + n_out)),
                arg_locs=[ir.Location.unknown()] * (len(leaves) + n_out),
            )
            with ir.InsertionPoint(body):
                leaf_scalars = {v: body.arguments[i] for i, v in enumerate(leaves)}
                yields = []
                for members, _ in trees:
                    scalars = {}
                    for member in members:
                        n_in = len(member.operands) - len(member.results)
                        operand_scalars = [
                            scalars.get(o, leaf_scalars.get(o))
                            for o in list(member.operands)[:n_in]
                        ]
                        scalars[member.results[0]] = _inline_elementwise_body(
                            member, operand_scalars, scalars
                        )
                    yields.append(scalars[members[-1].results[0]])
                ir.Operation.create("linalg.yield", operands=yields)
        for i, ins in enumerate(chain):
            ins.operands[0] = merged.results[i]
        # The old chains are dead now, but nothing runs DCE before `classify_payload`, and a
        # dead linalg op would still be classified into the plan as its own kernel.
        all_members = [m for members, _ in trees for m in members]
        for member in reversed(all_members):
            if all(len(list(v.uses)) == 0 for v in member.results):
                member.operation.erase()
        # The chains' old `outs` slices die with them; drop them so the earlier inserts can
        # move below the merged op without leaving a use above its definition.
        for op_to_drop in list(block.operations):
            if op_to_drop.operation.name == "tensor.extract_slice" and all(
                len(list(v.uses)) == 0 for v in op_to_drop.results
            ):
                op_to_drop.operation.erase()
        for ins in chain[:-1]:
            ins.operation.move_before(last.operation)
        fused += 1
    return fused


# The RoPE-specific name this rewrite was introduced under; the drivers still use it.
fuse_rope_halves = fuse_sibling_slice_writers


def classify_payload(mod: ir.Module, func_name: str = "main") -> list[dict]:
    """Walk the payload function and return an ordered plan, one entry per kernel.

    Entry fields:
      kind    -- matmul / batch_matmul / contraction / reduction / elementwise
      members -- op names owned by this kernel, in program order; the LAST one is tiled
      mnk     -- (M, N, K) for contractions, for per-shape tile selection
      shape   -- the kernel's result shape
    """
    plan: list[dict] = []
    # value -> index of the plan entry that produces it, so an absorb candidate can be
    # checked for operands coming from a DIFFERENT kernel (see the foreign-operand rule).
    owner: dict = {}
    # How many ops of each name have been seen, so every member can record its ORDINAL in
    # IR order. `generic_schedule` needs that: it matches one handle per op name and must
    # index by IR position, not pop in plan order -- an absorbed member (attention swallows
    # the K^T transpose) is consumed at its GROUP's position, which can be later than the
    # transpose's own position in the IR, and popping then hands other transposes the wrong
    # handle and therefore the wrong tile params.
    seen: dict = {}
    for op in mod.body.operations:
        if op.operation.name != "func.func":
            continue
        if func_name not in str(op.attributes["sym_name"]):
            continue
        for blk in op.regions[0].blocks:
            for o in blk.operations:
                name = o.operation.name
                results = {o.results[i] for i in range(len(o.results))}
                operands = {o.operands[i] for i in range(len(o.operands))}

                if name in _VIEW_OPS:
                    # No kernel, but keep the dataflow chain intact so the ops on the far
                    # side of a reshape still group with their producer.
                    if plan and operands & plan[-1]["produced"]:
                        plan[-1]["produced"] |= results
                        plan[-1]["reads"] |= operands
                        for v in results:
                            owner[v] = len(plan) - 1
                    continue
                if name not in _TILED_OPS:
                    continue
                kind = _TILED_OPS[name] or _classify_generic(o)
                ordinal = seen.get(name, 0)
                seen[name] = ordinal + 1

                mnk = None
                if kind in ("matmul", "batch_matmul"):
                    a_shape = ir.ShapedType(o.operands[0].type).shape
                    b_shape = ir.ShapedType(o.operands[1].type).shape
                    mnk = (a_shape[-2], b_shape[-1], a_shape[-1])
                shape = (
                    ir.ShapedType(o.results[0].type).shape if len(o.results) else None
                )
                # Group an elementwise op into the preceding kernel when it consumes that
                # kernel's result AND iterates the same rank: puts casts/activations in
                # their producer's kernel, and makes a reduction plus its normalize step a
                # single kernel.
                #
                # The rank test matters. A group is tiled as ONE iteration space, so
                # merging ops of different rank is wrong in principle -- and it breaks in
                # practice: folding RoPE's rank-3 head view `(T,H,hs/2)` into a rank-2
                # projection leaves a `vector.contract` that `convert-vector-to-xegpu`
                # will not turn into an `xegpu.dpas`, so the matmul annotation finds
                # nothing to anchor. Keeping them separate gives the projection a real
                # DPAS kernel and RoPE its own (which is also Inductor's F1/F2 shape).
                # Ops only share a kernel when they share an ITERATION SPACE, so the
                # shapes must match exactly -- rank alone is not enough. RoPE reshapes a
                # (T,C) projection to (T*H, hs): same rank, different extents, and fusing
                # those into the matmul's forall leaves a `vector.contract` that
                # `convert-vector-to-xegpu` will not turn into an `xegpu.dpas`, so the
                # matmul annotation finds nothing to anchor.
                #
                # A reduction is exempt: it changes shape by definition (its consuming
                # elementwise op is full-rank while the reduced value is not), and grouping
                # them is mandatory since the reduced intermediate cannot be a kernel
                # output. There the iteration space is the OUTPUT's, with the reduction
                # fused in as a producer (exactly `_tile_one_rmsnorm`).
                prev = plan[-1] if plan else None

                # ATTENTION absorbs a whole region, not just an epilogue. A batch_matmul
                # opens an "attention" group, which then swallows everything downstream of
                # it -- the scale multiply, the decomposed softmax's reductions and
                # elementwise ops, and the SECOND contraction -- because the flash rewrite
                # needs `QK^T -> softmax -> @V` inside ONE forall to replace as a unit. The
                # shape/rank test used for ordinary grouping deliberately does not apply
                # here: the region's shapes legitimately change along the chain (scores are
                # (.., T, T) while the output is (.., T, hs)).
                #
                # The group closes once it holds both contractions AND stops receiving
                # elementwise consumers, which is what ends it at the output cast.
                if prev is not None and prev["kind"] == "attention":
                    absorb = operands & prev["produced"] and (
                        kind in ("elementwise", "reduction")
                        or (kind == "batch_matmul" and prev["n_contractions"] < 2)
                    )
                    if absorb:
                        prev["members"].append(name)
                        prev["handles"].append((name, ordinal))
                        prev["member_results"].append(set(results))
                        prev["produced"] |= results
                        prev["reads"] |= operands
                        prev["shape"] = shape
                        for v in results:
                            owner[v] = len(plan) - 1
                        if kind == "batch_matmul":
                            prev["n_contractions"] += 1
                        continue
                if kind == "batch_matmul":
                    # A K^T transpose feeding the region gets NO kernel of its own:
                    # `replace_with_fused_attention` transposes K itself, so once the P@V
                    # contraction is replaced the explicit transpose is dead and is DCE'd.
                    # (Measured: the library `fused_attention_schedule` lowers a torch-mlir
                    # attention payload to 1 kernel with 0 transposes.) Leaving it as its own
                    # entry would emit a pointless copy kernel; absorbing it lets
                    # `fuse_producers` pull it into the region where it dies.
                    #
                    # ONLY a K^T-shaped one, though: a permutation that swaps just the last
                    # two dims. A Llama block also has a HEAD-MAJOR transpose feeding
                    # attention -- `q.view(T,H,hs).transpose(0,1)`, permutation [1,0,2] --
                    # and absorbing that one is wrong, because it leaves Q in (T,H,hs) while
                    # the flash op requires the head dim OUTERMOST: it rank-reduces the Q
                    # slice by dropping dim 0, which on a (128,1,64) token-major slice drops
                    # the 128 and builds the invalid
                    #   extract_slice ... sizes [128,1,64] : tensor<128x1x64xf16> -> tensor<1x64xf16>
                    # that later asserts in `getDroppedDims` ("expected unit dim").
                    # The transpose is found by OWNERSHIP of an operand, not by being the
                    # entry immediately before. In a whole block it is not: the plan reads
                    # `transpose (4,64,256)` = K^T, `transpose (4,256,64)` = the head-major Q
                    # transpose, then attention -- so a `prev`-only test sees only the
                    # head-major one, correctly refuses it, and misses the K^T entirely. The
                    # K^T then keeps a kernel, stays LIVE inside the region, and the flash op
                    # is handed K^T as its `k`: it reads the shape as `(n_ctx=64, d_head=256)`
                    # instead of `(256, 64)`, so the flash loop collapses to a single
                    # iteration over a 64x64 corner of K and V -- silently wrong, and the
                    # extra full-width `tensor_desc<64x256xf16>` load then shifts the
                    # annotation's positional Q/K/V assignment and fails as "TensorDesc shape
                    # is not distributable with the layout".
                    absorbed_members: list = []
                    absorbed_handles: list = []
                    absorbed_results: list = []
                    absorbed_heads: list = []
                    for v in operands:
                        cand = plan[owner[v]] if v in owner else None
                        if (
                            cand is not None
                            and cand["kind"] == "transpose"
                            and not cand.get("absorbed")
                            and _is_minor_transpose(cand.get("permutation"))
                        ):
                            # Marked rather than popped: popping renumbers `plan`, which
                            # would invalidate every `owner` index above it and misattribute
                            # kinds in the JOIN rule below. Absorbed entries are dropped at
                            # the end, once `owner` is dead.
                            cand["absorbed"] = True
                            absorbed_members = cand["members"]
                            absorbed_handles = cand["handles"]
                            absorbed_results = cand["member_results"]
                            absorbed_heads = [cand]
                            break
                    plan.append(
                        {
                            "op": name,
                            "kind": "attention",
                            "mnk": mnk,
                            "shape": shape,
                            "members": absorbed_members + [name],
                            "handles": absorbed_handles + [(name, ordinal)],
                            "member_results": absorbed_results + [set(results)],
                            # A COPY: `produced` grows as view ops chain off this entry, and
                            # `member_results` must keep each member's own results.
                            "produced": set(results),
                            # Every value the group's members READ. The reduction absorb below
                            # walks this to find the head of a chain that is more than one op
                            # deep.
                            "reads": set(operands).union(
                                *[h["reads"] for h in absorbed_heads]
                            )
                            if absorbed_heads
                            else set(operands),
                            "n_contractions": 1,
                        }
                    )
                    for v in results:
                        owner[v] = len(plan) - 1
                    continue

                # An elementwise op may not join the preceding kernel if it reads results
                # from TWO different contraction kernels. Such an op is a JOIN, and
                # absorbing it over-fuses: the group is
                # tiled through its last member with `fuse_producers`, so a residual add
                # like `h + o` -- h from the attention output projection, o from the FFN --
                # drags BOTH matmul chains into one forall. That kernel then holds two
                # k-loops and two `vector.contract`s, and the single-DPAS mlp annotation
                # cannot take it ("requires exactly one target value handle (got 2)").
                # This is the same over-fusion the leaf-consumer experiments hit; see the
                # "Composition findings" note about tiling directly in topological order.
                #
                # Two earlier, WRONGER versions of this rule, for the record: refusing on
                # ANY foreign operand split each RoPE from 4 kernels to 6 (a foreign
                # elementwise operand is harmless), and refusing on ONE foreign contraction
                # broke RoPE outright ("conversion failed for builtin.unrealized_conversion_cast"),
                # because `hi*sin` reads the projection just as `lo*cos` does.
                # Count DISTINCT kernels holding a contraction that this op reads from.
                # Two or more means it JOINS two contraction chains -- the residual add
                # `h + o`, with h from the attention output projection and o from the FFN.
                # One is the ordinary case and must stay fusable: both RoPE halves read the
                # SAME projection (`lo*cos` and `hi*sin`), and a matmul epilogue reads its
                # own matmul.
                contraction_srcs = {
                    owner[v]
                    for v in operands
                    if v in owner
                    and plan[owner[v]]["kind"] in _CONTRACTION_KINDS | {"attention"}
                }
                foreign = len(contraction_srcs) >= 2

                # MERGE rule: an elementwise op reading TWO OR MORE elementwise kernels that
                # share its iteration space merges those entries into one kernel, with itself
                # as the new last member.
                #
                # Needed because the ordinary grouping rule only looks at `plan[-1]`, so it
                # can only ever extend the entry that happens to sit immediately before. RoPE
                # is the case that costs: `out[:, :half] = lo*cos - hi*sin` emits
                #   mulf(lo,cos)   mulf(hi,sin)   subf
                # in that order, and `mulf(hi,sin)` does NOT read `mulf(lo,cos)`'s result (it
                # reads the projection and the sin table), so it opens its own entry; the
                # `subf` then joins THAT one and nothing ever goes back for `mulf(lo,cos)`,
                # which is left as a kernel computing a value its neighbour immediately
                # consumes. Merging gives one kernel per RoPE half instead of two: 4 -> 2 per
                # RoPE, i.e. -4 kernels on a Llama block.
                #
                # Only ELEMENTWISE sources, and only on the same iteration space. Two
                # CONTRACTION sources are the JOIN case below and must NOT be merged -- fusing
                # there clones a matmul chain into the kernel and the single-DPAS annotation
                # rejects the two `vector.contract`s. A transpose/reduction/attention source is
                # excluded for the same reason: the merged group is tiled as one forall through
                # its last member, which is only valid when every member shares that space.
                # The merge target is the LATEST source, never the earliest: the combined group
                # then sits at a plan position after all of its producers, which is what keeps
                # the tiling order topological (tile a group before its producers are in
                # foralls and `fuse_producers` clones them, leaving the originals live outside
                # every kernel). Entries are MARKED absorbed rather than removed, because
                # removing renumbers `plan` and invalidates every `owner` index above it.
                # If a merged member's result turns out to escape, `_split_multi_output_groups`
                # splits the group again, so a wrong merge degrades to today's behaviour rather
                # than miscompiling.
                if kind == "elementwise" and not foreign:
                    srcs = sorted({owner[v] for v in operands if v in owner})
                    if len(srcs) >= 2 and all(
                        plan[s]["kind"] == "elementwise"
                        and not plan[s].get("join", False)
                        and not plan[s].get("absorbed", False)
                        and _shape_key(plan[s]["shape"]) == _shape_key(shape)
                        for s in srcs
                    ):
                        target_idx = srcs[-1]
                        target = plan[target_idx]
                        members: list = []
                        handles: list = []
                        results_per_member: list = []
                        for s in srcs:
                            src = plan[s]
                            members += src["members"]
                            handles += src["handles"]
                            results_per_member += src["member_results"]
                            if s != target_idx:
                                src["absorbed"] = True
                                target["produced"] |= src["produced"]
                            target["reads"] |= src["reads"]
                        target["reads"] |= operands
                        target["members"] = members + [name]
                        target["handles"] = handles + [(name, ordinal)]
                        target["member_results"] = results_per_member + [set(results)]
                        target["produced"] |= results
                        target["shape"] = shape
                        for v in list(owner):
                            if owner[v] in srcs:
                                owner[v] = target_idx
                        for v in results:
                            owner[v] = target_idx
                        continue

                # NOT DONE, and recorded because it looks free and is not: absorbing the
                # elementwise op that FEEDS a reduction. RMSNorm's leading `x*x` is its own
                # entry -- it is the head of the reduction chain, not part of it, so nothing
                # groups it -- which costs a kernel per norm (2 on a Llama block) for a value
                # nobody reads, since `_tile_one_reduction` fuses `x*x` in as a producer
                # anyway. Absorbing it into the reduction group DOES give 27 kernels instead
                # of 28, but it then breaks tiling: with `x*x` inside the forall,
                # `_tile_one_reduction`'s second `fuse_elementwise_producers(tiled_red,
                # red_loop)` finds it among the reduction's traced producers and fails with
                # "could not find next producer to fuse into container". Fixing that means
                # reworking the reduction's inner fusion, which is the most delicate path in
                # this schedule, for two kernels out of 28 -- the transpose/GQA (+6) and RoPE
                # (+6) items are worth far more. See the kernel-count table in §3b of
                # llama3_torch_mlir_optimization_plan.md.
                # Note it only ever fires for the FIRST norm anyway: the second norm's `x*x`
                # is absorbed into the O-projection group during the walk and only becomes its
                # own entry later, in `_split_multi_output_groups`.
                # A JOIN entry must stay a SINGLE op, so it cannot be joined either. A join is
                # tiled with `fuse_producers=False` (that is the whole point: fusing would
                # clone a foreign matmul in), and a group is tiled through its LAST member --
                # so any earlier member of a join group is never pulled into the forall and
                # stays live outside every kernel. It then reaches LLVM translation as a
                # `linalg.generic` over memrefs and fails as "LLVM Translation failed for
                # operation: builtin.unrealized_conversion_cast" (the cast that feeds it), which
                # surfaces as `RuntimeError: Failure while creating the ExecutionEngine`.
                # This is not hypothetical: with the norm in f32 the LAST block's residual add
                # `h + o` is a join whose ONLY consumer is the final norm's `x.float()` extf, so
                # the extf grouped into it and the add was left behind. It bites only there --
                # an intermediate block's output is read by both the next norm and the next
                # residual add, so two results escape and `_split_multi_output_groups` already
                # separates those.
                if (
                    kind == "elementwise"
                    and prev is not None
                    and operands & prev["produced"]
                    and not foreign
                    and not prev.get("join", False)
                    and (
                        prev["kind"] == "reduction"
                        or _shape_key(shape) == _shape_key(prev["shape"])
                    )
                ):
                    prev["members"].append(name)
                    prev["handles"].append((name, ordinal))
                    prev["member_results"].append(set(results))
                    prev["produced"] |= results
                    prev["reads"] |= operands
                    prev["shape"] = shape  # the group's output shape
                    for v in results:
                        owner[v] = len(plan) - 1
                    continue

                entry = {
                    "op": name,
                    "kind": kind,
                    "mnk": mnk,
                    "shape": shape,
                    "members": [name],
                    "handles": [(name, ordinal)],
                    "member_results": [set(results)],
                    "produced": set(results),
                    "reads": set(operands),
                    # A JOIN: an elementwise op reading results of other CONTRACTION kernels
                    # (the residual adds). It gets its own kernel, and it must be tiled with
                    # `fuse_producers=False` -- it has no members of its own to fuse, and
                    # fusing is exactly what would CLONE a foreign matmul into it. Without
                    # this the "elementwise" kernel ends up holding a `vector.contract` and
                    # gets elementwise layouts, which fails as
                    # "'xegpu.load_nd' op TensorDesc shape is not distributable".
                    "join": kind == "elementwise" and foreign,
                }
                if kind == "transpose":
                    # Kept because it decides whether the attention region may absorb this
                    # transpose -- see the absorb rule above.
                    # `permutation` is a DenseI64ArrayAttr, so iterating gives plain ints.
                    entry["permutation"] = list(o.attributes["permutation"])
                plan.append(entry)
                for v in results:
                    owner[v] = len(plan) - 1
    plan = [e for e in plan if not e.pop("absorbed", False)]
    plan = _split_multi_output_groups(plan)
    plan = _absorb_reduction_heads(plan)
    for entry in plan:
        entry.pop("produced", None)
        entry.pop("reads", None)
        entry.pop("member_results", None)
    return plan


def _escapes(value, produced) -> bool:
    """True if `value` is read by an op that is not part of the same kernel group.

    "Part of the group" is decided by results, not by identity: `produced` holds every value
    the group's members and their chained view ops define, so a consumer whose results are
    all in `produced` is inside the group. An op with NO results (the function's
    `tensor.insert_slice`/return path) always counts as outside -- that is the group's real
    output.
    """
    for use in value.uses:
        owner = use.owner
        results = [owner.results[i] for i in range(len(owner.results))]
        if not results or any(r not in produced for r in results):
            return True
    return False


def _absorb_reduction_heads(plan: list[dict]) -> list[dict]:
    """Fold the elementwise HEAD of a reduction chain into that reduction's group.

    RMSNorm reads `xf = x.float()`, then `xf*xf`, then sums. Those two ops share the
    reduction's iteration space and feed nothing else, but neither can join a group during the
    walk: `x*x` heads the chain so there is no preceding kernel to join, and the reduction
    itself opens a new entry rather than extending them. Left alone they are 2 kernels per norm
    -- 4 per block -- computing values that `_tile_one_reduction` re-fuses as producers anyway,
    so nothing ever reads what those kernels write.

    Run as a POST-PASS, after `_split_multi_output_groups`, because the second norm's heads are
    not their own entries during the walk at all: `h` comes from the O-projection kernel, so the
    `extf` joins THAT group by the ordinary elementwise rule and only becomes a separate entry
    when the split pass cuts the group at its first escaping result. A walk-time absorb catches
    the first norm and misses the second.

    All-or-nothing per norm, and that is not a simplification: absorbing only the `extf` would
    leave `x*x` a separate kernel READING `xf`, so `xf` would escape the group and
    `_split_multi_output_groups` would cut it straight back out.
    """
    absorbed: set = set()
    for j, red in enumerate(plan):
        if red["kind"] != "reduction":
            continue
        # Candidate heads: earlier elementwise entries this group reads, on its own iteration
        # space. `reads` is the union over ALL members, which is what makes the direct test
        # enough here -- the normalize member reads `xf`, so the cast is a direct hit even
        # though only `x*x` touches the reduction op itself.
        heads = [
            i
            for i, cand in enumerate(plan[:j])
            if i not in absorbed
            and cand["kind"] == "elementwise"
            and not cand.get("join", False)
            and _shape_key(cand["shape"]) == _shape_key(red["shape"])
            and set().union(*cand["member_results"]) & red["reads"]
        ]
        if not heads:
            continue
        # Refuse if a head's result is read from OUTSIDE the merged group: the group is tiled
        # through its last member, so an escaping earlier member would be cloned in and left
        # live outside every kernel (the orphan that fails LLVM translation as a stray
        # `builtin.unrealized_conversion_cast`).
        produced = set(red["produced"]).union(*[plan[i]["produced"] for i in heads])
        if any(
            _escapes(v, produced)
            for i in heads
            for results in plan[i]["member_results"]
            for v in results
        ):
            continue
        absorbed |= set(heads)
        red["members"] = [m for i in heads for m in plan[i]["members"]] + red["members"]
        red["handles"] = [h for i in heads for h in plan[i]["handles"]] + red["handles"]
        red["member_results"] = [
            r for i in heads for r in plan[i]["member_results"]
        ] + red["member_results"]
        red["produced"] = produced
        red["reads"] = set(red["reads"]).union(*[plan[i]["reads"] for i in heads])
    return [e for i, e in enumerate(plan) if i not in absorbed]


def _split_multi_output_groups(plan: list[dict]) -> list[dict]:
    """Split any group that has more than one ESCAPING result.

    Every group is tiled through its LAST member with `fuse_producers`, which only makes a
    correct kernel when that member's result is the one value the rest of the payload reads.
    A group with two escaping results breaks it: tile-and-fuse CLONES producers into the new
    loop rather than moving them, so the earlier escaping member's original op stays live
    OUTSIDE any forall, reaches LLVM lowering as a `linalg.matmul` over memrefs and fails
    with "expected add/mul op in the body".

    A Llama block produces exactly that. The second RMSNorm's `x*x` consumes the residual add
    `h = x + attn`, has the same shape, and so joins the output-projection group by the
    ordinary elementwise rule -- but `h` itself is read by the FINAL residual add and by the
    norm's own rescale, so both `h` and `x*x` escape. Splitting after `h` leaves the
    projection group ending at its real output and gives `x*x` its own kernel, which is
    exactly the shape the FIRST RMSNorm already has (there `x` is a function argument, so
    `x*x` never had a group to join). `_tile_one_reduction` then fuses it back in as a
    producer, as it already does for the first norm.

    Attention is exempt: the flash rewrite replaces the whole region as a unit, and its
    intermediates are dead afterwards rather than escaping.
    """
    out: list[dict] = []
    for entry in plan:
        if entry["kind"] == "attention" or len(entry["members"]) < 2:
            out.append(entry)
            continue
        produced = entry["produced"]
        escaping = [
            i
            for i, results in enumerate(entry["member_results"])
            if any(_escapes(v, produced) for v in results)
        ]
        # Keep the group whole while only its last member escapes, which is the normal case.
        if len(escaping) < 2:
            out.append(entry)
            continue
        cut = escaping[0] + 1
        tail = dict(entry)
        entry["members"] = entry["members"][:cut]
        entry["handles"] = entry["handles"][:cut]
        entry["member_results"] = entry["member_results"][:cut]
        entry["shape"] = _shape_of(entry["member_results"][-1])
        # The tail is elementwise by construction: only elementwise ops are ever absorbed
        # into a group, so everything after the cut is one.
        tail.update(
            op="linalg.generic",
            kind="elementwise",
            mnk=None,
            join=False,
            members=tail["members"][cut:],
            handles=tail["handles"][cut:],
            member_results=tail["member_results"][cut:],
        )
        out.append(entry)
        out.extend(_split_multi_output_groups([tail]))
    return out


def _shape_of(results):
    """Result shape of a member, from its (single-element) result set."""
    for v in results:
        return ir.ShapedType(v.type).shape
    return None


def _is_minor_transpose(permutation) -> bool:
    """True for a permutation that swaps ONLY the last two dims, i.e. the K^T shape.

    `[0, 2, 1]` yes; `[1, 0, 2]` (the head-major transpose) no. The distinction matters
    because the fused-attention rewrite subsumes a K^T but requires its Q/K/V operands to be
    batch-major, so a transpose that moves the BATCH dim must keep its own kernel.
    """
    if not permutation:
        return False
    n = len(permutation)
    return n >= 2 and list(permutation) == list(range(n - 2)) + [n - 1, n - 2]


def _rank(shape) -> int:
    """Rank of a result shape, or -1 when unknown (so it never matches)."""
    return len(shape) if shape is not None else -1


def _shape_key(shape):
    """Hashable shape for iteration-space comparison; None never compares equal."""
    return tuple(shape) if shape is not None else None


def _floor_to(value: int, multiple: int) -> int:
    """Largest multiple of `multiple` that is <= value (at least one multiple)."""
    return max(multiple, (value // multiple) * multiple)


def params_for_plan(
    plan: list[dict], device: str, ew_params: dict = EW_PARAMS
) -> list[dict]:
    """Pick per-kernel schedule params: one selector call per distinct contraction shape.

    Elementwise work-group tiles are clamped to the result extent, so a narrow tensor does
    not get a tile wider than it is (the annotation asserts wg % sg == 0, and an over-wide
    column tile inflates the live vector).
    """
    selector = XeGPUParameterSelector(device=device)
    cache: dict[tuple, dict] = {}
    out = []
    for entry in plan:
        if entry["kind"] in _CONTRACTION_KINDS and entry["mnk"] is not None:
            mnk = entry["mnk"]
            if mnk not in cache:
                cache[mnk] = selector.get_parameters_dict(mnk)
            out.append(cache[mnk])
        elif entry["kind"] == "reduction":
            out.append(dict(RED_PARAMS))
        elif entry["kind"] == "attention":
            # d_head comes from the region's own output width, so a payload with a
            # different head dim is at least reported rather than silently mis-tiled.
            p = dict(FA_PARAMS)
            if entry["shape"]:
                p["n_head"] = entry["shape"][-1]
            out.append(p)
        else:
            p = dict(ew_params)
            shape = entry["shape"]
            if shape and len(shape) >= 2:
                # Which extent sizes the row tile depends on how the kind is tiled, so the two
                # must stay in step -- a disagreement shows up as "'xegpu.load_nd' op
                # TensorDesc shape is not distributable with the layout". Elementwise
                # (`_ew_tile_sizes`) blocks the largest non-innermost dim (`_block_axis`); a
                # transpose (`_transpose_tile_sizes`) always blocks the LAST non-innermost one
                # (`_transpose_block_axis`, and that is a correctness requirement -- see there).
                # Rank 2 blocks the rows in both cases.
                if len(shape) <= 2:
                    rows = shape[-2]
                elif entry["kind"] == "transpose":
                    rows = shape[_transpose_block_axis(shape)]
                else:
                    rows = shape[_block_axis(shape)]
                # Shrink the SUBGROUP tile before clamping the work-group tile. The blocked
                # extent can be smaller than one subgroup -- the attention output transpose
                # blocks H = 4 against `sg_m` = 32 -- and `_floor_to` floors to at least one
                # subgroup, so clamping alone would claim a 32-row tile of a 4-row dim. All of
                # these are powers of two in practice, so `min` keeps the divisibility the
                # annotation asserts (wg % sg == 0, sg % load == 0).
                p["sg_m"] = min(p["sg_m"], rows)
                p["load_m"] = min(p["load_m"], p["sg_m"])
                p["sg_n"] = min(p["sg_n"], shape[-1])
                p["load_n"] = min(p["load_n"], p["sg_n"])
                p["wg_m"] = min(p["wg_m"], _floor_to(rows, p["sg_m"]))
                p["wg_n"] = min(p["wg_n"], _floor_to(shape[-1], p["sg_n"]))
            out.append(p)
    return out


def _block_axis(shape) -> int:
    """Which dim a rank > 2 kernel blocks: the LARGEST non-innermost extent.

    The innermost dim is always left whole -- it is the contiguous one, and keeping it whole
    is what makes the store a block store. Among the rest, blocking the largest extent is the
    only choice that is safe in general. Blocking a fixed position instead breaks on real
    payloads: GQA's broadcast is `(kv, rep, hs, T)` with LEADING extents of 2, so blocking
    dim 0 asks for a 32-wide tile of a 2-wide dim (`_floor_to` floors to at least one
    subgroup, which cannot go below `sg_m`), and the tiled op then dies with "Attempted to
    vectorize, but failed". The output transpose `(T, H, hs)` has the same problem one dim
    over, with H = 4.
    """
    lead = list(shape[:-1])
    return max(range(len(lead)), key=lambda i: lead[i])


def _transpose_block_axis(shape) -> int:
    """Which dim a rank > 2 TRANSPOSE blocks: always the last non-innermost one.

    Not `_block_axis` (largest extent), and the difference is a correctness bug rather than a
    tuning choice. Peeling every dim before the blocked one to 1 puts the resulting unit dims
    LEADING in the output slice, and only a leading unit dim survives the rank reduction:

      * blocked dim last  -> output slice `(1, ..., 1, tile, innermost)`. Vectorizing the
        `tensor.insert_slice` of the folded 2-D value into that slice gives a minor-identity
        `transfer_write` whose dims line up, `in_bounds = [true, true]`. Correct.
      * blocked dim earlier -> a unit dim in the MIDDLE, e.g. `(tile, 1, innermost)` for the
        attention output transpose `(H,T,hs) -> (T,H,hs)` when T is blocked. Upstream
        `vectorizeAsInsertSliceOp` then emits `vector.transfer_write vector<128x64xf16>` into
        `tensor<128x1x64xf16>` under a MINOR-IDENTITY map with `in_bounds = [false, true]` --
        it computes the vector SHAPE correctly for a non-trailing dropped dim but not the
        permutation map -- so the 128 lands on the size-1 dim and the write is CLIPPED TO ONE
        ROW. That lowered to `create_nd_tdesc` on a `memref<1x64xf16>` producing a
        `tensor_desc<128x64xf16>`, and it silently threw away the whole attention result: the
        full block "passed" at rel 0.0052 while `attn @ wo` was bitwise zero.

    The read side is free to have its unit dim in the middle: the folding leaves an explicit
    2-D `tensor.extract_slice`, which `_bufferize_keeping_transpose_subviews` keeps as a
    rank-reduced strided `memref.subview` that `xegpu.create_nd_tdesc` takes directly.

    For the Q/K/V head-major transposes `(T,H,hs) -> (H,T,hs)` this picks the same dim
    `_block_axis` already picked, so it changes nothing there.
    """
    return len(shape) - 2


def _rank_n_tile_sizes(shape, params, axis=None) -> list[int]:
    """Peel every non-innermost dim by 1 except the blocked one; leave the innermost whole.

    The unit dims are load-bearing: they let the post-tiling unit-extent folding collapse the
    op to 2-D, which is the only shape the XeGPU work-group layouts can distribute over (a
    rank-3 `tensor_desc` is rejected outright). Keeping the innermost dim whole is what keeps
    the store contiguous.
    """
    if axis is None:
        axis = _block_axis(shape)
    return [params["wg_m"] if i == axis else 1 for i in range(len(shape) - 1)] + [0]


def _ew_tile_sizes(shape, params) -> list[int]:
    """Work-group tile sizes for an elementwise op of any rank.

    Rank 2 tiles both dims; higher ranks go through `_rank_n_tile_sizes`. The hand schedule
    does the same thing for RoPE with `tile_sizes=[1, wg_rows, 0]`.
    """
    rank = len(shape) if shape else 2
    if rank <= 2:
        return [params["wg_m"], params["wg_n"]]
    return _rank_n_tile_sizes(shape, params)


def _transpose_tile_sizes(shape, params) -> list[int]:
    """Work-group tile sizes for a `linalg.transpose`.

    Differs from elementwise only at rank 2, where the innermost dim is left WHOLE rather
    than tiled by `wg_n`. A transpose's iteration space is its OUTPUT (`linalg.transpose`
    takes its rank from `getInit()`, giving the output the identity map and the input the
    inverse permutation), and what has to stay contiguous is the output STORE -- so block
    rows and keep the full row. The read is then the strided side, which is the correct way
    round: a column-strided STORE is what silently corrupts a transpose.
    """
    rank = len(shape) if shape else 2
    if rank <= 2:
        return [params["wg_m"], 0]
    return _rank_n_tile_sizes(shape, params, axis=_transpose_block_axis(shape))


def _tile_one_reduction(anytype, output, wg_rows, rss):
    """Tile a reduction group into one kernel (mirrors `_tile_one_rmsnorm`).

    `output` is the group's final elementwise generic; fusing its producers pulls the
    reduction in. Mandatory for a reduction: the reduced intermediate cannot be a kernel
    output, since a reduced/1-wide `xegpu.tensor_desc` is invalid.
    """
    _, [forall], _ = lh_transform.tile(
        output,
        tile_sizes=[wg_rows],
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    func = transform.get_parent_op(
        anytype, forall, op_name="func.func", deduplicate=True
    )
    transform.apply_cse(forall)
    transform.apply_dce(func)

    generics = match(forall, ops={"linalg.generic"})
    reduction = transform_ext.filter_reduction_ops(generics)
    out_op = transform_ext.extract_handle(
        transform_ext.filter_elementwise(generics), -1
    )

    def fuse_elementwise_producers(target, loop, stop_at_reductions=False):
        producers = transform_ext.filter_by_name(
            transform_ext.filter_elementwise(
                transform_ext.trace_producers(
                    target, stop_at_reductions=stop_at_reductions
                )
            ),
            "linalg.generic",
        )
        structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=producers, containing_op=loop
        )

    # `stop_at_reductions=True` is load-bearing once the group owns the HEAD of its chain
    # (`x*x`, and `x.float()` for the f32 norm). Tracing from the group's OUTPUT walks back
    # through the reduction and, unbounded, picks up `x*x` -- whose only consumer is the
    # reduction, not this loop -- so `fuse_into_containing_op` fails with "could not find next
    # producer to fuse into container". (That is the failure recorded in `classify_payload` as
    # the reason absorbing the head "breaks tiling": it is about the PRODUCER SET, not about
    # fusion being impossible.) With the reduction as a barrier the set is exactly the normalize
    # chain plus `x.float()`, which the normalize reads DIRECTLY and so is still reachable.
    tiled_out, out_loop = structured.TileUsingForOp(out_op, sizes=[0, rss]).results
    fuse_elementwise_producers(tiled_out, out_loop, stop_at_reductions=True)
    transform.apply_dce(forall)

    # The reduction's own loop gets the other side: `x*x` and, again, `x.float()`.
    _, tiled_red, _, red_loop = structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=reduction, tile_sizes=[0, rss]
    )
    fuse_elementwise_producers(tiled_red, red_loop)
    transform.apply_cse(forall)
    canonicalize(forall)
    return forall


def _tile_one_attention(anytype, pv_op, shape, params, absorbed_transpose=False):
    """Tile one attention region into a forall and rewrite it as a flash loop.

    Structure follows the hand `llama3_schedule.py:_fuse_attention_in_region`: tile the P@V
    contraction, pull the rest of the region in, then hand
    `transform_ext.replace_with_fused_attention` five explicit handles. It replaces the P@V
    contraction with an online-softmax loop, after which the materialized scores, the
    softmax chain and the K^T transpose are all dead and get DCE'd.

    Two deliberate differences from the hand version:

    * Producers are pulled in with `fuse_producers=True` rather than by hand-walking the
      SSA chain op by op. The hand walk (`div -> den -> num -> mx -> scaled -> qkt` plus
      four fills) is written against the hand payload's exact op sequence; torch-mlir's is
      longer and differently shaped -- an extra `truncf` after the f32-accumulating
      batch_matmul, a separate subtract, a `tensor.expand_shape` before it, and a
      TWO-result max (torch's max carries an i64 argmax) -- so hand hops would be brittle.
      Topological tiling order makes this safe: every earlier kernel is already inside its
      own forall and cannot be re-absorbed.

    * The SCALE is found the library schedule's way, not the hand way. The hand code matches
      a `linalg.mul`/`linalg.elementwise` and walks operand 1 -> `linalg.fill` ->
      `arith.constant`, which is the hand payload's scale-broadcast-into-a-tensor shape.
      torch-mlir captures the scale as a SCALAR constant inside a generic
      (`arith.mulf %in, %cst`), with no fill and no named mul op, so we go
      max reduction -> its first generic ancestor -> `arith.mulf` -> `arith.constant`.

    The max reduction is only a landmark for that search; nothing else needs it.
    """
    rank = len(shape) if shape else 3
    # Peel the batch dims by 1 and block the query rows, leaving head_dim whole: the inner
    # op is then plain single-head attention, which is the shape the flash rewrite expects.
    tile_sizes = [1] * (rank - 2) + [params["wg_rows"], 0]
    _, [forall], _ = lh_transform.tile(
        pv_op,
        tile_sizes=tile_sizes,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    func = transform.get_parent_op(
        anytype, forall, op_name="func.func", deduplicate=True
    )
    transform.apply_cse(forall)
    lh_transform.cleanup(func)

    linalg_ops = match(forall, ops={"linalg.generic", "linalg.batch_matmul"})
    qk_matmul, pv_matmul = transform.split_handle(
        2 * [anytype], transform_ext.filter_contraction_ops(linalg_ops)
    )

    def producers_by_name(target, op_names):
        return transform_ext.filter_by_name(
            transform_ext.trace_producers(target), op_names=op_names
        )

    # Q/K/V are taken from the contractions' OPERANDS, not by picking the Nth
    # `tensor.extract_slice` producer as both reference schedules do. Positional selection
    # over traced producers only holds when Q/K/V are function arguments: in a whole block
    # they are other kernels' outputs, more slices appear in the trace, and the handles slide
    # -- the flash rewrite then gets the wrong `k`, its K^T transpose stays LIVE, and the
    # kernel ends up with an extra full-width `tensor_desc<64x256xf16>` load. That shifts the
    # annotation's positional load assignment (it expects Q, K, V) and fails as
    # "'xegpu.load_nd' op TensorDesc shape is not distributable with the layout".
    #
    # Passing the SLICES (rather than the underlying buffers) is deliberate: it is how the
    # flash op recovers the query-row offset that causal masking needs.
    q = transform.get_producer_of_operand(anytype, qk_matmul, operand_number=0)
    k = transform.get_producer_of_operand(anytype, qk_matmul, operand_number=1)
    if absorbed_transpose:
        # K^T was absorbed into this region, so QK^T's rhs is the transpose. The flash op
        # wants K itself -- `[*batch, n_ctx, d_head]`, which it transposes internally -- so
        # hop over it. The transpose is then dead and is DCE'd, which is exactly what makes
        # the kernel come out with 3 loads instead of 4.
        k = transform.get_producer_of_operand(anytype, k, operand_number=0)
    # P@V: operand 0 is the softmax output, operand 1 is V.
    v = transform.get_producer_of_operand(anytype, pv_matmul, operand_number=1)

    # The scale: torch-mlir emits it as a generic holding `arith.mulf %in, %cst`, so match
    # the mulf INSIDE the forall and trace to its constant. This is region-scoped, which is
    # better than either reference: the hand schedule matches a `linalg.mul`/
    # `linalg.elementwise` and walks operand 1 -> `linalg.fill` -> constant (the hand
    # payload's scale-broadcast-into-a-tensor, which torch-mlir never produces), and the
    # library schedule uses the max reduction as a landmark and searches the whole FUNCTION
    # for `arith.max*` -- which would not even work here, because the softmax reductions do
    # NOT end up inside the forall (the `tensor.expand_shape` that restores the reduced rank
    # blocks the fusion), and searching the function breaks as soon as there are two
    # attention regions. Nothing else needs the reductions: once the P@V contraction is
    # replaced by the flash loop the whole softmax chain is dead and is DCE'd where it lies.
    # Take the FIRST `arith.mulf`, not the only one: there are several. The scale multiply
    # feeds both the max reduction and the subtract, and tile-and-fuse clones a producer per
    # consumer, so the region ends up with three copies of it. They are clones reading the
    # same constant, so any of them traces to the right `arith.constant`, and all of them
    # die with the rest of the chain when the flash loop replaces the contraction.
    #
    # The library schedule avoids the duplication by running `linalg-fuse-elementwise-ops`
    # over the function before tiling, which collapses the chain first. We deliberately do
    # NOT: that pass rewrites every other contraction into a generic too, and on a payload
    # with projections it takes `linalg.matmul` to 0 and kills the DPAS path (measured -- see
    # `_gqa_probe.py`). Tolerating the clones is the cheaper trade.
    # Matching `arith.mulf` across the region is NOT enough to find the scale: a
    # `linalg.batch_matmul` over f16 operands with an f32 accumulator carries an IMPLICIT
    # body of `arith.extf` + `arith.mulf` + `arith.addf`, which the printer does not show for
    # a named op. So the FIRST `arith.mulf` in the region is the QK^T contraction's own
    # multiply -- an f32 one whose operand is an `arith.extf` -- and using it fails with
    # "Expected scale to be arith.constant, got arith.extf". Select by ENCLOSING OP rather
    # than by position: keep only the multiplies whose immediate parent is a
    # `linalg.generic`, which is the form torch-mlir emits the scale in.
    mulfs = match(forall, ops={"arith.mulf"})
    scale_generic = transform_ext.extract_handle(
        transform_ext.filter_by_name(
            transform.get_parent_op(anytype, mulfs, deduplicate=True),
            op_names="linalg.generic",
        ),
        0,
    )
    scale_mul = transform_ext.extract_handle(
        match(scale_generic, ops={"arith.mulf"}), 0
    )
    # Operand 1 of `arith.mulf %in, %cst` IS the scale constant, so ask what defines it
    # rather than tracing producers: the constant sits at FUNCTION scope, outside the
    # forall, and `trace_producers` does not walk out of the region (it returns an empty
    # handle and `extract_handle` then fails with "Invalid index 0 for target of length 0").
    scale = transform.get_producer_of_operand(anytype, scale_mul, operand_number=1)

    transform_ext.replace_with_fused_attention(
        q=q,
        k=k,
        v=v,
        scale=scale,
        output=pv_matmul,
        tile_size=params["inner_loop_tile_size"],
        causal=params.get("causal", False),
    )
    transform.apply_cse(forall)
    lh_transform.cleanup(func)
    return forall


def _vectorize_kernels_only(func):
    """Vectorize the linalg ops INSIDE the foralls, and nothing else.

    The reusable `vectorize()` hands the whole function to
    `transform.structured.vectorize_children_and_apply_patterns`, and that is wrong here.
    Every op that becomes a kernel is inside an `scf.forall` by construction, so whatever
    is left outside one is HOST code -- and `tensor.insert_slice` is vectorizable
    (`linalg::hasVectorizationImpl`). A payload that writes its result in slices (RoPE's
    two rotated halves) therefore gets its two host-level inserts turned into a
    `vector<1024x32xf16>` transfer pair in the host function, reading and writing device
    memory, which faults. Left unvectorized they reach bufferization, which sees that each
    forall already wrote that exact subset of the output -- empty-tensor elimination made
    the forall's init that subset -- and elides them entirely.

    `vectorize_children_and_apply_patterns` cannot simply be pointed at the foralls: it
    requires an isolated-from-above target. So the per-op form is used and the patterns it
    would have run afterwards are applied here instead. They are not optional --
    `reduction_to_contract` and `fold_arith_extension` are what turn a vectorized matmul
    into a `vector.contract` over f16 operands with an f32 accumulator, which is the one
    shape `convert-vector-to-xegpu` will emit an `xegpu.dpas` for.
    """
    # Matched by INTERFACE, not by op name: reduction tiling introduces linalg ops the
    # plan never names (`tile_reduction_using_for` adds a partial-accumulator op and a
    # merge), and leaving one unvectorized crashes the XeGPU lowering later. Per forall,
    # because a `structured.match` takes a single target op.
    #
    # `tensor.insert_slice` is vectorized too, but ONLY here, inside a forall. That is the
    # whole distinction this function draws: inside a forall is kernel code, where an insert
    # must become vector transfers; outside is host code, where vectorizing one produces a
    # giant host-side transfer over device memory. A transpose needs this -- the unit-extent
    # folding reduces a tiled `linalg.transpose` to two rank-2 strided subviews with only a
    # subset copy between them, no linalg op left, and unvectorized that bufferizes to a
    # `memref.copy` in the KERNEL, which lowers to a `memrefCopy` runtime call that is not
    # linked into the device binary ("'llvm.call' op 'memrefCopy' does not reference a
    # symbol in the current scope").
    with lh_transform.foreach(match(func, ops={"scf.forall"})) as loop:
        structured.structured_vectorize(
            match(loop, interface=structured.MatchInterfaceEnum.LinalgOp), []
        )
        structured.structured_vectorize(match(loop, ops={"tensor.insert_slice"}), [])
        # The follow-up patterns are applied PER FORALL, for the same reason vectorization
        # is: a forall is one kernel, and these patterns reach across kernel boundaries and
        # into the host code between them when applied func-wide. Scoping them keeps each
        # rewrite inside the one kernel it is allowed to reason about. (Unlike a registered
        # PASS, `apply_patterns` has no isolated-from-above requirement, so it CAN be scoped
        # to a forall.)
        #
        # NOTE what is deliberately NOT here: `fold_tensor_subset_ops_into_vector_transfers`.
        # Folding an `extract_slice` into a `transfer_read` is fine only while the slice is
        # minor-identity. A head-major transpose's INPUT slice is not: tiling
        # `(T,H,hs) -> (H,T,hs)` blocks the query rows and peels the head, so the input slice
        # is `(rows, 1, hs)` -- a unit dim in the MIDDLE. Folded, that becomes a read of the
        # whole tensor under `permutation_map = (d0,d1,d2) -> (d0,d2)`, which
        # `convert-vector-to-xegpu` does not convert; the store next to it does convert, and
        # the kernel then fails work-group distribution with "'xegpu.store_nd' op Value shape
        # [128, 64] is not consistent with tensor descriptor ...<32x32xf16>". Left unfolded,
        # the slice bufferizes to a rank-reduced `memref.subview` -- a strided 2-D memref,
        # which `xegpu.create_nd_tdesc` takes directly. See `_bufferize_keeping_subviews`,
        # which is the other half of this: the library `bufferize` re-does the same fold at
        # the memref level.
        with ir.InsertionPoint(transform.apply_patterns(loop).patterns):
            vector_transform.apply_patterns_vector_transfer_permutation_patterns()
            vector_transform.apply_patterns_vector_reduction_to_contract()
            vector_transform.apply_patterns_vector_sink_ops()
            vector_transform.apply_patterns_vector_fold_arith_extension()
        transform.yield_()
    lh_transform.cleanup(func)

    # Same trailing cleanup as `vectorize()`: hoist loop-invariant transfers out of the
    # k-loop, then drop the unit dims tiling introduced.
    lh_transform.loop_hoisting(match(func, ops={"scf.for"}))
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        vector_transform.apply_patterns_vector_cast_away_vector_leading_one_dim()
        vector_transform.apply_patterns_vector_drop_unit_dims_with_shape_cast()
    lh_transform.cleanup(func)


def _bufferize_keeping_transpose_subviews(mod, payload_func_name, plan):
    """One-shot bufferization; fold memref aliases into transfers EXCEPT in transposes.

    Same as `lowering_common.bufferize`, except its `fold_memref_alias_ops` step is applied
    per kernel and skipped for the transpose ones. Both halves of that are load-bearing.

    Why it must be skipped for a transpose: the step folds a `memref.subview` into the
    `vector.transfer_read` that reads it, and for a RANK-REDUCING subview that means
    re-expressing the drop as a projected `permutation_map`. A head-major transpose's input
    slice drops a MIDDLE dim -- tiling `(T,H,hs) -> (H,T,hs)` blocks the query rows and peels
    the head, giving an input slice of `(rows, 1, hs)` -- so folding yields a read of the
    whole tensor under `(d0,d1,d2) -> (d0,d2)`, which `convert-vector-to-xegpu` does not
    convert. The store beside it DOES convert, so the kernel ends up with a work-group-shaped
    `vector.transfer_read` feeding an already-distributed `xegpu.store_nd` and fails as
    "'xegpu.store_nd' op Value shape [128, 64] is not consistent with tensor descriptor
    ...<32x32xf16>". Left unfolded the slice stays a rank-reduced `memref.subview` -- a
    strided 2-D memref, which `xegpu.create_nd_tdesc` takes directly.

    Why it must be KEPT for the others: in a k-loop the subview's offset is the loop
    induction variable, and the `create_nd_tdesc` built from it gets hoisted by LICM, leaving
    "operand #0 does not dominate this use". Folding puts the varying part in the transfer's
    INDICES, so the descriptor is built once from the loop-invariant base. A transpose has no
    k-loop, which is why skipping it there costs nothing.
    """
    bufferization_transform.bufferization_eliminate_empty_tensors(mod)
    mod = bufferization_transform.bufferization_one_shot_bufferize(
        transform.any_op_t(),
        mod,
        function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
        bufferize_function_boundaries=True,
    )
    func = get_payload_func(mod, func_name=payload_func_name)
    foralls = match_and_split(func, ops={"scf.forall"}, nhandles=len(plan))
    for forall, entry in zip(foralls, plan):
        if entry["kind"] == "transpose":
            continue
        with ir.InsertionPoint(transform.apply_patterns(forall).patterns):
            memref_transform.apply_patterns_memref_fold_memref_alias_ops()
    transform.apply_cse(mod)
    canonicalize(mod)
    return mod


_ALIAS_OPS = ("memref.subview", "memref.expand_shape", "memref.collapse_shape")


def _memref_chain(value):
    """Every memref value from `value` back to its underlying buffer, closest first.

    Walks `subview` / `expand_shape` / `collapse_shape`. The whole chain is returned, not just
    the base, because the interesting value is usually in the MIDDLE: a tiled transpose reads
    `subview(subview(expand_shape(buffer)))`, and the op that the permutation describes is the
    `expand_shape` -- the buffer viewed at the transpose's own rank, not the flat buffer.
    """
    chain = [value]
    while True:
        owner = value.owner
        if not hasattr(owner, "operation"):
            return chain
        op = owner.operation
        if op.name not in _ALIAS_OPS:
            return chain
        value = op.operands[0]
        chain.append(value)


def _row_major_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _writes_memref(op, target):
    """True if `op` (or anything nested in it) writes `target` through a transfer_write."""
    found = [False]

    def walk(o):
        for region in o.regions:
            for blk in region.blocks:
                for inner in blk.operations:
                    if inner.operation.name == "vector.transfer_write":
                        if _memref_chain(inner.operands[1])[-1] == target:
                            found[0] = True
                            return
                    walk(inner)

    walk(op)
    return found[0]


def _top_level_index(use_owner, ops):
    """Index in `ops` of the top-level op containing `use_owner`, or None."""
    for i, candidate in enumerate(ops):
        if candidate.operation == use_owner.operation:
            return i
        matched = [False]

        def walk(o):
            for region in o.regions:
                for blk in region.blocks:
                    for inner in blk.operations:
                        if inner.operation == use_owner.operation:
                            matched[0] = True
                            return
                        walk(inner)

        walk(candidate)
        if matched[0]:
            return i
    return None


def replace_transpose_kernels_with_views(mod, payload_func_name, plan, kernel_params):
    """Delete every transpose KERNEL and present its result as a strided `memref` view.

    This is how BOTH reference implementations avoid these kernels, and neither does it by
    fusing: the permutation is carried in the STRIDES, not by moving data.
      * the hand payload (`llama3_payload.py:_heads_view_of`) builds
        `memref.expand_shape` + `memref.transpose` -- pure layout, no kernel -- and tiling then
        peels head h into a 2-D `memref<T x hs, strided<[C,1], offset: h*hs>>` that
        `xegpu.create_nd_tdesc` block-loads directly.
      * Inductor's captured graph passes `f16[1,32,256,64][524288,64,2048,1]` straight into the
        SDPA call: `aten.permute` is metadata, and head stride 64 / token stride 2048 is the
        same strided view.
    A torch-mlir payload cannot express it, because linalg-on-tensors has no view concept for a
    permutation of a VALUE -- `linalg.transpose` is a copy. But after bufferization the buffers
    exist, so the copy can be replaced by a view of the producer's buffer, which is what this
    does. It runs in Python between two halves of the schedule
    (`generic_schedule(..., stop_after_bufferize=True)` then `generic_schedule_tail`) because
    the transform dialect has no op for "replace this copy with a view".

    Returns the plan and params with the transpose entries dropped, for the tail -- which
    matches one `gpu.module` per remaining entry.
    """
    func = get_payload_func_op(mod, payload_func_name)
    block = func.regions[0].blocks[0]
    foralls = [o for o in block.operations if o.operation.name == "scf.forall"]
    if len(foralls) != len(plan):
        raise ValueError(f"{len(foralls)} foralls vs {len(plan)} plan entries")

    replaced_idx: set = set()
    for idx, (entry, forall) in enumerate(zip(list(plan), list(foralls))):
        if entry["kind"] != "transpose":
            continue
        perm = list(entry["permutation"])
        reads, writes = [], []
        for inner in forall.regions[0].blocks[0].operations:
            if inner.operation.name == "vector.transfer_read":
                reads.append(inner)
            elif inner.operation.name == "vector.transfer_write":
                writes.append(inner)
        if len(reads) != 1 or len(writes) != 1:
            continue  # not a plain copy kernel; leave it alone
        dst = _memref_chain(writes[0].operands[1])[-1]
        dst_type = ir.MemRefType(dst.type)
        # Pick the source value the permutation actually describes: same rank, and permuting
        # its shape gives the destination's. Searched from the BUFFER end outwards, so the
        # whole-buffer `expand_shape` wins over the per-tile `subview`s above it.
        src = None
        for cand in reversed(_memref_chain(reads[0].operands[0])):
            cand_type = ir.MemRefType(cand.type)
            if cand_type.rank == len(perm) and list(dst_type.shape) == [
                cand_type.shape[p] for p in perm
            ]:
                src = cand
                break
        if src is None:
            continue  # nothing in the chain matches the permutation; be conservative
        src_type = ir.MemRefType(src.type)
        # A view only works if every reader can take STRIDED memory. A `memref.collapse_shape`
        # cannot: collapsing dims requires them to be contiguous, and a permuted view never is
        # ("'memref.collapse_shape' op invalid source layout map or collapsing non-contiguous
        # dims"). That is exactly the attention OUTPUT transpose, whose (H,T,hs) result is
        # collapsed back to a 2-D (T, H*hs) operand for the output projection -- so it stays a
        # real copy kernel. The hand payload avoids it differently, by having the attention
        # kernel STORE through a strided view of the (T,C) buffer rather than transposing after.
        if any(u.owner.operation.name == "memref.collapse_shape" for u in dst.uses):
            continue
        # Build the view immediately BEFORE the copy kernel it replaces. That spot is after
        # the source (the kernel reads it) and before every reader of the destination (they
        # consume the kernel's result), so it dominates all of them. `ir.InsertionPoint(op)`
        # inserts BEFORE `op`, so anchoring on the source itself would not dominate.
        ip = ir.InsertionPoint(forall.operation)
        strides = _row_major_strides(list(src_type.shape))
        with ip:
            dims = [ir.AffineDimExpr.get(i) for i in range(len(perm))]
            pmap = ir.AffineMap.get(len(perm), 0, [dims[p] for p in perm])
            view_type = ir.MemRefType.get(
                [src_type.shape[p] for p in perm],
                src_type.element_type,
                layout=ir.StridedLayoutAttr.get(0, [strides[p] for p in perm]),
            )
            view = memref.transpose(view_type, src, pmap)
        # Point the readers at the view -- but ONLY the ones in this writer's live range.
        # A blind `replace_all_uses_with` is UNSOUND here: one-shot bufferization REUSES one
        # alloc for the K and V head-major results (their lifetimes do not overlap), so the
        # same buffer is written twice and read twice. Replacing every use would hand V's
        # readers K's view -- which shows up as "operand #0 does not dominate this use",
        # because one of those reads precedes the view. So: replace uses that sit after this
        # copy kernel and before the next op that WRITES the same buffer.
        ops = list(block.operations)
        here = next(
            i for i, o in enumerate(ops) if o.operation == forall.operation
        )
        next_writer = len(ops)
        for j in range(here + 1, len(ops)):
            if _writes_memref(ops[j], dst):
                next_writer = j
                break
        for use in list(dst.uses):
            owner = use.owner
            if owner.operation.name == "memref.dealloc":
                continue
            at = _top_level_index(owner, ops)
            if at is not None and here < at < next_writer:
                owner.operands[use.operand_number] = view
        forall.operation.erase()
        # The buffer and its dealloc go only once nothing writes it any more.
        remaining = [u for u in dst.uses if u.owner.operation.name != "memref.dealloc"]
        if not remaining:
            for use in list(dst.uses):
                if use.owner.operation.name == "memref.dealloc":
                    use.owner.operation.erase()
            if hasattr(dst.owner, "operation") and not list(dst.uses):
                dst.owner.operation.erase()
        replaced_idx.add(idx)

    # Only the entries actually replaced are dropped: a transpose whose reader needs
    # contiguous memory keeps its kernel, and the tail matches one `gpu.module` per entry.
    keep = [i for i in range(len(plan)) if i not in replaced_idx]
    return (
        [plan[i] for i in keep],
        [kernel_params[i] for i in keep],
        len(replaced_idx),
    )


def redirect_staged_destination_copies(mod, payload_func_name: str) -> int:
    """Delete the staging buffer bufferization inserts for a SECOND destination slice.

    WHY THIS EXISTS, for the record (2026-09-30 review). The staging is a one-shot
    bufferization limitation, not a property of the payload: the tiled forall carries one
    shared_out per slice and writes them back through an `insert_slice` chain, and the
    analysis has no rule that lets that chain bufferize in place. Fixing it at the source needs
    two pieces, neither small: a tensor-level rewrite that gives the forall the whole destination
    as its single shared_out, and a disjoint-inserts rule in
    `OneShotAnalysis.cpp::areNonConflictingSubsets` (drafted with lit tests as
    `one-shot-bufferize-disjoint-inserts.patch`, next to this file; unbuilt). Judged too heavy
    for the gain, so this memref-level fix-up stays.

    Companion to `fuse_rope_halves`, and the reason that rewrite can run at all. One-shot
    bufferization cannot prove that two `extract_slice` destinations of one buffer are
    disjoint subsets, so for the second one it allocates a staging buffer and brackets the
    kernel with host copies:

        %subview = memref.subview %dst[0, half] ...
        %alloc   = memref.alloc()
        memref.copy %subview, %alloc          <- in
        scf.forall { ... vector.transfer_write %v, %alloc[...] }
        memref.copy %alloc, %subview          <- out

    Both copies are host accesses over DEVICE memory, which fault. This points the kernel's
    write at `%subview` itself and drops the alloc and both copies.

    WHY IT IS SOUND, and note it needs no coverage argument: after the redirect the kernel
    writes exactly the elements it wrote before, into the buffer those elements came from, so
    whatever it does NOT write simply keeps the value it already had -- which is precisely
    what the copy IN was there to preserve. The copy in has become a self-copy and the copy
    out the identity. (This is also why the kernel must not READ the staging buffer: then the
    redirect could observe a write made earlier in the same loop. Allocs that are read are
    skipped.)

    Disjointness from the OTHER destination -- the one bufferization did keep in place -- is
    not re-derived here; it comes from `fuse_rope_halves`, which only ever merges two halves
    writing offsets 0 and `half` of the same row range.

    Runs in Python after bufferization and before the tail, the same slot as
    `replace_transpose_kernels_with_views` -- which is what keeps it cheap: the kernel is
    still an `scf.forall` in the host function, so no outlined `gpu.func` signature changes.
    Returns the number of staging buffers removed.
    """
    func = get_payload_func_op(mod, payload_func_name)
    block = func.regions[0].blocks[0]
    removed = 0
    # Iterate the copies OUT, not the allocs: bufferization REUSES one staging buffer across
    # layers (measured -- at 2 layers `%alloc_15` is staged once per layer), so an alloc can
    # have several, and an alloc-keyed pass skips exactly the multi-layer case. Each copy out
    # is handled on its own, windowed to its live range, the same discipline
    # `replace_transpose_kernels_with_views` needs for the shared K/V buffer.
    for copy_out in list(block.operations):
        if copy_out.operation.name != "memref.copy":
            continue
        buf, dest = copy_out.operands[0], copy_out.operands[1]  # (source, target)
        if not isinstance(buf, ir.OpResult) or buf.owner.operation.name != "memref.alloc":
            continue
        if not isinstance(dest, ir.OpResult) or dest.owner.operation.name != "memref.subview":
            continue
        alloc = buf.owner
        dest_type, buf_type = ir.MemRefType(dest.type), ir.MemRefType(buf.type)
        if list(dest_type.shape) != list(buf_type.shape):
            continue
        if list(ir.DenseI64ArrayAttr(dest.owner.attributes["static_strides"])) != [1] * (
            dest_type.rank
        ):
            continue  # non-unit strides would not map the kernel's indices 1:1

        # This copy out's live range starts after the PREVIOUS copy out of the same buffer:
        # each range is [optional copy in] -> kernel writes -> copy out, so those copies are
        # the boundaries.
        ops = list(block.operations)
        here = _top_level_index(copy_out, ops)
        window_start = -1
        for j in range(here - 1, -1, -1):
            op_j = ops[j]
            if op_j.operation.name == "memref.copy" and op_j.operands[0] == buf:
                window_start = j
                break

        copies_in, writes, disqualified = [], [], False
        for use in buf.uses:
            owner, name = use.owner, use.owner.operation.name
            at = _top_level_index(owner, ops)
            if owner.operation == copy_out.operation or name == "memref.dealloc":
                continue
            if at is None or not (window_start < at < here):
                continue  # belongs to another copy out's live range
            if name == "memref.copy" and use.operand_number == 1:
                copies_in.append(owner)
            elif name == "vector.transfer_write" and use.operand_number == 1:
                writes.append(owner)
            else:
                disqualified = True  # anything else (notably a READ) makes this unsafe
        if disqualified or not writes or len(copies_in) > 1:
            continue
        if copies_in and copies_in[0].operands[0] != dest:
            continue  # staged against a different buffer than it is written back to

        # The copy OUT sits after the kernel, and so may its `memref.subview` -- in a whole
        # block bufferization emits the subview next to the copy that uses it, below the
        # loop. Writing to it from inside the loop then does not dominate, so hoist it above
        # the first kernel that will write it. Only legal while its own operands already
        # dominate that point (the base buffer and any dynamic offsets).
        ops = list(block.operations)
        first_write = min(_top_level_index(w, ops) for w in writes)
        sub_op = dest.owner
        sub_at = _top_level_index(sub_op, ops)
        if sub_at is not None and sub_at > first_write:
            if any(
                isinstance(o, ir.OpResult)
                and (_top_level_index(o.owner, ops) or 0) >= first_write
                for o in sub_op.operands
            ):
                continue  # cannot hoist it that far; leave this one staged
            sub_op.operation.move_before(ops[first_write].operation)

        for write in writes:
            write.operands[1] = dest
        for copy in copies_in + [copy_out]:
            copy.operation.erase()
        # The buffer and its dealloc go only once NOTHING else uses it -- with a staging
        # buffer shared across layers the later live ranges still do.
        if not [u for u in buf.uses if u.owner.operation.name != "memref.dealloc"]:
            for use in list(buf.uses):
                if use.owner.operation.name == "memref.dealloc":
                    use.owner.operation.erase()
            if not list(buf.uses):
                alloc.operation.erase()
        removed += 1
    return removed


def _index_value(v):
    """A hashable identity for an index operand: the SSA value, or the constant it holds."""
    owner = v.owner
    if hasattr(owner, "operation") and owner.operation.name == "arith.constant":
        return ("const", ir.IntegerAttr(owner.attributes["value"]).value)
    return ("ssa", v)


def fold_gqa_broadcasts(mod, payload_func_name, plan, kernel_params):
    """Delete the GQA K/V broadcast kernels; their readers index the source directly.

    torch-mlir spells `repeat_interleave(n_rep, 0)` as a broadcast `linalg.generic` that
    MATERIALIZES K and V at the query-head count -- two copy kernels per layer and 4x the KV
    read traffic, against a hand payload whose K/V indexing map simply OMITS the `rep` dim and
    an Inductor capture that hands SDPA the 8-head tensor as-is. Every route to the zero-copy
    form at the TENSOR level was measured and rejected (`_gqa_probe.py`, §3d of the plan doc:
    the fold needs `linalg-fuse-elementwise-ops`, which cannot be scoped and rewrites every
    projection into a generic; the flat `(n_kv, n_rep*T, hs)` payload needs two library
    changes and dies in bufferization on a middle unit dim).

    After bufferization the problem is trivial, because both sides are already explicit:

        scf.forall (%kv, %rep, %t) {                              <- the broadcast kernel
          %x = vector.transfer_read %k_view[%kv, %t, 0]
          vector.transfer_write %x, %k_bcast[%kv, %rep, %t, 0]
        }
        ...
        %kv, %rep = affine.delinearize_index %head into (n_kv, n_rep)   <- inside attention
        %k = vector.transfer_read %k_bcast[%kv, %rep, %j, 0]

    The broadcast writes `dst[kv, rep, t]` from `src[kv, t]`, so a reader of `dst[a, b, c]` is
    a reader of `src[a, c]`: rewrite each reader in the kernel's live range to index the
    source with the broadcast dims dropped, and the copy kernel, its buffer and its dealloc
    are dead. The source here is the head-major `memref.transpose` view that
    `replace_transpose_kernels_with_views` already made, so K and V are read straight out of
    the projection buffers -- exactly the hand payload's `_grouped_heads_view_of` -- and the
    attention kernel's three loads (Q, K, V) all become strided reads of the same kind, which
    is what its layout annotation expects.

    Recognised shape: a forall holding exactly one `transfer_read` + one `transfer_write`,
    whose write indices are a superset of its read indices IN ORDER (the extra ones are the
    broadcast dims), the broadcast dims all outside the vector-covered trailing dims, and
    trailing shapes equal. The rewrite is windowed to the copy's live range, because
    bufferization reuses one alloc across layers (the same trap the transpose views and the
    RoPE staging buffers both hit).

    Returns the plan and params with the broadcast entries dropped, for the tail.
    """
    func = get_payload_func_op(mod, payload_func_name)
    block = func.regions[0].blocks[0]
    foralls = [o for o in block.operations if o.operation.name == "scf.forall"]
    if len(foralls) != len(plan):
        raise ValueError(f"{len(foralls)} foralls vs {len(plan)} plan entries")

    replaced_idx: set = set()
    for idx, (entry, forall) in enumerate(zip(list(plan), list(foralls))):
        if entry["kind"] != "elementwise":
            continue
        body = forall.regions[0].blocks[0]
        body_ops = [
            o for o in body.operations if o.operation.name != "scf.forall.in_parallel"
        ]
        reads = [o for o in body_ops if o.operation.name == "vector.transfer_read"]
        writes = [o for o in body_ops if o.operation.name == "vector.transfer_write"]
        if len(reads) != 1 or len(writes) != 1 or len(body_ops) != 2:
            continue  # not a plain copy kernel
        read, write = reads[0], writes[0]
        if write.operands[0] != read.results[0]:
            continue
        src, dst = read.operands[0], write.operands[1]
        src_type, dst_type = ir.MemRefType(src.type), ir.MemRefType(dst.type)
        vec_rank = ir.VectorType(read.results[0].type).rank
        if dst_type.rank <= src_type.rank:
            continue  # not a broadcast
        # transfer_read/write operands: base, indices..., padding/vector, [mask].
        r_idx = [read.operands[1 + i] for i in range(src_type.rank)]
        w_idx = [write.operands[2 + i] for i in range(dst_type.rank)]
        # Map each read index to the write index carrying the same value, in order; the
        # write positions left over are the broadcast dims.
        mapping, w_pos = [], 0
        for rv in r_idx:
            key = _index_value(rv)
            while w_pos < len(w_idx) and _index_value(w_idx[w_pos]) != key:
                w_pos += 1
            if w_pos == len(w_idx):
                break
            mapping.append(w_pos)
            w_pos += 1
        if len(mapping) != src_type.rank:
            continue
        bcast_dims = sorted(set(range(dst_type.rank)) - set(mapping))
        if any(d >= dst_type.rank - vec_rank for d in bcast_dims):
            continue  # a broadcast inside the vector-covered dims is a real data expansion
        if list(src_type.shape[-vec_rank:]) != list(dst_type.shape[-vec_rank:]):
            continue
        if any(src_type.shape[i] != dst_type.shape[m] for i, m in enumerate(mapping)):
            continue
        if not isinstance(dst, ir.OpResult) or dst.owner.operation.name != "memref.alloc":
            continue

        # Live range of this copy: readers after it and before the next writer of `dst`.
        ops = list(block.operations)
        here = _top_level_index(forall, ops)
        next_writer = len(ops)
        for j in range(here + 1, len(ops)):
            if _writes_memref(ops[j], dst):
                next_writer = j
                break
        readers = []
        for use in list(dst.uses):
            owner = use.owner
            name = owner.operation.name
            if owner.operation == forall.operation or name == "memref.dealloc":
                continue
            at = _top_level_index(owner, ops)
            if at is None or not (here < at < next_writer):
                continue
            if name != "vector.transfer_read" or use.operand_number != 0:
                readers = None  # something other than a plain read; leave it materialized
                break
            readers.append(owner)
        if not readers:
            continue
        # Every reader must cover the same trailing dims with a minor-identity map, i.e. be
        # the same kind of read the broadcast kernel itself did.
        ok = True
        for rd in readers:
            if ir.VectorType(rd.results[0].type).rank != vec_rank:
                ok = False
            pmap = ir.AffineMapAttr(rd.attributes["permutation_map"]).value
            if pmap != ir.AffineMap.get_minor_identity(dst_type.rank, vec_rank):
                ok = False
        if not ok:
            continue
        # The SOURCE must still hold the same data when the reader runs. The copy kernel is
        # what decoupled the two, and bufferization exploits that: without the transpose
        # views, K's and V's head-major results share ONE alloc (their lifetimes do not
        # overlap), so by the time attention runs, K's source buffer holds V. Folding then
        # reads V twice -- and the two identical loads CSE into one, which the attention
        # annotation reports as "expected to contain 3 payloads but it contains 2". So if
        # anything writes the source's underlying buffer between the copy and a reader, keep
        # this one materialized. (Windowing the destination alone is not enough.)
        src_base = src
        while True:
            owner = src_base.owner
            if not hasattr(owner, "operation") or owner.operation.name not in (
                *_ALIAS_OPS,
                "memref.transpose",
            ):
                break
            src_base = owner.operation.operands[0]
        last_reader = max(_top_level_index(rd, ops) for rd in readers)
        if any(_writes_memref(ops[j], src_base) for j in range(here + 1, last_reader + 1)):
            continue

        new_map = ir.AffineMap.get_minor_identity(src_type.rank, vec_rank)
        for rd in readers:
            old_idx = [rd.operands[1 + i] for i in range(dst_type.rank)]
            new_idx = [old_idx[m] for m in mapping]
            padding = rd.operands[1 + dst_type.rank]
            with ir.InsertionPoint(rd):
                new = vector.TransferReadOp(
                    rd.results[0].type,
                    src,
                    new_idx,
                    new_map,
                    padding,
                    rd.attributes["in_bounds"],
                )
            rd.results[0].replace_all_uses_with(new.result)
            rd.operation.erase()
        forall.operation.erase()
        remaining = [u for u in dst.uses if u.owner.operation.name != "memref.dealloc"]
        if not remaining:
            for use in list(dst.uses):
                if use.owner.operation.name == "memref.dealloc":
                    use.owner.operation.erase()
            if not list(dst.uses):
                dst.owner.operation.erase()
        replaced_idx.add(idx)

    keep = [i for i in range(len(plan)) if i not in replaced_idx]
    return (
        [plan[i] for i in keep],
        [kernel_params[i] for i in keep],
        len(replaced_idx),
    )


def _emit_tail_after_bufferize(
    mod, payload_func_name, plan, kernel_params, gpu_specs, has_reduction
):
    """Emit the post-bufferization half of the tail: promote, outline, per-kernel XeGPU.

    Split out of `generic_schedule` so it can also run as a SECOND schedule, after a Python
    pass has rewritten the bufferized module (see `replace_transpose_kernels_with_views`).
    `plan` and `kernel_params` must line up with the `scf.forall`s that are actually left.
    """
    if has_reduction:
        pfunc = get_payload_func(mod, func_name=payload_func_name)
        pfunc = apply_registered_pass(
            pfunc,
            "promote-buffers-to-stack",
            options={
                "max-alloc-size-in-bytes": "8192",
                "max-rank-of-allocated-memref": "2",
            },
        )
    convert_allocs_to_gpu(mod, payload_func_name=payload_func_name)
    convert_to_gpu_launch(mod, payload_func_name=payload_func_name)
    mod = outline_gpu_function(
        mod,
        payload_func_name=payload_func_name,
        gpu_specs=gpu_specs,
        params=kernel_params,
    )

    # vector -> xegpu per kernel, so SLM stays selective.
    mod = apply_registered_pass(
        mod, "xevm-attach-target", options={"O": "3", "chip": "pvc"}
    )
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=len(plan))
    for gpu_mod, entry, params in zip(gpu_mods, plan, kernel_params):
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        if entry["kind"] == "reduction":
            transform_ext.update_address_space(
                match(gpu_func, ops={"memref.alloca"}), address_space=3
            )
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)
        with lh_transform.foreach(match(gpu_func, ops={"scf.for"})) as loop:
            transform.apply_licm(loop)
            transform.yield_()

        if entry["kind"] == "attention":
            # Reused from the hand schedule rather than re-derived: it anchors the one
            # store_nd, the three load_nd (Q hoisted, then K and V in the flash loop)
            # and both dpas ops.
            xegpu_fa_annotation(gpu_func, params)
        elif entry["kind"] in _CONTRACTION_KINDS:
            xegpu_wg_annotation_for_mlp_layer(
                gpu_func, gpu_specs=gpu_specs, **params
            )
        elif entry["kind"] == "reduction":
            # Rows across subgroups, rss-wide blocks per subgroup. Anchor the stores;
            # layout propagation derives the loads.
            sg_layout = [params["wg_m"] // params["sg_m"], 1]
            sg_data = [params["sg_m"], params["rss"]]
            for op_name in ("xegpu.store_nd", "xegpu.store_matrix"):
                xegpu_transform.set_anchor_layout(
                    match(gpu_func, ops={op_name}),
                    sg_layout=sg_layout,
                    sg_data=sg_data,
                )
        else:
            # 2-D for every elementwise kernel, including ones that started rank > 2:
            # the unit-extent folding after tiling collapsed those to 2-D.
            xegpu_wg_annotation_for_elemwise_layer(
                gpu_func, gpu_specs=gpu_specs, **params
            )
    transform.apply_cse(mod)
    canonicalize(mod)
    return mod


def generic_schedule_tail(
    payload_func_name: str,
    device: str,
    plan: list[dict],
    kernel_params: list[dict],
) -> ir.Module:
    """The post-bufferization half of `generic_schedule`, as a standalone schedule.

    Used with `generic_schedule(..., stop_after_bufferize=True)` when something has to run on
    the bufferized module in Python -- which is the only way to do a MEMREF-level rewrite,
    since the transform dialect has no op for "replace this copy with a view".
    """
    gpu_specs = XeGPUParameterSelector(device=device).gpu_specs
    has_reduction = any(e["kind"] == "reduction" for e in plan)
    with schedule_boilerplate() as (schedule, named_seq):
        func = get_payload_func(named_seq.bodyTarget, func_name=payload_func_name)
        mod = transform.get_parent_op(
            transform.AnyOpType.get(), func, op_name="builtin.module", deduplicate=True
        )
        _emit_tail_after_bufferize(
            mod, payload_func_name, plan, kernel_params, gpu_specs, has_reduction
        )
        transform.yield_()
    return schedule


def _emit_empty_tensor_elimination(func, mod):
    """Decompose `tensor.concat` and eliminate empty tensors, BEFORE any tiling.

    Split out of `generic_schedule` so it can also run as a standalone schedule ahead of a
    Python rewrite that needs the destinations already resolved (`fuse_rope_halves`).
    Running it twice is harmless: rounds past the chain depth are no-ops.
    """
    # Decompose `tensor.concat` into `insert_slice`s. A concat is real data movement
    # with no tiling interface, so it bufferizes into copies that sit OUTSIDE every
    # kernel -- host accesses over device buffers, which fault at run time. An
    # `insert_slice` can instead be folded away entirely by the elimination below.
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        tensor_transform.apply_patterns_tensor_decompose_concat()
    lh_transform.cleanup(func)

    # Eliminate empty tensors HERE, before tiling -- `bufferize()` runs this too, but
    # by then it is too late. A payload that assembles its result from slices (RoPE
    # writing two rotated halves) reads as `tensor.empty` -> `insert_slice` ->
    # `insert_slice` -> `materialize_in_destination(arg0)`. Run now, elimination
    # rewrites each producer's `outs` to an `extract_slice` OF arg0, so the kernels
    # write their halves straight into the output and both `insert_slice`s become
    # no-ops that bufferize away. Run after tiling, those same empties have become
    # `scf.forall` inits, the insert_slices survive into the host function and get
    # VECTORIZED there -- a `vector<1024x32xf16>` transfer pair over device memory,
    # which faults.
    #
    # Iterated because one round only peels one level of an insert_slice chain: it
    # rewrites the chain's destination to arg0, but a producer it already redirected
    # still points at the now-dead intermediate empty. `fold_tensor_empty` between
    # rounds is what makes progress possible -- it turns that stale
    # `extract_slice(empty)` back into a plain `empty` for the next round to eliminate.
    # (It is a pattern set, not a canonicalization, so `cleanup` alone does nothing and
    # the loop would spin without it.) Rounds past the chain depth are no-ops.
    for _ in range(_EMPTY_ELIM_ROUNDS):
        bufferization_transform.bufferization_eliminate_empty_tensors(mod)
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            tensor_transform.apply_patterns_tensor_fold_tensor_empty()
        lh_transform.cleanup(func)


def eliminate_empty_tensors_schedule(payload_func_name: str) -> ir.Module:
    """Just the empty-tensor-elimination preamble, as its own schedule.

    `fuse_rope_halves` has to run with the RoPE destination already rooted at the real
    output buffer: while it is still an intermediate `tensor.empty`, the `fold_tensor_empty`
    pattern collapses the `extract_slice(empty)` destinations the merged op needs back into
    bare `empty`s, and the upper half then reaches the output through a HOST `insert_slice`
    over device memory (which faults). So drivers that fuse run this first, then fuse, then
    `classify_payload`.
    """
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        func = get_payload_func(named_seq.bodyTarget, func_name=payload_func_name)
        mod = transform.get_parent_op(
            anytype, func, op_name="builtin.module", deduplicate=True
        )
        _emit_empty_tensor_elimination(func, mod)
        transform.yield_()
    return schedule


def generic_schedule(
    payload_func_name: str,
    device: str,
    plan: list[dict],
    kernel_params: list[dict],
    inspect: bool = False,
    stop_after_bufferize: bool = False,
) -> ir.Module:
    """Build the transform schedule for `plan`: tile every kernel, then one shared tail."""
    gpu_specs = XeGPUParameterSelector(device=device).gpu_specs

    supported = {"matmul", "elementwise", "reduction", "transpose", "attention"}
    unsupported = {e["kind"] for e in plan} - supported
    if unsupported:
        raise NotImplementedError(f"op classes not handled yet: {sorted(unsupported)}")
    has_reduction = any(e["kind"] == "reduction" for e in plan)

    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        func = get_payload_func(named_seq.bodyTarget, func_name=payload_func_name)
        mod = transform.get_parent_op(
            anytype, func, op_name="builtin.module", deduplicate=True
        )

        _emit_empty_tensor_elimination(func, mod)

        # One handle per op, per op name. Handles stay in program order and stay valid
        # while other ops are tiled, so they can be consumed in plan order.
        # One handle per op, per op name, INDEXED BY IR POSITION. Popping in plan order is
        # wrong: a group can own a member that sits earlier in the IR than another group's
        # member of the same name (attention absorbs the K^T transpose, which precedes
        # nothing in particular), and then every later handle of that name is off by one --
        # which silently hands a kernel another kernel's tile params and fails as
        # "'xegpu.load_nd' op TensorDesc shape is not distributable with the layout".
        queues: dict[str, list] = {}
        for op_name in {m for e in plan for m in e["members"]}:
            count = sum(1 for e in plan for m in e["members"] if m == op_name)
            queues[op_name] = list(match_and_split(func, ops={op_name}, nhandles=count))

        for entry, params in zip(plan, kernel_params):
            # The group's LAST member is its output, and that is what gets tiled.
            member, ordinal = entry["handles"][-1]
            handle = queues[member][ordinal]

            if entry["kind"] == "reduction":
                _tile_one_reduction(anytype, handle, params["wg_m"], params["rss"])
                continue
            if entry["kind"] == "attention":
                _tile_one_attention(
                    anytype,
                    handle,
                    entry["shape"],
                    params,
                    absorbed_transpose="linalg.transpose" in entry["members"],
                )
                continue

            if entry["kind"] in _CONTRACTION_KINDS:
                tile_sizes = [params["wg_m"], params["wg_n"]]
            elif entry["kind"] == "transpose":
                tile_sizes = _transpose_tile_sizes(entry["shape"], params)
            else:
                tile_sizes = _ew_tile_sizes(entry["shape"], params)
            _, [wg_loop], _ = lh_transform.tile(
                handle,
                tile_sizes=tile_sizes,
                fuse_producers=not entry.get("join", False),
                use_forall=True,
                apply_cleanup=False,
            )
            if entry["kind"] in _CONTRACTION_KINDS:
                lh_transform.tile(
                    match(wg_loop, ops={entry["op"]}),
                    tile_sizes=[0, 0, params["k_tile"]],
                )
        lh_transform.cleanup(func)

        # Generalize tiled transposes to generics. `linalg.transpose` is a NAMED op and the
        # unit-extent folding below only rewrites generics, so a tiled transpose keeps its
        # rank-3 slices -- and because the read is permuted, the unit dim lands in the
        # MIDDLE of the input slice (`tensor_desc<128x1x64xf16>`), which fails with
        # "'xegpu.load_nd' op TensorDesc shape is not distributable with the layout".
        # Generalizing gives the equivalent permuted-read generic, which the folding can
        # collapse to 2-D. Note this happens AFTER tiling on purpose: classification and
        # tile-size choice need to see a `linalg.transpose` (its own kind, blocked by output
        # rows), not an anonymous all-parallel generic that would be taken for elementwise.
        if any(e["kind"] == "transpose" for e in plan):
            structured.structured_generalize(
                anytype, match(func, ops={"linalg.transpose"})
            )
            lh_transform.cleanup(func)

        # Collapse rank > 2 kernels to 2-D. XeGPU's work-group->subgroup distribution
        # wants 2-D vectors: a rank-3 tensor desc rejects a 2-D layout outright, and a
        # rank-matched layout is accepted but then distributes into an inconsistent
        # vector.shape_cast. Tiling above already made every middle dim 1
        # (`_ew_tile_sizes`), so folding unit extents HERE -- after tiling, on the tiled
        # ops -- rewrites those generics to rank 2 and the ordinary 2-D layouts apply.
        #
        # Note this is the same pattern that must NOT run before tiling: there it
        # collapses a reduction's accumulator and breaks vectorization (see
        # torch_mlir_rmsnorm_gpu.py). Timing is what makes it safe, and it is gated on a
        # rank > 2 kernel actually being present to keep the blast radius small.
        if any(_rank(e["shape"]) > 2 for e in plan):
            with ir.InsertionPoint(transform.apply_patterns(func).patterns):
                structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
            lh_transform.cleanup(func)

        if inspect:
            transform.yield_()
            return schedule

        # The tail, spelled out rather than via `vectorize_bufferize_and_outline_gpu_func`,
        # for two reasons: vectorization is scoped (below), and a reduction needs
        # `promote-buffers-to-stack` inserted after bufferization, which the reusable
        # helper omits (the hand tail runs it).
        _vectorize_kernels_only(func)
        mod = _bufferize_keeping_transpose_subviews(mod, payload_func_name, plan)
        if stop_after_bufferize:
            transform.yield_()
            return schedule
        _emit_tail_after_bufferize(
            mod, payload_func_name, plan, kernel_params, gpu_specs, has_reduction
        )
        transform.yield_()
    return schedule
