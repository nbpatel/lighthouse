"""XeGPU schedule for a torch-mlir Llama-3 payload.

Counterpart of the hand-written `llama3_schedule.py`. That schedule is told what each op is
via a `kinds` list kept alongside the hand payload; here `classify_payload` derives the
kernel plan by walking the imported IR, so any torch-mlir payload built from the same op
classes can be scheduled.

Flow (payload / schedule / driver split, as in nanoGPT):
    payload   <- torch-mlir import of llama3_torch_model.py
    schedule  <- this file
    driver    <- torch_mlir_llama_gpu.py

Steps:

1. `classify_payload` returns one plan entry per GPU kernel: an op class plus the ops the
   kernel owns. Classes are derived from the IR (matmul / batch_matmul / contraction /
   reduction / elementwise / transpose / attention); see `_classify_generic` for why a
   reduction iterator alone does not identify a reduction.

2. Ops are grouped into kernels by dataflow: an elementwise op joins the entry producing its
   input when both share an iteration space. That puts casts and activations in their
   producer's kernel and makes RMSNorm one kernel (mandatory: a reduced intermediate cannot
   be a kernel output). The global `linalg-fuse-elementwise-ops` pass is deliberately not
   used: it rewrites `linalg.matmul` into generics and loses the DPAS path.

3. Each group is tiled into one work-group `scf.forall` through its LAST member with
   `fuse_producers=True`, in topological order. Tiling through the last member pulls the
   rest of the group in, `fuse_producers` also pulls in the `linalg.fill` accumulators
   (one left outside a forall page-faults), and topological order keeps earlier kernels
   from being re-absorbed into later ones.

4. One shared tail (vectorize, bufferize, outline), then per-kernel XeGPU layout anchors.
   Vectorization is scoped to the foralls (`_vectorize_kernels_only`), a reduction gets
   `promote-buffers-to-stack` after bufferization, and vector->xegpu runs per kernel so
   only reductions get shared local memory.

5. Memref-level rewrites run in Python between bufferization and the tail to remove copy
   kernels that cannot be avoided at the tensor level: transposes become strided views
   (`replace_transpose_kernels_with_views`), GQA broadcasts fold into their readers
   (`fold_gqa_broadcasts`), and RoPE's two half-writes are merged into one kernel
   (`fuse_sibling_slice_writers` + `redirect_staged_destination_copies`).
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

# The fused-attention XeGPU layouts are shared with the hand schedule rather than copied.
from llama3_schedule import xegpu_fa_annotation

# Elementwise geometry. wg_n is bounded on purpose: a row-only tile at FFN width (8192)
# makes one per-subgroup vector that overruns the register file.
EW_PARAMS = {
    "wg_m": 128,
    "wg_n": 256,
    "sg_m": 32,
    "sg_n": 32,
    "load_m": 8,
    "load_n": 16,
}
# Reduction geometry: wg_m/sg_m split the rows (wg_n = sg_n = 1), rss is the per-subgroup
# reduction step.
RED_PARAMS = {"wg_m": 64, "sg_m": 8, "wg_n": 1, "sg_n": 1, "rss": 16}
# Fused-attention geometry for d_head == 64 (values from kernel_bench.py). wg_m/sg_m/wg_n/sg_n
# give the shared tail its thread count ((128/16)*(1/1)*16 = 128 = num_subgroups *
# subgroup_size); wg_rows/sg_rows/n_head are what `xegpu_fa_annotation` reads.
FA_PARAMS = {
    "wg_m": 128,
    "sg_m": 16,
    "wg_n": 1,
    "sg_n": 1,
    "wg_rows": 128,
    "sg_rows": 16,
    "n_head": 64,  # d_head; the annotation's key name
    "inner_loop_tile_size": 64,
    "causal": False,
}

# Rounds of empty-tensor elimination before tiling. One round peels one level of an
# `insert_slice` chain; RoPE's two halves are the deepest chain in the payload.
_EMPTY_ELIM_ROUNDS = 4

# linalg ops that get a kernel. linalg.fill is an accumulator and is fused into its consumer.
_TILED_OPS = {
    "linalg.matmul": "matmul",
    "linalg.batch_matmul": "batch_matmul",
    "linalg.transpose": "transpose",
    "linalg.generic": None,  # decided by _classify_generic
}
_CONTRACTION_KINDS = {"matmul", "batch_matmul", "contraction"}

# Metadata-only ops: no kernel, but they carry dataflow (torch-mlir puts a view between a
# projection and its RoPE), so grouping has to see through them. `linalg.transpose` is real
# data movement and gets its own kernel instead.
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

    torch-mlir often emits a projection as a generic with a reduction iterator (the head
    reshape folded into the contraction), so the iterator alone would send a matmul down the
    RMSNorm path. A contraction reduces over a dim shared by two or more inputs; a true
    reduction has a single input. DPS ops have one init per result, so inputs are
    `operands - results`.
    """
    if "reduction" not in str(op.attributes["iterator_types"]):
        return "elementwise"
    n_inputs = len(op.operands) - len(op.results)
    return "contraction" if n_inputs >= 2 else "reduction"


def get_payload_func_op(mod: ir.Module, func_name: str = "main"):
    """The payload `func.func` as an IR op, for the rewrites that run in Python."""
    for op in mod.body.operations:
        if op.operation.name == "func.func" and func_name in str(
            op.attributes["sym_name"]
        ):
            return op
    raise ValueError(f"no func.func {func_name!r}")


def _identity_maps(op) -> bool:
    """True when every indexing map of `op` is the identity, i.e. its body can be inlined
    into another op over the same shape."""
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
    # A body that reads its `outs` element cannot be inlined into a fresh destination.
    body = op.regions[0].blocks[0]
    return len(list(body.arguments[-1].uses)) == 0


def _elementwise_tree(root, shape):
    """Collect the elementwise generics that compute `root`, plus the values they read.

    Returns `(members, leaves)` with members in topological order, or None if `root` is not
    produced by an inlinable elementwise generic. An op is a member only when all its uses
    are inside the tree; otherwise it stays a real op and is treated as a leaf.
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
    # Topological order: emit an op once everything it reads is emitted.
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
            produced_here = (
                isinstance(operand, ir.OpResult) and operand.owner in member_set
            )
            if not produced_here and operand not in leaves:
                leaves.append(operand)
    return ordered, leaves


def _inline_elementwise_body(op, operand_scalars, scalars):
    """Clone `op`'s body at the current insertion point and return the scalar it yields.

    `operand_scalars` stand in for the op's tensor inputs, in order; `scalars` maps
    already-inlined members' results to their scalars. The bindings have no value-remapping
    clone, so each body op is recreated by name; inlinable bodies hold only scalar
    arithmetic, so that is enough.
    """
    body = op.regions[0].blocks[0]
    local = dict(scalars)
    for i, scalar in enumerate(operand_scalars):
        local[body.arguments[i]] = scalar
    for inner in body.operations:
        if inner.operation.name == "linalg.yield":
            return local.get(inner.operands[0], inner.operands[0])
        # Indexing the attribute map gives NamedAttribute; iterating it gives bare names.
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
        # DenseI64ArrayAttr (`array<i64: 0, 0>`), not ArrayAttr.
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
    """Match a chain of two or more `tensor.insert_slice`s writing static, unit-stride,
    pairwise-disjoint, equally shaped slices of one destination. `op` is the candidate LAST
    insert; the chain is followed backwards through each insert's destination. Returns
    `(chain, shape)` with `chain` in program order, or None. Nothing here is RoPE-specific.
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
        # An intermediate may only be read by the next insert and by the `extract_slice`s
        # that supply later siblings' `outs`. Anything else would observe a value the merge
        # does not preserve.
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
                oi[d] + shape[d] <= oj[d] or oj[d] + shape[d] <= oi[d]
                for d in range(rank)
            ):
                return None  # overlapping writes: the later one wins, a merge would not keep that
    return chain, shape


def fuse_sibling_slice_writers(mod: ir.Module, func_name: str = "main") -> int:
    """Merge sibling elementwise chains that write disjoint slices of one destination into
    ONE multi-result `linalg.generic`.

    Motivation: RoPE writes `out[:, :half]` and `out[:, half:]` as two independent chains,
    which costs 2 kernels per RoPE against Inductor's 1. Grouping cannot fix it (the lower
    half is not a producer of the upper one, so tiling through the last member leaves it
    outside every kernel), and `transform.loop.fuse_sibling` rejects the two tiled foralls
    on a false dominance dependence through the `insert_slice` chain. A single two-result
    generic over the same `(rows, half)` space is ordinary elementwise and tiles like any
    other op; it is the form the hand payload's `Builder.rope` already uses.

    Runs on tensor-level IR after empty-tensor elimination (so the destination is already
    the real output buffer) and before `classify_payload`. Must be paired with
    `redirect_staged_destination_copies`: one-shot bufferization cannot prove the two
    destination slices disjoint and stages the second one through a host-copied alloc,
    which faults on device memory.

    Returns the number of chains fused.
    """
    func = get_payload_func_op(mod, func_name)
    block = func.regions[0].blocks[0]
    fused = 0
    # Visit last-insert-first so a chain is taken whole rather than as a sub-chain. Each
    # merge erases ops, so the op list is re-snapshotted after every merge: touching an
    # erased OpView is a use-after-free.
    progress = True
    while progress:
        progress = False
        ops = list(block.operations)
        block_ops = set(ops)
        for op in reversed(ops):
            if _fuse_one_sibling_chain(block, block_ops, op):
                fused += 1
                progress = True
                break
    return fused


def _fuse_one_sibling_chain(block, block_ops, op) -> bool:
    """Merge the sibling chain ending at insert `op`, if it is one. True if the IR changed."""
    if True:
        matched = _sibling_slice_chain(op, block_ops)
        if matched is None:
            return False
        chain, shape = matched
        trees = [_elementwise_tree(ins.operands[0], shape) for ins in chain]
        if any(t is None for t in trees):
            return False
        member_sets = [set(members) for members, _ in trees]
        if any(
            member_sets[i] & member_sets[j]
            for i in range(len(trees))
            for j in range(i + 1, len(trees))
        ):
            return False  # shared work: merging would duplicate it
        roots = [members[-1] for members, _ in trees]
        result_type = roots[0].results[0].type
        if any(r.results[0].type != result_type for r in roots):
            return False
        elem = ir.ShapedType(result_type).element_type
        leaves = []
        for _, tree_leaves in trees:
            leaves.extend(v for v in tree_leaves if v not in leaves)
        last = chain[-1]
        n_out = len(chain)
        # Insert before the LAST insert: a later sibling's leaf `extract_slice`s sit between
        # the inserts, so anchoring on the first would place the merged op above operands it
        # reads. The earlier inserts are moved down past it below.
        with ir.InsertionPoint(last), ir.Location.unknown():
            # `outs` must be slices of the destination, not fresh `tensor.empty`s: elimination
            # only redirects the first empty onto the output, and the other sibling would then
            # be copied in by a host `insert_slice` over device memory, which faults.
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
        # The old chains are dead, but nothing runs DCE before `classify_payload`, which
        # would still plan a kernel for them.
        all_members = [m for members, _ in trees for m in members]
        for member in reversed(all_members):
            if all(len(list(v.uses)) == 0 for v in member.results):
                member.operation.erase()
        # Drop the dead `outs` slices so the earlier inserts can move below the merged op.
        for op_to_drop in list(block.operations):
            if op_to_drop.operation.name == "tensor.extract_slice" and all(
                len(list(v.uses)) == 0 for v in op_to_drop.results
            ):
                op_to_drop.operation.erase()
        for ins in chain[:-1]:
            ins.operation.move_before(last.operation)
        return True


# The name the drivers use.
fuse_rope_halves = fuse_sibling_slice_writers


def classify_payload(mod: ir.Module, func_name: str = "main") -> list[dict]:
    """Walk the payload function and return an ordered plan, one entry per kernel.

    Entry fields:
      kind    -- matmul / batch_matmul / contraction / reduction / elementwise / transpose /
                 attention
      members -- op names owned by this kernel, in program order; the LAST one is tiled
      handles -- (op name, ordinal in IR order) per member, for `generic_schedule`
      mnk     -- (M, N, K) for contractions, for per-shape tile selection
      shape   -- the kernel's result shape
    """
    plan: list[dict] = []
    # value -> index of the plan entry producing it (for the join and merge rules).
    owner: dict = {}
    # Per-op-name ordinal in IR order. `generic_schedule` indexes handles by IR position,
    # not plan order: an absorbed member (attention swallows the K^T transpose) is consumed
    # at its group's position, which can be later than the op's own position.
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
                    # No kernel, but keep the dataflow chain intact across reshapes.
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
                # Grouping rule: an elementwise op joins the preceding kernel when it
                # consumes that kernel's result and has the SAME result shape, i.e. shares
                # its iteration space. Fusing different spaces into one forall leaves a
                # `vector.contract` that never becomes an `xegpu.dpas` (RoPE's head view
                # folded into a projection). A reduction is exempt: its consumer is
                # full-rank by definition and must join it, since the reduced intermediate
                # cannot be a kernel output.
                prev = plan[-1] if plan else None

                # Attention absorbs a whole region: a batch_matmul opens an "attention"
                # group that swallows everything downstream (scale, softmax reductions and
                # elementwise ops, the second contraction), because the flash rewrite needs
                # `QK^T -> softmax -> @V` inside ONE forall. Shapes legitimately change
                # along the chain, so the shape test does not apply. The group closes once
                # it holds both contractions and stops receiving elementwise consumers.
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
                    # A K^T transpose feeding the region gets no kernel of its own:
                    # `replace_with_fused_attention` transposes K itself, so the explicit
                    # transpose is dead after the rewrite. Only a transpose of the last two
                    # dims qualifies; the head-major `[1,0,2]` transpose must keep its
                    # kernel, because the flash op needs the head dim outermost. It is found
                    # by ownership of an operand, not by plan adjacency: in a full block the
                    # head-major transpose sits between K^T and the attention entry.
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
                            # Marked, not popped: popping renumbers `plan` and invalidates
                            # every `owner` index above it.
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
                            # A copy: `produced` grows as view ops chain off this entry.
                            "produced": set(results),
                            # Every value the group's members read.
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

                # JOIN rule: an elementwise op reading results of two or more DIFFERENT
                # contraction kernels (the residual add `h + o`) must not join either.
                # Fusing it would drag both matmul chains into one forall, and the
                # single-DPAS annotation rejects two `vector.contract`s. One contraction
                # source is the normal case (a matmul epilogue, or both RoPE halves reading
                # the same projection) and must stay fusable.
                contraction_srcs = {
                    owner[v]
                    for v in operands
                    if v in owner
                    and plan[owner[v]]["kind"] in _CONTRACTION_KINDS | {"attention"}
                }
                foreign = len(contraction_srcs) >= 2

                # MERGE rule: an elementwise op reading two or more ELEMENTWISE kernels on
                # its own iteration space merges them into one kernel, with itself as the
                # last member. The grouping rule only extends `plan[-1]`, so RoPE's
                # `mulf(lo,cos); mulf(hi,sin); subf` would otherwise leave `mulf(lo,cos)` as
                # its own kernel. Only elementwise sources: contraction sources are the join
                # case, and transpose/reduction/attention sources do not share the space.
                # The target is the LATEST source so tiling order stays topological; the
                # others are marked absorbed, not removed (removing renumbers `plan`). A
                # merge whose member escapes is undone by `_split_multi_output_groups`.
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

                # A join entry stays a single op: it is tiled with `fuse_producers=False`,
                # so an earlier member would never enter the forall and would be left live
                # outside every kernel (LLVM translation then fails on a stray
                # `unrealized_conversion_cast`). The last block's residual add is exactly
                # that case: its only consumer is the final norm's `extf`.
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
                    # A join gets its own kernel and is tiled with `fuse_producers=False`:
                    # fusing would clone a foreign matmul into an elementwise kernel.
                    "join": kind == "elementwise" and foreign,
                }
                if kind == "transpose":
                    # Decides whether attention may absorb this transpose. DenseI64ArrayAttr
                    # iterates as plain ints.
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
    """True if `value` is read by an op outside its kernel group.

    `produced` holds every value the group's members and their chained view ops define, so
    a consumer whose results are all in `produced` is inside the group. An op with no
    results (the function's `insert_slice`/return path) always counts as outside.
    """
    for use in value.uses:
        owner = use.owner
        results = [owner.results[i] for i in range(len(owner.results))]
        if not results or any(r not in produced for r in results):
            return True
    return False


def _absorb_reduction_heads(plan: list[dict]) -> list[dict]:
    """Fold the elementwise head of a reduction chain (`x.float()`, `x*x`) into the
    reduction's group.

    The head cannot join during the walk: nothing precedes it, and the reduction opens a new
    entry rather than extending it. Left alone it is a kernel per norm computing a value that
    `_tile_one_reduction` re-fuses as a producer anyway. Runs after
    `_split_multi_output_groups`, because the second norm's head is only cut out of the
    O-projection group there. All-or-nothing per norm: absorbing only the cast would make it
    escape, and the split pass would cut it back out.
    """
    absorbed: set = set()
    for j, red in enumerate(plan):
        if red["kind"] != "reduction":
            continue
        # Earlier elementwise entries this group reads, on its own iteration space.
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
        # Refuse if a head's result is read from outside the merged group: it would be cloned
        # in and left live outside every kernel.
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
    """Split any group with more than one escaping result.

    Tile-and-fuse CLONES producers into the new loop, so a group is only correct when its
    last member's result is the one value the rest of the payload reads. Otherwise the
    earlier escaping member's original op stays live outside every forall and fails LLVM
    lowering. A Llama block hits this: the second norm's `x*x` joins the O-projection group,
    but the residual `h` is also read by the final add, so both escape. Attention is exempt:
    its intermediates die with the flash rewrite.
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
        # Only the last member escapes: the normal case.
        if len(escaping) < 2:
            out.append(entry)
            continue
        cut = escaping[0] + 1
        tail = dict(entry)
        entry["members"] = entry["members"][:cut]
        entry["handles"] = entry["handles"][:cut]
        entry["member_results"] = entry["member_results"][:cut]
        entry["shape"] = _shape_of(entry["member_results"][-1])
        # The tail is elementwise by construction: only elementwise ops are ever absorbed.
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
    """True for a permutation that swaps only the last two dims (`[0, 2, 1]`), i.e. a K^T.

    The fused-attention rewrite subsumes a K^T but needs batch-major Q/K/V, so the head-major
    `[1, 0, 2]` transpose must keep its own kernel.
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
            # d_head comes from the region's own output width.
            p = dict(FA_PARAMS)
            if entry["shape"]:
                p["n_head"] = entry["shape"][-1]
            out.append(p)
        else:
            p = dict(ew_params)
            shape = entry["shape"]
            if shape and len(shape) >= 2:
                # The row extent must be the dim the tiling will block (`_ew_tile_sizes` vs
                # `_transpose_tile_sizes`); a mismatch makes the layout non-distributable.
                if len(shape) <= 2:
                    rows = shape[-2]
                elif entry["kind"] == "transpose":
                    rows = shape[_transpose_block_axis(shape)]
                else:
                    rows = shape[_block_axis(shape)]
                # Shrink the subgroup tile before clamping the work-group tile: the blocked
                # extent can be smaller than one subgroup (H = 4 against sg_m = 32). Extents
                # are powers of two, so `min` keeps the divisibility the annotation asserts.
                p["sg_m"] = min(p["sg_m"], rows)
                p["load_m"] = min(p["load_m"], p["sg_m"])
                p["sg_n"] = min(p["sg_n"], shape[-1])
                p["load_n"] = min(p["load_n"], p["sg_n"])
                p["wg_m"] = min(p["wg_m"], _floor_to(rows, p["sg_m"]))
                p["wg_n"] = min(p["wg_n"], _floor_to(shape[-1], p["sg_n"]))
            out.append(p)
    return out


def _block_axis(shape) -> int:
    """Dim a rank > 2 kernel blocks: the largest non-innermost extent.

    The innermost dim stays whole so the store is a block store. Blocking a fixed position
    breaks on real payloads: the GQA broadcast `(kv, rep, hs, T)` has leading extents of 2,
    and a subgroup-wide tile of a 2-wide dim fails to vectorize.
    """
    lead = list(shape[:-1])
    return max(range(len(lead)), key=lambda i: lead[i])


def _transpose_block_axis(shape) -> int:
    """Dim a rank > 2 transpose blocks: always the last non-innermost one.

    Peeling the dims before the blocked one to 1 must leave the unit dims LEADING in the
    output slice. A unit dim in the middle, `(tile, 1, innermost)`, makes the upstream
    insert_slice vectorization emit a `transfer_write` with the wrong permutation map for
    the dropped dim, so the write is clipped to one row; that silently zeroed the whole
    attention result once. The read side may keep a middle unit dim: it stays a
    rank-reduced `memref.subview` that `xegpu.create_nd_tdesc` takes directly.
    """
    return len(shape) - 2


def _rank_n_tile_sizes(shape, params, axis=None) -> list[int]:
    """Peel every non-innermost dim to 1 except the blocked one; keep the innermost whole.

    The unit dims let the post-tiling unit-extent folding collapse the op to 2-D, the only
    rank the XeGPU work-group layouts distribute. The whole innermost dim keeps the store
    contiguous.
    """
    if axis is None:
        axis = _block_axis(shape)
    return [params["wg_m"] if i == axis else 1 for i in range(len(shape) - 1)] + [0]


def _ew_tile_sizes(shape, params) -> list[int]:
    """Work-group tile sizes for an elementwise op of any rank."""
    rank = len(shape) if shape else 2
    if rank <= 2:
        return [params["wg_m"], params["wg_n"]]
    return _rank_n_tile_sizes(shape, params)


def _transpose_tile_sizes(shape, params) -> list[int]:
    """Work-group tile sizes for a `linalg.transpose`.

    At rank 2 the innermost dim stays whole: the iteration space is the OUTPUT, and the
    output store must stay contiguous (a column-strided store silently corrupts a transpose).
    """
    rank = len(shape) if shape else 2
    if rank <= 2:
        return [params["wg_m"], 0]
    return _rank_n_tile_sizes(shape, params, axis=_transpose_block_axis(shape))


def _tile_one_reduction(anytype, output, wg_rows, rss):
    """Tile a reduction group into one kernel (mirrors the hand `_tile_one_rmsnorm`).

    `output` is the group's final elementwise generic; fusing its producers pulls the
    reduction in. Mandatory: a reduced, 1-wide `xegpu.tensor_desc` is invalid, so the
    reduced intermediate cannot be a kernel output.
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

    # `stop_at_reductions=True`: tracing from the output would walk through the reduction
    # and pick up `x*x`, whose only consumer is the reduction, and
    # `fuse_into_containing_op` fails on a producer with no use in the loop. With the
    # barrier the set is the normalize chain plus `x.float()`, which the normalize reads.
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

    Follows the hand schedule's `_fuse_attention_in_region`: tile the P@V contraction, pull
    the region in, then hand `replace_with_fused_attention` the Q/K/V, scale and output
    handles. It replaces the P@V contraction with an online-softmax loop; the materialized
    scores, the softmax chain and the K^T transpose are dead afterwards and get DCE'd.

    Producers are pulled in with `fuse_producers=True` instead of walking the op chain by
    hand: torch-mlir's sequence is longer than the hand payload's (an extra `truncf`, a
    separate subtract, an `expand_shape`, a two-result max) and would make hops brittle.
    Topological tiling order makes this safe, since every earlier kernel is already in its
    own forall.
    """
    rank = len(shape) if shape else 3
    # Peel the batch dims to 1 and block the query rows, leaving head_dim whole: the inner
    # op is then single-head attention, the shape the flash rewrite expects.
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

    # Q/K/V come from the contractions' operands, not from positional `extract_slice`
    # producers: in a whole block Q/K/V are other kernels' outputs, more slices appear in
    # the trace, and positions slide. Passing the slices rather than the buffers is how the
    # flash op recovers the query-row offset that causal masking needs.
    q = transform.get_producer_of_operand(anytype, qk_matmul, operand_number=0)
    k = transform.get_producer_of_operand(anytype, qk_matmul, operand_number=1)
    if absorbed_transpose:
        # QK^T's rhs is the absorbed K^T. The flash op wants K itself (`[*batch, n_ctx,
        # d_head]`, transposed internally), so hop over it; the transpose is then dead.
        k = transform.get_producer_of_operand(anytype, k, operand_number=0)
    # P@V: operand 0 is the softmax output, operand 1 is V.
    v = transform.get_producer_of_operand(anytype, pv_matmul, operand_number=1)

    # torch-mlir emits the scale as `arith.mulf %in, %cst` inside a generic. Match the
    # mulfs inside the forall and keep only those whose parent is a `linalg.generic`: a
    # named `batch_matmul` over f16 with an f32 accumulator has an implicit
    # `extf + mulf + addf` body whose mulf would otherwise be matched first. Several clones
    # of the scale multiply exist (tile-and-fuse clones a producer per consumer); any of
    # them traces to the same constant and all die with the softmax chain.
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
    # The constant sits at function scope, outside the forall, where `trace_producers`
    # does not walk; ask what defines operand 1 instead.
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

    Everything outside a forall is host code, and `tensor.insert_slice` is vectorizable: a
    payload that writes its result in slices would get a `vector<1024x32xf16>` transfer pair
    in the host function over device memory, which faults. Left alone, bufferization elides
    those inserts because the foralls already wrote the exact subsets.

    `vectorize_children_and_apply_patterns` needs an isolated-from-above target, so the
    per-op form is used and its follow-up patterns are applied here. `reduction_to_contract`
    and `fold_arith_extension` are what make a matmul a `vector.contract` over f16 with an
    f32 accumulator, the one shape `convert-vector-to-xegpu` turns into `xegpu.dpas`.
    """
    # Match by interface, not op name: reduction tiling adds linalg ops the plan never names,
    # and one left unvectorized crashes the XeGPU lowering. Per forall, because
    # `structured.match` takes a single target.
    #
    # Inside a forall an `insert_slice` is kernel code and must become vector transfers: a
    # tiled transpose folds to two strided subviews with a subset copy between them, which
    # unvectorized becomes a `memref.copy` in the kernel and an unlinked `memrefCopy` call.
    with lh_transform.foreach(match(func, ops={"scf.forall"})) as loop:
        structured.structured_vectorize(
            match(loop, interface=structured.MatchInterfaceEnum.LinalgOp), []
        )
        structured.structured_vectorize(match(loop, ops={"tensor.insert_slice"}), [])
        # Patterns are scoped per forall: func-wide they reach across kernel boundaries into
        # the host code. `fold_tensor_subset_ops_into_vector_transfers` is deliberately
        # absent: a head-major transpose's input slice `(rows, 1, hs)` folded into a
        # `transfer_read` gets a permutation map `convert-vector-to-xegpu` cannot convert,
        # while unfolded it bufferizes to a strided subview that `create_nd_tdesc` takes
        # directly. `_bufferize_keeping_transpose_subviews` is the memref-level half of this.
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
    """One-shot bufferization; fold memref aliases into transfers except in transposes.

    Like `lowering_common.bufferize`, but `fold_memref_alias_ops` runs per kernel and skips
    the transpose ones. Folding a rank-reducing subview that drops a MIDDLE dim (a head-major
    transpose's `(rows, 1, hs)` input slice) yields a `transfer_read` with a projected
    permutation map that `convert-vector-to-xegpu` rejects; left unfolded it is a strided
    2-D subview that `create_nd_tdesc` takes directly. The fold is kept everywhere else: a
    k-loop's subview offset is the induction variable, and an unfolded `create_nd_tdesc`
    gets hoisted by LICM past its definition.
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
    """Every memref value from `value` back to its buffer, closest first, through subview /
    expand_shape / collapse_shape.

    The whole chain is returned because the interesting value is usually in the middle: a
    tiled transpose reads `subview(subview(expand_shape(buffer)))`, and the value its
    permutation describes is the `expand_shape`, not the flat buffer.
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


def _redirect_transpose_writer(block, forall, perm, src, dst):
    """Remove a transpose copy kernel by making the SOURCE's producer write the destination.

    This is the attention output transpose: attention writes `(H, T, hs)`, the copy kernel
    makes it `(T, H, hs)`, and a `memref.collapse_shape` reads that as `(T, C)`. The
    destination cannot become a permuted view (collapsing needs contiguity), so instead the
    producer's stores are retargeted at a permuted view of the destination that has the
    source's shape, and the copy kernel, its buffer and its dealloc die. This is how the hand
    payload's attention kernel stores through a strided view of the `(T, C)` buffer.

    Conditions, any failure leaves the kernel in place:
      * `src` and `dst` are plain `memref.alloc`s; `src` is written by exactly one preceding
        top-level forall and read only by this kernel until it is next written
        (bufferization reuses allocs across layers);
      * `dst` is not touched between the producer and this kernel;
      * every access to `src` inside the producer is a `vector.transfer_write` directly on
        the buffer, so swapping the operand for the view keeps the indices unchanged
        (`view[i0, i1, i2] == dst[permuted]`).
    Returns True if the kernel was removed.
    """
    ops = list(block.operations)
    here = _top_level_index(forall, ops)

    def is_alloc(v):
        return isinstance(v, ir.OpResult) and v.owner.operation.name == "memref.alloc"

    if here is None or not is_alloc(src) or not is_alloc(dst):
        return False
    writer = None
    for j in range(here - 1, -1, -1):
        if _writes_memref(ops[j], src):
            writer = j
            break
    if writer is None or ops[writer].operation.name != "scf.forall":
        return False
    next_writer = len(ops)
    for j in range(here + 1, len(ops)):
        if _writes_memref(ops[j], src):
            next_writer = j
            break
    for use in src.uses:
        owner = use.owner
        if owner.operation.name == "memref.dealloc":
            continue
        at = _top_level_index(owner, ops)
        if at is None:
            return False
        if at in (writer, here):
            continue
        if writer < at < next_writer:
            return False  # src is read or written while live
    for use in dst.uses:
        owner = use.owner
        if owner.operation.name == "memref.dealloc":
            continue
        at = _top_level_index(owner, ops)
        if at is None:
            return False
        if writer <= at < here:
            return False  # dst holds live data the producer would overwrite
    stores = []

    def collect(o):
        for region in o.regions:
            for blk in region.blocks:
                for inner in blk.operations:
                    if any(opnd == src for opnd in inner.operands):
                        if (
                            inner.operation.name != "vector.transfer_write"
                            or inner.operands[1] != src
                        ):
                            return False
                        stores.append(inner)
                    if collect(inner) is False:
                        return False
        return True

    producer = ops[writer]
    if not collect(producer) or not stores:
        return False
    src_type, dst_type = ir.MemRefType(src.type), ir.MemRefType(dst.type)
    rank = len(perm)
    if src_type.rank != rank or dst_type.rank != rank:
        return False
    # dst.shape[i] == src.shape[perm[i]]; the view has src's shape, so position i takes
    # dst's dim inv[i].
    inv = [perm.index(i) for i in range(rank)]
    dst_strides = _row_major_strides(list(dst_type.shape))
    # Hoist the destination alloc above the producer only when it sits below it: an alloc
    # reused across layers may already be above, with earlier uses.
    dst_at = _top_level_index(dst.owner, ops)
    if dst_at is not None and dst_at > writer:
        dst.owner.operation.move_before(producer.operation)
    with ir.InsertionPoint(producer.operation):
        dims = [ir.AffineDimExpr.get(i) for i in range(rank)]
        pmap = ir.AffineMap.get(rank, 0, [dims[inv[i]] for i in range(rank)])
        view_type = ir.MemRefType.get(
            [dst_type.shape[inv[i]] for i in range(rank)],
            dst_type.element_type,
            layout=ir.StridedLayoutAttr.get(
                0, [dst_strides[inv[i]] for i in range(rank)]
            ),
        )
        view = memref.transpose(view_type, dst, pmap)
    for store in stores:
        store.operands[1] = view
    forall.operation.erase()
    if not [u for u in src.uses if u.owner.operation.name != "memref.dealloc"]:
        for use in list(src.uses):
            use.owner.operation.erase()
        src.owner.operation.erase()
    return True


def replace_transpose_kernels_with_views(mod, payload_func_name, plan, kernel_params):
    """Delete every transpose kernel and present its result as a strided `memref` view.

    Both references avoid these kernels by carrying the permutation in STRIDES: the hand
    payload builds `memref.expand_shape` + `memref.transpose`, and Inductor passes a permuted
    strided tensor straight into SDPA. Linalg-on-tensors cannot express that
    (`linalg.transpose` is a copy), but after bufferization the buffers exist and the copy
    can be replaced by a view of the producer's buffer. Runs in Python between
    `generic_schedule(..., stop_after_bufferize=True)` and `generic_schedule_tail`, since the
    transform dialect has no op for it.

    Returns the plan and params with the replaced entries dropped, for the tail.
    """
    func = get_payload_func_op(mod, payload_func_name)
    block = func.regions[0].blocks[0]
    foralls = [o for o in block.operations if o.operation.name == "scf.forall"]
    if len(foralls) != len(plan):
        raise ValueError(f"{len(foralls)} foralls vs {len(plan)} plan entries")
    kind_of = {f.operation: e["kind"] for f, e in zip(foralls, plan)}

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
            continue  # not a plain copy kernel
        dst = _memref_chain(writes[0].operands[1])[-1]
        dst_type = ir.MemRefType(dst.type)
        # The source the permutation describes: same rank, and permuting its shape gives the
        # destination's. Searched from the buffer end so the whole-buffer `expand_shape`
        # wins over the per-tile subviews above it.
        src = None
        for cand in reversed(_memref_chain(reads[0].operands[0])):
            cand_type = ir.MemRefType(cand.type)
            if cand_type.rank == len(perm) and list(dst_type.shape) == [
                cand_type.shape[p] for p in perm
            ]:
                src = cand
                break
        if src is None:
            continue  # nothing in the chain matches the permutation
        src_type = ir.MemRefType(src.type)
        # A `memref.collapse_shape` reader needs contiguous memory, which a permuted view
        # never is. That is the attention output transpose: redirect its producer's stores
        # instead of replacing the destination.
        if any(u.owner.operation.name == "memref.collapse_shape" for u in dst.uses):
            if _redirect_transpose_writer(block, forall, perm, src, dst):
                replaced_idx.add(idx)
            continue
        # Build the view right before the copy kernel: after the source and before every
        # reader of the destination, so it dominates all of them.
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
        # Only readers in this writer's live range: bufferization reuses one alloc for K's
        # and V's head-major results, so a blind replace-all would hand V's readers K's view.
        ops = list(block.operations)
        here = next(i for i, o in enumerate(ops) if o.operation == forall.operation)
        next_writer = len(ops)
        for j in range(here + 1, len(ops)):
            if _writes_memref(ops[j], dst):
                next_writer = j
                break
        # A contraction cannot block-load its operand through permuted strides (e.g. `W.T`).
        if any(
            (at := _top_level_index(u.owner, ops)) is not None
            and here < at < next_writer
            and kind_of.get(ops[at].operation) in _CONTRACTION_KINDS
            for u in dst.uses
        ):
            view.owner.operation.erase()
            continue
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

    # Only the entries actually replaced are dropped; the tail matches one `gpu.module`
    # per remaining entry.
    keep = [i for i in range(len(plan)) if i not in replaced_idx]
    return (
        [plan[i] for i in keep],
        [kernel_params[i] for i in keep],
        len(replaced_idx),
    )


def redirect_staged_destination_copies(mod, payload_func_name: str) -> int:
    """Delete the staging buffer bufferization inserts for a second destination slice.

    Companion to `fuse_sibling_slice_writers`. One-shot bufferization cannot prove that two
    `extract_slice` destinations of one buffer are disjoint subsets, so for the second one it
    allocates a staging buffer and brackets the kernel with host copies:

        %subview = memref.subview %dst[0, half] ...
        %alloc   = memref.alloc()
        memref.copy %subview, %alloc          <- in
        scf.forall { ... vector.transfer_write %v, %alloc[...] }
        memref.copy %alloc, %subview          <- out

    Both copies are host accesses over device memory and fault. This points the kernel's
    writes at `%subview` and drops the alloc and both copies. Sound because the kernel then
    writes exactly the elements it wrote before into the buffer they came from, so unwritten
    elements keep their value, which is all the copy in preserved. Allocs the kernel READS
    are skipped, since a redirected read could observe an earlier write of the same loop.
    Fixing this upstream needs a disjoint-inserts rule in one-shot analysis plus a
    tensor-level rewrite of the forall's shared_outs; judged too heavy for the gain.

    Runs after bufferization and before the tail, like `replace_transpose_kernels_with_views`.
    Returns the number of staging buffers removed.
    """
    func = get_payload_func_op(mod, payload_func_name)
    block = func.regions[0].blocks[0]
    removed = 0
    # Iterate the copies out, not the allocs: bufferization reuses one staging buffer across
    # layers, so an alloc can have several live ranges, each handled on its own.
    for copy_out in list(block.operations):
        if copy_out.operation.name != "memref.copy":
            continue
        buf, dest = copy_out.operands[0], copy_out.operands[1]  # (source, target)
        if (
            not isinstance(buf, ir.OpResult)
            or buf.owner.operation.name != "memref.alloc"
        ):
            continue
        if (
            not isinstance(dest, ir.OpResult)
            or dest.owner.operation.name != "memref.subview"
        ):
            continue
        alloc = buf.owner
        dest_type, buf_type = ir.MemRefType(dest.type), ir.MemRefType(buf.type)
        if list(dest_type.shape) != list(buf_type.shape):
            continue
        if list(ir.DenseI64ArrayAttr(dest.owner.attributes["static_strides"])) != [
            1
        ] * (dest_type.rank):
            continue  # non-unit strides would not map the kernel's indices 1:1

        # This copy out's live range starts after the previous copy out of the same buffer.
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
                continue  # belongs to another live range
            if name == "memref.copy" and use.operand_number == 1:
                copies_in.append(owner)
            elif name == "vector.transfer_write" and use.operand_number == 1:
                writes.append(owner)
            else:
                disqualified = True  # anything else (notably a read) makes this unsafe
        if disqualified or not writes or len(copies_in) > 1:
            continue
        if copies_in and copies_in[0].operands[0] != dest:
            continue  # staged against a different buffer than it is written back to

        # Bufferization emits the subview next to the copy out, below the loop. Hoist it
        # above the first kernel that writes it, if its own operands already dominate there.
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
        # The buffer and its dealloc go only once nothing else uses it.
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

    torch-mlir spells `repeat_interleave(n_rep, 0)` as a broadcast generic that materializes
    K and V at the query-head count: two copy kernels per layer and 4x the KV read traffic.
    The hand payload's K/V indexing map simply omits the `rep` dim, and Inductor hands SDPA
    the 8-head tensor as-is. Tensor-level fixes were measured and rejected: they need
    `linalg-fuse-elementwise-ops`, which rewrites every projection into a generic.

    After bufferization both sides are explicit:

        scf.forall (%kv, %rep, %t) {                              <- the broadcast kernel
          %x = vector.transfer_read %k_view[%kv, %t, 0]
          vector.transfer_write %x, %k_bcast[%kv, %rep, %t, 0]
        }
        ...
        %kv, %rep = affine.delinearize_index %head into (n_kv, n_rep)   <- inside attention
        %k = vector.transfer_read %k_bcast[%kv, %rep, %j, 0]

    A reader of `dst[a, b, c]` is a reader of `src[a, c]`, so each reader in the kernel's
    live range is rewritten to index the source with the broadcast dims dropped, and the
    kernel, its buffer and its dealloc die. K and V are then read straight out of the
    projection buffers through the head-major views `replace_transpose_kernels_with_views`
    made, so the attention kernel's three loads are all strided reads of the same kind.

    Recognised shape: a forall holding exactly one `transfer_read` + one `transfer_write`,
    whose write indices contain the read indices in order (the extra ones are the broadcast
    dims), with the broadcast dims outside the vector-covered trailing dims and equal
    trailing shapes. Windowed to the copy's live range, since bufferization reuses allocs
    across layers.

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
        if (
            not isinstance(dst, ir.OpResult)
            or dst.owner.operation.name != "memref.alloc"
        ):
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
                readers = (
                    None  # something other than a plain read; leave it materialized
                )
                break
            readers.append(owner)
        if not readers:
            continue
        # Every reader must be the same kind of minor-identity read the broadcast kernel did.
        ok = True
        for rd in readers:
            if ir.VectorType(rd.results[0].type).rank != vec_rank:
                ok = False
            pmap = ir.AffineMapAttr(rd.attributes["permutation_map"]).value
            if pmap != ir.AffineMap.get_minor_identity(dst_type.rank, vec_rank):
                ok = False
        if not ok:
            continue
        # The source must still hold the same data when the reader runs. Bufferization can
        # give K's and V's head-major results one alloc, so if anything writes the source's
        # underlying buffer between the copy and a reader, keep this one materialized.
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
        if any(
            _writes_memref(ops[j], src_base) for j in range(here + 1, last_reader + 1)
        ):
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

    Split out of `generic_schedule` so it can also run as a second schedule after a Python
    pass has rewritten the bufferized module. `plan` and `kernel_params` must line up with
    the `scf.forall`s that are actually left.
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
            # Shared with the hand schedule: anchors the store_nd, the three load_nd
            # (Q hoisted, then K and V in the flash loop) and both dpas ops.
            xegpu_fa_annotation(gpu_func, params)
        elif entry["kind"] in _CONTRACTION_KINDS:
            xegpu_wg_annotation_for_mlp_layer(gpu_func, gpu_specs=gpu_specs, **params)
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
            # 2-D for every elementwise kernel: unit-extent folding collapsed the rank > 2
            # ones after tiling.
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

    Used with `generic_schedule(..., stop_after_bufferize=True)` when a memref-level rewrite
    has to run on the bufferized module in Python.
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
    """Decompose `tensor.concat` and eliminate empty tensors, before any tiling.

    Also run standalone (`eliminate_empty_tensors_schedule`) ahead of `fuse_rope_halves`.
    Running it twice is harmless: rounds past the chain depth are no-ops.
    """
    # `tensor.concat` has no tiling interface and bufferizes into host copies over device
    # buffers; as `insert_slice`s it is folded away by the elimination below.
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        tensor_transform.apply_patterns_tensor_decompose_concat()
    lh_transform.cleanup(func)

    # Eliminate empty tensors BEFORE tiling. A payload that assembles its result from slices
    # reads as `tensor.empty -> insert_slice -> ... -> materialize_in_destination(arg0)`;
    # eliminated now, each producer writes straight into its slice of the output and the
    # inserts bufferize away. After tiling the empties are forall inits, the inserts survive
    # into the host function and get vectorized there, which faults on device memory.
    #
    # One round peels one level of the chain; `fold_tensor_empty` between rounds turns the
    # stale `extract_slice(empty)` back into an `empty` for the next round (it is a pattern
    # set, not a canonicalization, so `cleanup` alone would spin). Extra rounds are no-ops.
    for _ in range(_EMPTY_ELIM_ROUNDS):
        bufferization_transform.bufferization_eliminate_empty_tensors(mod)
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            tensor_transform.apply_patterns_tensor_fold_tensor_empty()
        lh_transform.cleanup(func)


def eliminate_empty_tensors_schedule(payload_func_name: str) -> ir.Module:
    """The empty-tensor-elimination preamble as its own schedule.

    `fuse_rope_halves` needs the RoPE destination already rooted at the real output buffer:
    otherwise `fold_tensor_empty` collapses the merged op's `extract_slice(empty)`
    destinations and the second half reaches the output through a host `insert_slice` over
    device memory. Drivers run this, then fuse, then `classify_payload`.
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

        # One handle per op per op name, indexed by IR position (the ordinals from
        # `classify_payload`). Popping in plan order would be off by one after an absorbed
        # member and hand kernels each other's tile params.
        queues: dict[str, list] = {}
        for op_name in {m for e in plan for m in e["members"]}:
            count = sum(1 for e in plan for m in e["members"] if m == op_name)
            queues[op_name] = list(match_and_split(func, ops={op_name}, nhandles=count))

        for entry, params in zip(plan, kernel_params):
            # The group's last member is its output, and that is what gets tiled.
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

        # Generalize tiled transposes: the unit-extent folding below only rewrites generics,
        # and a tiled `linalg.transpose` keeps a middle unit dim in its input slice that is
        # not distributable. After tiling on purpose: classification and tile-size choice
        # need to see a `linalg.transpose`, not an anonymous all-parallel generic.
        if any(e["kind"] == "transpose" for e in plan):
            structured.structured_generalize(
                anytype, match(func, ops={"linalg.transpose"})
            )
            lh_transform.cleanup(func)

        # Collapse rank > 2 kernels to 2-D: XeGPU work-group distribution wants 2-D vectors.
        # Tiling made every middle dim 1, so folding unit extents here rewrites those
        # generics to rank 2. It must not run before tiling (it collapses a reduction's
        # accumulator and breaks vectorization), and is gated on a rank > 2 kernel existing.
        if any(_rank(e["shape"]) > 2 for e in plan):
            with ir.InsertionPoint(transform.apply_patterns(func).patterns):
                structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
            lh_transform.cleanup(func)

        if inspect:
            transform.yield_()
            return schedule

        # The tail is spelled out rather than `vectorize_bufferize_and_outline_gpu_func`:
        # vectorization is scoped, and a reduction needs `promote-buffers-to-stack` after
        # bufferization.
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
