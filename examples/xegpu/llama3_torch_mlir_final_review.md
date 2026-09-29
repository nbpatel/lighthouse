# Llama-3 from torch-mlir on Intel GPU: final review of the fusion work

Branch `llama3_actual_data_torch_mlir`, reviewed 2026-09-30.

This is the end-of-effort summary. It records every fusion applied between the raw torch-mlir
export of a Llama-3 decoder block and the 15-kernel-per-layer schedule that now runs all 16
layers of Llama-3.2-1B on the Intel Data Center GPU Max 1100 with real weights, and it compares
each fusion with what the hand-written lighthouse payload and PyTorch Inductor do for the same
op. The last section pulls out the rules that are not Llama-specific.

Revised 2026-09-30 after a review of the RoPE rewrites (section 4.6): the matcher is now
shape-agnostic, and the post-bufferization copy fix-up is traced to an upstream
one-shot-bufferization gap, with a patch drafted (`one-shot-bufferize-disjoint-inserts.patch`,
unbuilt here). Replacing the fix-up properly would also need a new tensor-level transform op;
that was judged too heavy for the gain, so the fix-up stays, documented.

The working notes behind this file are `llama3_torch_mlir_RESUME.md` (session log),
`llama3_torch_mlir_optimization_plan.md` (worklist, spikes, kernel census),
`llama3_inductor_fusions.md` (the Inductor capture) and `../../llama3_fusions_lighthouse.txt`
(the hand-payload fusion record). All numbers below were measured and are cited from those.

## 1. The question this effort answered

torch-mlir emits a Llama block as one linalg op per math step. Inductor compiles the same
block to 6 generated Triton kernels, 7 external matmuls and 1 external fused attention call.
The hand-written lighthouse payload (`llama3_payload.py` + `llama3_schedule.py`) reaches 15
kernels per layer, but only because a human wrote the payload in an already-fused shape and
told the schedule what every op is.

The question was: **can a schedule alone, with no human-authored payload and no per-op `kinds`
list, take the compiler-generated IR to the same place, and which fusion rules does it need?**

Answer: yes, to parity with the hand payload (15 = 15) and one kernel short of Inductor (14),
with every rule derived from the IR. The rules are listed in section 5.

## 2. Starting point: what torch-mlir hands over

Per decoder block plus final RMSNorm and LM head, the plain export contains
(`llama3_torch_mlir_optimization_plan.md` section 1):

| Op | Count | What it is |
| --- | --- | --- |
| `linalg.generic` | 55 | every pointwise or reduction step separately |
| `linalg.matmul` | 9 | 7 projections + LM head (+1 in the outer ops) |
| `linalg.batch_matmul` | 3 | attention QK^T and PV, materialising the full T x T scores |
| `linalg.transpose` | 6 | head-major transposes around attention |
| `linalg.fill` | 10 | accumulator inits |
| `tensor.expand_shape` / `collapse_shape` | 8 / 3 | head reshapes (metadata in Inductor, dataflow here) |

Lowered naively that is one kernel per op, a score-materialising attention, and host-side
`tensor.concat` copies for RoPE that fault on device memory. Nothing about heads, GQA or RoPE
is a view.

## 3. The ladder: kernel count per block at each step

| Step | Per block | 16 layers | Rule or rewrite that produced it | Where |
| --- | --- | --- | --- | --- |
| 0. one kernel per linalg op | ~70 | -- | none | -- |
| 1. group by dataflow, tile in topological order | ~28 | 484 | `classify_payload` grouping; flash attention region; no global fuse pass | section 4.1, 4.2 |
| 2. RMSNorm accumulates in f32 | 30 | -- | payload correctness fix; costs +2 (the `extf` heads) | section 4.3 |
| 3. MERGE rule | 26 | 420 | elementwise op reading two elementwise kernels merges them | section 4.4 |
| 4. ABSORB rule | 22 | 354 | reduction group owns the elementwise head of its chain | section 4.3 |
| 5. head-major transposes become views | 19 | 306 | `replace_transpose_kernels_with_views` (post-bufferization) | section 4.5 |
| 6. RoPE halves fused | 17 | 274 | `fuse_sibling_slice_writers` (alias `fuse_rope_halves`) + `redirect_staged_destination_copies` | section 4.6 |
| 7. GQA broadcasts folded | **15** | **242** | `fold_gqa_broadcasts` (post-bufferization) | section 4.7 |
| hand payload | 15 | 243 | | |
| Inductor | 14 | -- | | |

Correctness held at every step: block rel error 4e-4 to 7e-4 against the f32 reference, and the
16-layer real-weights run predicts ' Paris' for "The capital of France is" with **bit-identical
logits** on the copy path and after each of steps 5, 6 and 7 (rel 9.01e-4 to the f16 payload in
all four). Bit-identity is the evidence that the three post-bufferization rewrites are pure
layout changes and not approximations.

## 4. Each fusion: what, why, how, and what did not work

### 4.1 Fuse through tiling, not through a fusion pass (step 1, the foundation)

**What.** `classify_payload` walks the payload once and returns an ordered plan, one entry per
kernel. Each entry names its class (matmul, batch_matmul, contraction, reduction, elementwise,
transpose, attention) and its member ops. An elementwise op joins the preceding entry when it
consumes that entry's result and shares its iteration space. Each group is then tiled into one
`scf.forall`, in topological order, through its last member, with `fuse_producers=True`.

**Why not `linalg-fuse-elementwise-ops`.** Measured on the full block: the global pass turned
8 `linalg.matmul` + 2 `linalg.batch_matmul` into 4 matmul + 6 generic contractions. It absorbs
neighbours into contractions and rewrites them as generics, which the DPAS path (matching
`linalg.matmul` / `xegpu.dpas`) cannot take. Tiling-based fusion is layout-aware and preserves
op identity. On the mini-FFN: global pass 5 kernels, no fusion 11, grouped tiling 3.

**Why each part of the tiling recipe is load-bearing** (from the `_ffn_probe.py` comparison of
four variants):
- *tile every op directly*, not through its leaf consumer: leaf-consumer tiling over-fuses and
  drags a second matmul into the same forall, a two-DPAS kernel the annotation rejects;
- *`fuse_producers=True`*: pulls each contraction's `linalg.fill` inside the forall. A fill
  left outside is a host write over device memory and page-faults (the layer_norm
  host-accumulator trap);
- *topological order*: an op's producers are already inside foralls when it is reached, so
  they cannot be re-absorbed, giving exactly one kernel per plan entry.

**Grouping is by full shape, not rank.** A group is one iteration space. Grouping RoPE's
`(T*H, hs)` view with the `(T, C)` projection left a `vector.contract` that never became a
DPAS. Reductions are exempt because they change shape by definition and their fusion is
mandatory (a reduced mean cannot be a kernel output).

**Dataflow is traced through view ops.** `expand_shape`, `extract_slice`, `collapse_shape`
compute nothing but carry the producer/consumer link; ignoring them splits kernels.

**What it gives for free.** Every matmul-adjacent pointwise op lands in the matmul's kernel as
an epilogue: the f16 casts after every projection, SiLU after the gate projection, the residual
add after the O projection. (Not the SwiGLU multiply: it reads two different matmul kernels,
so it is a join and stays its own kernel, see section 6.3.) This is where the V-cast kernel the
hand payload still spends disappears.

### 4.2 Flash attention from the decomposed export (part of step 1)

**What.** A `batch_matmul` opens an attention group that absorbs the scale multiply, the
decomposed softmax (max, sub, exp, sum, div), the second contraction, the output cast and the
K^T `linalg.transpose` feeding it. The group is rewritten with the existing
`transform_ext.replace_with_fused_attention` into one online-softmax kernel, causal flag
included, so the T x T scores are never materialised.

**How it differs from the library attention schedule.** That schedule assumes an attention-only
payload (`split_handle(2)` over all contractions, "only one `arith.max*`"). Here the handles
are scoped to the group, producers are pulled in with `fuse_producers` rather than a hand-written
SSA walk, and the softmax reductions that end up outside the forall are simply dead after the
rewrite.

**Payload spellings that matter.** No `keepdim` on the row max (the `(d0,d1,0)` output map is
not tileable); hand-spelled softmax rather than `torch.softmax` (its f32 upcast dies in XeGPU
layout assignment); no explicit mask tensor (the causal flag does it, and `scores + mask` breaks
the scale tracing). Finding the scale needs care: the f16 batch_matmul has an implicit
`extf/mulf/addf` body, so the scale is the multiply whose parent is a `linalg.generic`.

**Library fixes it needed.** `fused_attention_schedule` plumbs `causal`; the four validation
paths in `replace_with_fused_attention.py` called a nonexistent `emit_silenceable_error` and now
raise properly.

### 4.3 RMSNorm as one kernel: the ABSORB rule (steps 2 and 4)

**What.** Each RMSNorm is one kernel of 10 linalg ops: `extf` to f32, `x*x`, row sum,
mean/eps/rsqrt, scale, f16 cast, gain. Parity with Inductor F0/F3 and with the hand payload.

**Why it was not automatic.** The reduction group naturally owns its output chain, but the
head of the chain (`x*x`, and after the f32 fix `extf`) has no producer in the group, so it
opened its own kernel: 4 extra kernels per block computing values the reduction re-fuses as
producers anyway.

**How.** Three pieces, all needed:
1. `_absorb_reduction_heads` is a **post-pass** after `_split_multi_output_groups`, not a
   walk-time rule. The second norm's `extf` joins the O-projection group during the walk and only
   becomes its own entry when the split pass cuts the group at its first escaping result. A
   walk-time absorb measured 24, not 22. It is also all-or-nothing per norm: absorbing only the
   `extf` leaves `x*x` reading it from another kernel, so it escapes and is cut back out.
2. **Library change:** `transform_ext.trace_producers` gained `stop_at_reductions` (default
   off). Tracing producers from the normalize chain walked back *through* the reduction and
   returned `x*x`, whose only consumer is the reduction and not the output loop, and
   `fuse_into_containing_op` failed with "could not find next producer to fuse into container".
   The recorded folklore that absorbing the head "breaks tiling" was really this producer-set bug.
3. The reduction's own loop keeps the unbounded trace, since `x*x` belongs there.

**f32 accumulation is a correctness requirement, not a choice.** In f16 the 1B model does not
predict the right token. Inductor's F0 also accumulates in f32.

**Dead ends measured.** Tiling the reduction before the output chain hits an upstream
`cast<scf::ForallOp>` assertion (hard crash). `structured.fuse` tile-and-fuse for the output
chain fails with a silenceable failure; `TileUsingForOp` + explicit fuse is what works.

### 4.4 The MERGE rule (step 3)

**What.** An elementwise op that reads **two** elementwise kernels sharing its iteration space
merges them into one group, instead of joining only the immediately preceding entry.

**Why RoPE needed it.** `out[:, :half] = lo*cos - hi*sin` is emitted as `mulf(lo,cos)`,
`mulf(hi,sin)`, `subf`. The second `mulf` does not read the first, so it opened its own entry;
the `subf` joined that one and the first `mulf` stayed a kernel computing a value its neighbour
immediately consumed. 4 kernels per RoPE became 2.

**Guardrails.** Elementwise sources only: two contraction sources are the join case and must not
fuse (a two-DPAS kernel). Merge target is the latest source so plan order stays topological.
Sources are marked absorbed, not removed (removing renumbers the plan). And
`_split_multi_output_groups` undoes any merge whose members escape, so a wrong merge degrades to
the old plan rather than miscompiling. Numerics improved slightly (fewer f16 round trips).

### 4.5 Head-major transposes as strided views (step 5)

**What.** The three Q/K/V transposes `(T,H,hs) -> (H,T,hs)` are no longer kernels or copies.
Their readers read a `memref.transpose` view of the producer's buffer. The strides emitted are
identical to Inductor's capture: `strided<[64, 2048, 1]>` for Q at 32 heads, `[64, 512, 1]` for
the 8-head K/V.

**Why post-bufferization.** Both references carry the permutation in strides: the hand payload
via `memref.expand_shape` + `memref.transpose` (`_heads_view_of`), Inductor via `aten.permute`
metadata. Linalg-on-tensors has no view of a permuted *value*, so the tensor-level plan could
only give the transpose its own kernel class (which was built first, and lowers correctly, but
moves data). After bufferization the buffers exist, and the transform dialect has no "replace
this copy with a view" op, so `replace_transpose_kernels_with_views` runs in Python between
`generic_schedule(stop_after_bufferize=True)` and `generic_schedule_tail`.

**Two gotchas that cost time.**
- The attention **output** transpose `(H,T,hs) -> (T,H,hs)` cannot be a view: its reader
  collapses to `(T, C)` and `collapse_shape` of a permuted view is never contiguous. The pass
  skips any transpose with a `collapse_shape` reader. This is the one remaining kernel over
  Inductor.
- `replace_all_uses_with` is **unsound**. One-shot bufferization reuses one alloc for K's and V's
  results (non-overlapping lifetimes), so replacing every use hands V's readers K's view. The
  replacement is windowed to the live range: uses after this kernel and before the next writer
  of the same buffer.

### 4.6 RoPE in one kernel (step 6)

**What.** Each RoPE's two half-chains become one multi-result `linalg.generic` writing both
column halves of the destination, the exact form the hand payload's `Builder.rope` uses.

**Why it could not be fixed in the payload.** Inductor spells rotate-half as index arithmetic
inside the Triton kernel. Every PyTorch spelling was measured: `torch.cat` and `torch.stack` give
`tensor.concat` (host data movement, no TilingInterface, cannot be tiled through); `torch.flip`
gives a `tensor.extract` gather (does not lower to block loads); slice-assign into
`empty_like` gives two sibling generics writing disjoint slices, which is the best reachable
form but still two kernels. This is the one place where the "fix it in the payload" lesson
did not apply.

**How.** Two rewrites used together (`--fuse-halves`); revised 2026-09-30:
1. `fuse_sibling_slice_writers` (alias `fuse_rope_halves`) runs on tensor IR after empty-tensor
   elimination and before `classify_payload`. It matches ANY chain of `tensor.insert_slice`s
   writing static, unit-stride, pairwise-disjoint, equally shaped slices of one destination
   (any rank, any count) and merges the sibling elementwise cones into one N-result generic,
   so the plan sees one op. Nothing in the matcher is RoPE-specific any more.
2. `redirect_staged_destination_copies` fixes what bufferization then does: it cannot prove the
   two destination slices disjoint, keeps one result in place and stages the other through an
   alloc bracketed by host `memref.copy`s, which fault on device memory. The pass points the
   kernel's write at the destination `memref.subview` itself and drops the alloc and copies.
   Sound because the kernel writes exactly the elements it wrote before into the buffer they came
   from; the copy in becomes a self-copy and the copy out the identity. It skips any staging
   alloc that is also read.
   *Root cause, established 2026-09-30:* the tiled forall carries one shared_out per slice and
   writes them back through an `insert_slice` chain, and `OneShotAnalysis.cpp::
   areNonConflictingSubsets` has no rule for two inserts into disjoint subsets of one
   destination. A proper fix needs a tensor-level op that gives the forall the whole destination
   as one shared_out (a forall counterpart of `sink_extract_slice_into_loop`; prototyped, tested,
   withdrawn as too heavy) plus that analysis rule (`one-shot-bufferize-disjoint-inserts.patch`,
   with lit tests, unbuilt). The memref-level fix-up stays.

**Dead ends measured.** `transform.loop.fuse_sibling` rejects the pair on a structural
dominance check even though the dependence is false. Letting `classify_payload` group all six
RoPE ops reports "tiled OK" and dies in the ExecutionEngine (a group tiles through its last
member, and the lower half is not a producer of the upper). A rank-3 `(rows, 2, half)`
single-result form moves the problem into XeGPU (a 3-D `tensor_desc` is not distributable).

### 4.7 Zero-copy GQA (step 7)

**What.** The two K/V broadcast kernels (`repeat_interleave` materialised as a rank-4 broadcast
generic) are deleted. The attention kernel reads the head-major K/V views directly with the
`rep` index dropped. This is the hand payload's `_grouped_heads_view_of` (a K/V indexing map
that omits `rep`) reached at the memref level, with no library or payload change.

**How.** After bufferization the two sides were already aligned: the broadcast kernel wrote
`k_bcast[kv, rep, t]` from `k_view[kv, t]` (rep dropped), and the attention forall already
delinearised `head` into `(kv, rep)` and read `k_bcast[kv, rep, j]`. `fold_gqa_broadcasts`
matches an elementwise forall of exactly one `transfer_read` + one `transfer_write` whose write
indices are a superset of the read indices in order, rewrites every reader to index the source
view, and deletes the copy kernel, alloc and dealloc. It runs last because it reads K/V through
the transpose views.

**Why the tensor-level routes were rejected.** `linalg-fuse-elementwise-ops` does produce the
hand form (`populateFoldReshapeOpsByExpansionPatterns` re-expands the batch dim), but it also
rewrites the projections into rank-4 generic contractions, and the pass cannot be scoped to a
forall. `einsum` merges `rep` into the query-row dim, which breaks causal masking.

**Soundness hole caught by testing the flag alone.** `--fold-gqa` without `--transpose-views`
first tripped "expected 3 payloads but contains 2". Without views, K's and V's transposes share
one alloc; the broadcast copies were what decoupled K from that reuse, so after folding,
attention read K's source after V's transpose had overwritten it. K == V, the two loads CSE'd,
and the result would have been silently wrong. Fix: window **both** the destination and the
source live ranges, walking through `memref.transpose` to the underlying buffer. Now the flag
alone correctly folds V only.

## 5. Three-way comparison: Inductor, hand payload, torch-mlir schedule

Per decoder layer. "Same fusion?" asks whether the same ops end up in the same launch, not
whether the mechanism is the same.

| Inductor kernel | Inductor (Triton XPU) | Hand payload + `llama3_schedule.py` | torch-mlir + `llama3_torch_schedule.py` | Same fusion? |
| --- | --- | --- | --- | --- |
| F0 RMSNorm + casts | 1 Triton reduction kernel; f32 accumulate; f16 store | 1: `rmsnorm(materialize=False)` + cast fused via `fuse_into_containing_op`; f32 partials through SLM | 1: ABSORB rule + `trace_producers(stop_at_reductions)`; pure vector ops in registers, no SLM | Yes (ours keeps the reduction in registers) |
| F1 Q RoPE | 1; rotate-half as Triton index arithmetic | 1: `Builder.rope` multi-result generic writing two strided views; cast fused | 1: `fuse_sibling_slice_writers` rewrites to the same multi-result form; `redirect_staged_destination_copies` fixes bufferization | Yes (same form as hand) |
| F2 K RoPE | 1 | 1 | 1 | Yes |
| F3 residual add + RMSNorm 2 | 1; computes h = x+p inside, never stores h | 1: same recompute strategy (h is a tensor SSA value, recomputed in reduction and normalize loops) | 1 for the norm; the add is fused as the **O-projection epilogue** and h is materialised | Same launch count, different placement |
| F4 SwiGLU | 1: silu + mul + f16 | 1: silu, mul, cast as three generics in one forall via `structured.fuse` | 1: the product `silu(gate) * up` as its own kernel (a join of two matmul kernels); SiLU itself is the **gate-projection epilogue** | Same launch count, different placement (ours uses a matmul epilogue for SiLU) |
| F5 both residual adds | 1: recompute x+p, add d | 1: recompute (x+p)+d | 1: h + d (h already materialised by F3's placement) | Yes |
| SDPA | 1 external fused attention; causal; GQA by strides | 1: `replace_with_fused_attention` on a rank-5 GQA view whose K/V map omits `rep` | 1: same library op via the attention group; causal via the plumbed flag; GQA via `fold_gqa_broadcasts` | Yes |
| Head-major Q/K/V transposes | 0: `permute` is metadata | 0: `memref.expand_shape` + `memref.transpose` views | 0: `replace_transpose_kernels_with_views`, identical strides to Inductor | Yes |
| Attention output transpose | 0: attention writes the layout SDPA is told to | 0: attention stores through a strided view of the `(T,C)` buffer | **1**: cannot be a view (`collapse_shape` reader); the one remaining gap | No |
| GQA K/V broadcast | 0 | 0 | 0 (was 2) | Yes |
| V-projection f16 cast | 0: `mm` returns f16 | **1**: separate `cast_f16_buf` kernel | 0: fused as V-matmul epilogue | Ours matches Inductor; hand does not |
| 7 projections | 7 external `mm` | 7 `linalg.matmul` | 7 `linalg.matmul` | Yes, all three keep them separate |
| **Total** | **14** | **15** | **15** | |

Three observations from the table:

1. **All three converge on the same six pointwise/reduction fusion boundaries** (F0 to F5) plus
   one fused attention, despite three completely different mechanisms (Triton codegen, a
   human-authored payload, a derived plan). That is strong evidence the boundaries are
   properties of the graph, not of the compiler.
2. **Where the torch-mlir path differs, it is by using matmul epilogues.** Inductor's external
   `mm` cannot take an epilogue, so casts and SiLU and the residual add have to live in Triton
   kernels; the hand payload materialises matmul outputs to buffers, which bounds fusion the same
   way. Tiling-based fusion has no such boundary, which is why the V cast costs us nothing and
   why F3/F4 are placed differently at the same launch count.
3. **Neither reference ever moves data for a layout change.** Both carry head permutations and
   GQA in strides. The torch-mlir export has to be brought to that state after bufferization,
   which is what the three post-bufferization rewrites do.

## 6. Rules that are not Llama-specific

These came out of the work above and should carry to any model exported through torch-mlir
(nanoGPT, Mistral, Qwen, Gemma, ViTs). Items marked (L) still need a pattern matcher whose
*shape* is model-family-specific even though the rule is general.

**Grouping and tiling (tensor level, in `classify_payload`)**

1. **Fuse through tiling, never through `linalg-fuse-elementwise-ops`.** The global pass
   rewrites named contractions into generics and kills the DPAS path. Grouping + tiling with
   `fuse_producers` reaches the same fusions and keeps op identity.
2. **Group by dataflow and identical iteration space (full shape, not rank), tracing through
   view ops.** Reductions are exempt and their fusion is mandatory.
3. **Tile every group directly, in topological order, through its last member, with
   `fuse_producers=True`.** Each part prevents a specific failure: over-fusion into two-DPAS
   kernels, orphaned fills that page-fault, re-absorbed producers.
4. **Matmul-adjacent pointwise ops are epilogues.** Casts, activations and residual adds fall
   into the producing matmul's kernel with no special handling. This is a strict advantage over
   both references.
5. **MERGE:** an elementwise op reading two elementwise kernels of its own iteration space
   merges them (elementwise sources only; latest source is the target). Any `a*b - c*d` pattern
   needs it, not just RoPE.
6. **ABSORB:** a reduction group owns the elementwise head of its chain, as a post-pass after
   the escape split, all-or-nothing. Applies to LayerNorm (`x - mean`, `x*x`) exactly as to
   RMSNorm. Requires `trace_producers(stop_at_reductions=True)` for the output chain.
7. **A contraction reduces a dim shared by two inputs; a reduction reduces one input.** A
   reduction iterator alone does not tell them apart, and torch-mlir emits projections with
   folded reshapes as generic contractions.
8. **Safety net:** `_split_multi_output_groups` cuts any group whose intermediate escapes, so a
   wrong grouping degrades to more kernels rather than to a miscompile.
9. **Attention is a region, not an op chain.** A contraction pair with a reduction between them
   opens a group that skips the shape test, absorbs the K^T transpose, and is handed to
   `replace_with_fused_attention`. Dead softmax reductions outside the forall are fine.

**Layout (memref level, between bufferization and the tail)**

10. **Layout changes are views, not kernels, and they can only be made after bufferization.**
    The transform dialect has no op for it, so this stage is Python over the bufferized IR.
    Skip any transpose whose reader needs contiguous memory (`collapse_shape`).
11. **Window live ranges on BOTH source and destination.** One-shot bufferization reuses
    allocs across non-overlapping lifetimes, including across layers. Every post-bufferization
    rewrite that hit only the destination range was wrong at 2 layers or under a different flag
    combination. Buffer reuse is the default, not the exception.
12. **Sibling generics writing disjoint slices of one destination should be one multi-result
    generic.** No longer (L): the matcher is shape-agnostic. Bufferization will then stage one
    result through host copies (an upstream analysis gap, see 4.6); redirect the write at the
    destination subview. Sound whenever the staging alloc is write-only.
13. **A broadcast copy whose readers already index the broadcast dims is a fold (L).** Rewrite
    readers to the source with those dims dropped. Any `repeat_interleave` / `expand` feeding a
    kernel that iterates the repeated dim qualifies.

**Payload discipline (things that must be true of the exported IR)**

14. f16 inputs so DPAS gets `matmul ins(f16,f16) outs(f32)`; weights as forward arguments, not
    parameters (baked constants break operand prefetch); DPS via `convert_function_results`.
15. No `keepdim` reductions (restore rank at the use site); hand-spelled softmax; no explicit
    mask tensor; 2-D views for anything with a head dim (rank-3 does not distribute); slice-assign
    into `empty_like` instead of `torch.cat` / `stack` / `flip`.
16. "Fix it in the payload" worked four times (rank-3 RoPE, keepdim, softmax upcast, concat)
    and failed once (RoPE 2 -> 1). When no PyTorch spelling can produce the form, the rewrite
    belongs in the schedule.

**Kernel/host boundary and verification**

17. **Inside a forall is kernel code and everything vectorizable gets vectorized; outside is
    host code and nothing does.** A `linalg` op, `memref.copy` or `vector.transfer` left at
    function scope over device memory is a fault, not a slow path. Grep for the orphan; do not
    count foralls.
18. **"Tiled OK" is not a correctness signal**, and "Failure while creating the
    ExecutionEngine" carries no information: the real error is printed above it.
19. **Test each flag alone and in combination**, at toy and real geometry, at 1, 2 and 16
    layers. The GQA soundness hole and both live-range bugs were found this way and by nothing
    else. `_sweep.sh` (39 cases) is the artefact.
20. **Bit-identical outputs across a layout rewrite are the proof it is a pure view.** Any
    change in the logits of a supposedly-inert rewrite is a bug, not noise.

## 7. What is Llama-specific

- The matcher shape in `fold_gqa_broadcasts` (one read, one write, superset indices). The rule
  is general (section 6.13); the matcher would need widening for a broadcast over a non-head
  dim. (The RoPE matcher was generalised on 2026-09-30: `_sibling_slice_chain` accepts any
  count of disjoint static slices at any rank, so an interleaved RoPE variant only needs a
  spelling that yields disjoint slices.)
- The attention parameter set `FA_PARAMS` (128 threads, head-dim 64 tiles).
- Nothing in the grouping rules, the ABSORB/MERGE rules, the transpose-views pass or the
  redirect pass mentions Llama, heads, or RoPE.

## 8. Remaining gap and follow-ups

- **Attention output transpose (+1 vs Inductor).** The hand payload avoids it by storing
  attention output through a strided `(H,T,hs)` view of the `(T,C)` projection input. The
  analogous post-bufferization move is to redirect the attention kernel's `transfer_write` at a
  `memref.transpose` view of the O-projection's input buffer, the same trick as
  `redirect_staged_destination_copies` with a permuted view instead of a subview. Untested;
  would land at 14.
- **Joins as epilogues of the last contraction (untried, worth 1 to 2 kernels per layer).**
  `gate * up` and `h + o` each read two matmul kernels and are given their own kernel by the
  join rule. But the earlier of the two producers is already inside its forall when the later
  one is tiled, so the join could be the later matmul's epilogue reading the earlier result
  from memory, the way the residual add reads `x`. The mini-FFN spike (plan doc section 3c)
  did this and measured 3 kernels for the FFN; the coarse join rule, added for `h + o`,
  removed it. A finer rule ("fuse into the last contraction if all others are already tiled")
  would land at 13 or 14 per layer.
- **Performance is not benchmarked.** Everything here is launch count and correctness. The
  kernel-count parity says nothing yet about time per layer vs the hand path or vs oneDNN's
  `mm`, and the M-iii cost-model item in the plan is open.
- The three post-bufferization rewrites are behind flags (`--transpose-views`, `--fuse-halves`,
  `--fold-gqa`) on the block and model drivers and are not the default path.
- **Optional upstream follow-up:** `one-shot-bufferize-disjoint-inserts.patch` (drafted, unbuilt)
  adds a disjoint-inserts rule to `OneShotAnalysis.cpp::areNonConflictingSubsets`. On its own it
  does not retire `redirect_staged_destination_copies`; that also needs a tensor-level op that
  gives the tiled forall one shared_out (section 4.6). Deliberately not pursued.
- Library changes made along the way, all default-off or pure fixes: `trace_producers`
  `stop_at_reductions`; `fused_attention_schedule` plumbs `causal`; the
  `replace_with_fused_attention` validation paths raise instead of crashing.
- Agreed and deferred: consolidate the drivers onto `kernel_bench`. The block diagram
  `post-fusion-torch-mlir.jpg` (repo root) was regenerated at 15 kernels on 2026-09-30 from the
  measured plan; `_gen_post_fusion_graph.py` is its source.

## 9. Measured against the "smarter lowering schedule" plan

A parallel effort in the team frames the goal as a five-stage generic schedule: detect blocks
and anchor ops in the input IR, assign tile sizes automatically, propagate them, apply fusion,
then vectorize / bufferize / outline, annotate, and run the common tail. `generic_schedule` in
`llama3_torch_schedule.py` is an instance of exactly that pipeline, built bottom-up against a
real model. This is what each stage got, and what it did not.

| Stage in the plan | Achieved here | How | Not achieved / caveat |
| --- | --- | --- | --- |
| 1. Initial IR cleanup | Yes | `_emit_empty_tensor_elimination`: `tensor.concat` decomposition, iterated empty-tensor elimination with `fold_tensor_empty` between rounds, cleanup. `fuse_rope_halves` runs here too, as a tensor-level pre-tiling rewrite | The one cleanup deliberately **not** run is `linalg-fuse-elementwise-ops`, which destroys named contractions (section 4.1). Cleanup must preserve op identity |
| 2a. Detect blocks / anchor ops | Yes | `classify_payload` + `_classify_generic`: anchors are named contractions and generic contractions (a dim shared by two inputs), reductions (one input), transposes, elementwise. The **softmax block** is detected as a contraction pair with a reduction chain between them and becomes one "attention" entry that also absorbs the K^T transpose and the output cast | Detection is per op class, not per named pattern. A LayerNorm or GELU needs no new code; a new *block* (e.g. a fused MoE router) would need a region rule like the attention one |
| 2b. Assign tile sizes automatically | Yes for GEMMs, partly for the rest | `params_for_plan`: every contraction queries `XeGPUParameterSelector` (parameter database, cost-model fallback) with its own (M,N,K), cached per distinct shape. Elementwise / transpose tiles are clamped to the result extent and the blocked axis; attention takes d_head from the IR | Elementwise, reduction and attention start from constant parameter sets (`EW_PARAMS`, `RED_PARAMS`, `FA_PARAMS`) that are clamped, not cost-modelled. No GEMM tuning beyond the selector (M-iii open) |
| 2c. Propagate tile sizes | No | Each kernel is tiled and annotated independently from its own params | Nothing flows between kernels. It was not needed for launch-count parity, and it is unclear it is needed at all: layout propagation *inside* a kernel (anchor the stores, derive the loads) did the work |
| 2d. Apply fusion, incl. online softmax | Yes | Grouped tiling with `fuse_producers` in topological order; MERGE and ABSORB rules; `_split_multi_output_groups` as the safety net; online softmax via the existing `replace_with_fused_attention` scoped to the attention group | Two fusions were **impossible at the tensor level** and needed a post-bufferization stage the plan does not have (see below); a third, the RoPE write-back, would be expressible at the tensor level given a new transform op and an upstream bufferization rule, neither pursued |
| 3. Vectorize, bufferize, outline | Yes | `_vectorize_kernels_only` (foralls only, never host code), one-shot bufferize, `promote-buffers-to-stack` for reductions, `convert_to_gpu_launch`, `outline_gpu_function` with the thread count derived as (wg_m/sg_m) x (wg_n/sg_n) x subgroup size from each kernel's params | The tail is spelled out rather than taken from `vectorize_bufferize_and_outline_gpu_func`, because a reduction needs the stack promotion inserted after bufferization |
| 4. XeGPU annotations from WG/SG sizes | Yes | Per kernel, keyed by class: `xegpu_wg_annotation_for_mlp_layer` (contractions), `xegpu_wg_annotation_for_elemwise_layer`, `xegpu_fa_annotation` (attention), and for reductions `sg_layout = [wg_m // sg_m, 1]` anchored on the stores with layout propagation deriving the loads. `convert-vector-to-xegpu` runs per kernel so SLM promotion stays selective | Annotations are reused from the library and the hand schedule, not re-derived. The single-DPAS matmul annotation is what forces "one contraction per kernel" (section 6.3) |
| 5. Common tail | Yes | `xegpu_to_binary()` = `gpu-lower-to-xevm-pipeline`, identical to the hand path | -- |

**What the plan is missing, learned here.** Two of the seven fusions that reach parity cannot
be expressed at stage 2 at all, because linalg-on-tensors has no view of a permuted or
broadcast *value*: the head-major transposes as strided views and the zero-copy GQA (sections
4.5, 4.7). They live in a **stage 2.5, "layout rewrites after bufferization, before
outlining"**, run in Python because the transform dialect has no ops for them, and they need
live-range windowing on both source and destination because bufferization reuses allocs. Any
generic schedule that wants to match Inductor's launch count on an attention model needs this
stage. The fused RoPE's write-back lives there too (4.6); on review it is really a missing
bufferization rule (`one-shot-bufferize-disjoint-inserts.patch`) plus a missing tensor-level
transform op, but moving it was judged not worth the weight. Everything else in the plan is
validated by this effort on a 16-layer model with real weights.
