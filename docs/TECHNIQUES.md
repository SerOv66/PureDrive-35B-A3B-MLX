# Adopted optimizations

Every optimization in the final certified configuration (`configs/q6_omlx_cand5.json`), what it does, why it is exact, what it gained, and the
environment flag that switches it on. All of them are **lossless with respect to the target model**: the target is never changed (top-k 8 of 256
experts, shared expert, all 40 layers, dense causal attention over all keys, bf16 KV cache, full vocabulary and `lm_head` in the verifier).
Gains are those measured at the time of adoption (source experiment number in brackets; "EXP-nnn" are the project's internal experiment/audit numbers); they do not add up exactly because they
were measured on different configurations and contexts. The per-step record progression is in `docs/RESULTS.md`.

## How the decode path works

```
          ┌───────────── one cycle (≈ 21 ms at 2K … 36 ms at 128K) ─────────────┐
 MTP head │ draft token 1 → draft token 2        (proposer only, cheap, 32K-token vocabulary)
 target   │ verify 3 rows in ONE pass (last accepted token + 2 drafts): all 40 layers, full vocab
 accept   │ exact acceptance rule → 1 … 3 tokens emitted per cycle (≈ 2.3 on average)
```

* The model has 40 layers: 30 Gated-DeltaNet (linear attention, constant-size state) and 10 full-attention layers (GQA 16 query heads : 2 KV heads,
  head dim 256). Only the 10 full-attention layers touch a KV cache that grows with context; this is why decode cost grows with context.
* The checkpoint carries a native multi-token-prediction (MTP) head. It is used as the proposer of speculative decoding (oMLX "Lightning MTP").
* Per cycle the target is evaluated once on 3 rows. Its weights are read once for all 3 rows; this is what makes speculation pay on a bandwidth-bound decode.

## Summary table

| # | Technique | Env flag / setting | Why it is exact | Measured gain | Source |
|---|---|---|---|---|---|
| 1 | Runtime: oMLX instead of mlx-vlm | (server choice) | same weights, same math | +20% at 2K (72.95 vs 60.9), +17% at 16K (63.74 vs 54.3), single-run smoke | EXP-002/004 |
| 2 | Lightning MTP with **row-exact verify** | `MTP=true`, `OMLX_MTP_ROW_EXACT_FORCE=1` | every verify row computes bit-identically to a one-row decode, so greedy output equals non-speculative output | stock MTP was not lossless (diverged at token 4-5); row-exact fixes it at a small cost | EXP-005/006 |
| 3 | Custom Metal GQA attention kernel (reads K/V once for all 8 Q heads of a KV group and all verify rows) | `C35_GQA_ATTN=1`, `C35_GQA_VERSION=31` | fixed split points and fixed-order merges make each row bit-identical to a one-row decode of the same kernel | v2: +17...+32% from 64K; v16: -19...-21% attention time on 3-row verify; v31 + position-based split: cycle -0.4 ms (2K) ... -2.3 ms (128K) | EXP-013/054/088/091 |
| 4 | Block splitting of long verify windows (> 8 rows) | built into the patch (`MAX_ROWS=64`) | each block of ≤ 8 rows sees exactly the keys its last row may see | correctness fix: coding identity 54/57 -> 57/57 | EXP-033 |
| 5 | Pass-2 merge with batched loads (P2C) | `C35_GQA_P2=3` | same fp32 operations in the same order (bit-identical to the previous merge) | -1.8 ms/cycle at 128K, -0.43 ms at 32K | EXP-065/072 |
| 6 | Position-based split size (`BRULE`) | `C35_GQA_BRULE=1`, then `=2` ("exact wave") | split size B is a pure function of the row's position, never of content or call length | 128K 57.8 -> 61.0 tok/s (+5.6%) for BRULE=1; BRULE=2 cycle -0.1...-0.3 ms at equal tokens/cycle | EXP-090/091/110 |
| 7 | Verify pipelining | `C35_VERIFY_FLUSH=2` | same operations, same numerics; only the scheduling of `async_eval` changes | +5...+13% across contexts (GPU busy 85% -> 97%) | EXP-041/042 |
| 8 | Fixed MTP depth 2 | `"mtp_fixed_depth": 2` | proposer setting only | beats adaptive depth and depth 3 at every context | EXP-026/029/089 |
| 9 | Shared rows in row-exact projections | `C35_SHARED_ROWS_MIN_TILES=0` | each weight block is decoded once for up to 4 rows; per-row arithmetic unchanged | cycle 36.5 -> 34.8 ms at depth 3; 0/219 shapes differ bitwise | EXP-027 |
| 10 | Narrow projection tiles | `C35_RPS=2` | per-column arithmetic is independent of the tile width | -0.5 ms/cycle (~2.4% at 2K) | EXP-064/072 |
| 11 | Unrolled 6-bit projection tile + GDN recurrence with 8 value rows | `C35_RPS6U=4`, `C35_GDN_REC_ROWS=8` | transcribed instruction-for-instruction (same lane layout, same reduction order) | +0.9...+1.3% in all 6 greedy/default cells | EXP-081/085 |
| 12 | Fused residual-add + RMSNorm | `C35_FUSE_GLUE=1` | transcription of the MLX `rms_single_row` kernel with the add inside the load; 180/180 then 1152/1152 bitwise | +0.7...+0.9% (-80 launches/cycle) | EXP-077/078 |
| 13 | Fused q/k RMSNorm + M-RoPE; fused attention output | `C35_QK_PREP=1`, `C35_QK_PREP_MAIN=1`, `C35_ATTN_OUT=1` | bit-identical to the stock ops (630/630 and 60/60 cases) | together -0.1...-0.3 ms/cycle (≈0.3...1%) | EXP-093/110 |
| 14 | Draft vocabulary subset (32,768 tokens) | `C35_DRAFT_VOCAB=<file>` | proposer-only; the target verifies with the full `lm_head` and vocabulary | ≈ +3% | EXP-022 |
| 15 | Draft temperature 0.3 in sampling mode | `C35_DRAFT_TEMP=0.3` | the acceptance rule uses the draft's own distribution q, exact for any q | tokens/cycle +1.2% in default sampling; greedy unchanged | EXP-059/072 |
| 16 | Block verification for sampling | `C35_BLOCK_VERIFY=1` | Sun et al. (ICLR 2025), Algorithm 2: exact in distribution (χ² tests) | tokens/cycle 2.31 -> 2.34 (+1.3%) in default sampling | EXP-077/079 |
| 17 | Pre-queued draft | `C35_MTP_PREDRAFT=1` | the next cycle's draft chain is built before the verify sync; accepted-prefix rows are exact, later rows speculative and discarded; proposer only | +2.6...+4.0% (cycle -0.47...-0.86 ms) | EXP-102/103/105 |
| 18 | Compiled draft sampler; fast GDN cache commit | `C35_DRAFT_COMPILE=1`, `C35_FAST_COMMIT=1` | same ops inside `mx.compile` (300/300 identical outputs); host bookkeeping only | +0.3...+0.8% and +0.04...+0.3% | EXP-099/101 |
| 19 | Proposer attention window (4096) on the MTP head | `C35_MTP_WINDOW=4096` | affects only the proposer; the target keeps full attention | inactive at the benchmarked contexts (the window is never reached there) | EXP-056 |
| 20 | Prefix cache on SSD | `CACHE=1` | restores real KV blocks | needed for long contexts (128K prefill ≈ 287 s); TTFT 3.1 s at 16K | EXP-011 |

## 1. Kernel design (techniques 3-6)

The stock MLX SDPA reaches only ~125-157 GB/s on this shape and its cost grows steeply with the number of query rows (128K context, per layer: 1.74 ms for 1 row, 4.86 ms for 4, 21.5 ms for 8; EXP-010). The custom kernel (`kernels/gqa_attn.py`,
Metal, ≈ 4,800 lines including variants) is built around the speculative verify: 3 query rows × 8 query heads per KV head.

* **One K/V load serves all rows and all 8 query heads of a KV group** (GQA 16:2). Score tiles are computed with `simdgroup_matrix` 8×8 bf16→fp32 MMA.
* **Split-K ("flash-decoding" style)**: the keys are cut into splits of B keys; pass 1 computes per-split partial `(max, sum, output)` with an online softmax
  in key order; pass 2 merges the splits in a fixed ascending order in fp32.
* **Row-exactness by construction.** For MTP-on to equal MTP-off, a row at position p must produce the same bits whether it is computed alone (decode)
  or together with other rows (verify). Therefore split boundaries are *absolute* key positions, fixed-order sums, and B depends only on p:
  v2/v16 use B=256; v31 uses B = 128/256/512/1024 below 8192/49152/98304/above (BRULE=1) and, for BRULE=2, `B = 16·⌈p/(640·k)⌉` with k = 1/2/3 so that
  `40·k` splits fill `k` complete GPU waves (≈ 80 threadgroups per wave on a 20-core GPU). B is read from a runtime buffer rather than compiled in (a
  compile-time constant would trigger one kernel compilation per B and stall decode).
* **v31 numerics are new** (four partial D-quarter sums combined as ((p0+p1)+p2)+p3), so output bits differ from v16 and from stock SDPA. Losslessness is therefore defined
  as *MTP-on == MTP-off with the same numerics*, and accuracy against an fp32 reference is reported separately (`docs/RESULTS.md`, KL table).
* The pass-1 time per tile at production occupancy sits at ≈ 90% of the measured realistic MMA throughput of the GPU (see `docs/ENGINEERING_NOTES.md`).

## 2. Exact speculative acceptance (techniques 2, 8, 14-18)

* **Greedy**: the emitted token is the target's argmax at every position; the draft only decides how many positions are checked in one pass. With row-exact
  verify the target's logits are bit-identical to one-row decoding, so the output is identical to non-speculative greedy decoding. This is tested, not assumed (see `docs/METHODOLOGY.md`).
* **Sampling (temperature 0.6, top-p 0.95, top-k 20)**: standard speculative-sampling acceptance (Leviathan et al. 2023; Chen et al. 2023): accept with probability
  `min(1, p/q)`, on rejection sample from the residual `max(p−q, 0)` normalised, bonus token from `p`. `q` is the distribution the draft was actually sampled from
  (after the vocabulary subset and the 0.3 temperature), so exactness holds for any `q`. Block verification (Sun et al.) replaces the per-token test by a
  joint test over the draft block and is exact in distribution as well; it raised accepted tokens/cycle by 1.3%.
* **Why the proposer tweaks are allowed.** The 32K draft vocabulary, the draft temperature, the 4096-token window on the MTP head and the pre-queued draft change *which tokens are proposed*,
  never the probabilities used to accept or reject them. A token outside the draft vocabulary has q = 0 and can only be produced by the residual/bonus step.

## 3. Host-side scheduling (techniques 7, 17, 18)

The cycle is GPU-bound with small bubbles where the host builds the next graph. Traces (EXP-024/041/045/102) showed 83-87% GPU-busy at 2K before pipelining and 96.8-96.9%
after. Delay-injection A/B tests (EXP-094/096/097) located the remaining cost: a delay at the start of the draft costs 1:1 (≈ 0.5 ms ≈ 2%), while a delay in the verify graph build costs nothing
(the host has ~19 ms spare). Hence the verify graph is flushed every 2 layers, the next draft chain is queued before the verify result is read, and per-cycle host bookkeeping was trimmed.
