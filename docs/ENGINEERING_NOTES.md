# Engineering notes

What worked, why it works, and practical tips for anyone building exact (lossless) speculative decoding on Apple silicon. Measurements are in [RESULTS.md](RESULTS.md); each technique with its flag and exactness argument is in [TECHNIQUES.md](TECHNIQUES.md). "EXP-nnn" are the project's internal experiment/audit numbers.

## 1. What made the biggest difference

| Insight | Effect |
|---|---|
| **Row-exact verification.** Every row of a multi-row verify pass must be bit-identical to a one-row decode at the same position. The stock multi-row kernels were not (greedy output diverged at token 4-5 with MTP on); forcing row-exact kernels makes MTP-on output identical to MTP-off. | Makes speculation lossless; verified end-to-end on 57/57 coding problems (EXP-033, EXP-110) |
| **One K/V read for all rows and heads.** The verify pass has 3 query rows x 8 query heads per KV head. A custom Metal split-K attention kernel loads each K/V tile once and serves all 24 (row, head) pairs with 8x8 MMA tiles. | 2-3x faster than MLX SDPA on 3-8 row verifies; the main reason decode stays fast as context grows |
| **Split points that depend only on position.** Splits are absolute key positions, merged in a fixed ascending order, with the split size B a pure function of the row's position. | Row-exactness by construction, including across split-size changes |
| **Wave-aligned splits.** Choose B so that the threadgroups (`40·k` splits) fill whole GPU waves of a 20-core GPU. | +21...+61 µs/layer over a fixed split rule (EXP-110); hardware-driven, not tuned on the benchmark |
| **Verify pipelining.** Flush the lazy graph with `async_eval` every 2 layers instead of building the whole verify graph first. | GPU busy 85% -> 97%; +5...+13% (EXP-041/042) |
| **Hide the host.** Queue the next draft chain before reading the verify result; compile the draft sampler; trim per-cycle bookkeeping. | +2.6...+4.0% (EXP-103/105) |
| **Exact sampling-mode speculation.** Standard speculative-sampling acceptance with the draft's *own* distribution q, plus block verification. | Tokens/cycle +1.2% (draft temperature 0.3) and +1.3% (block verification), proven exact by chi-square tests |

## 2. Practical tips

* **Define losslessness precisely.** Greedy: MTP-on text == MTP-off text for the same server configuration and numerics. Sampling: emitted-token distribution equals the target's. When a kernel changes the numerics (the final attention kernel does), compare MTP-on with MTP-off *using that kernel*, and report accuracy against an fp32 reference separately.
* **Test bit-identity for every kernel parameter, including "grid-only" ones.** A one-simdgroup MoE geometry changed nothing about the math and still violated row-exactness 144 times (EXP-084). Shape-specific unrolled tiles prove each shape with a single input, so re-run the 57-problem coding identity after every MLX or macOS update.
* **Make tests sensitive.** Kernel bit tests include mutation checks: deliberately wrong kernels must be detected. A check that cannot fail proves nothing; the audits found such checks (a thermal check, a "stability" load phase that was idle) and fixed them.
* **Verify windows longer than the kernel's row limit.** Long copy-heavy windows (9-16 rows) silently fell back to a different kernel; splitting windows into blocks of at most 8 rows, each seeing exactly the keys its last row may see, restored 57/57 identity (EXP-033).
* **Read the split size from a runtime buffer.** A compile-time constant per B triggers one kernel compilation per value and stalls decode.
* **Clear MLX's buffer pool after prefill or a prefix-cache restore.** A pool of stale buffers slows 128K decode by 6-7% until `mx.clear_cache` runs; test any change to clearing or allocation at 128K as well as at short contexts.
* **A/B inside one GPU session** (A/B/A/B, same server lifetime) and decide on greedy (identical content when the arithmetic is unchanged) and on tokens per cycle. With new numerics, compare cycle time at equal tokens/cycle, never medians of ms/cycle alone: an A/B on medians overstated one step's gain by up to 7x (-1.04 ms versus -0.15 ms at equal tokens/cycle at 64K).
* **Microbenchmark with dependent chains.** Independent calls overlap and overstate; confirm online. A host-side delay injected into one phase (A/B at identical content) shows quickly whether that phase is on the critical path: delay at the start of the draft cost 1:1, delay in the verify graph build cost nothing.
* **Measure with the prefix cache in a defined state.** Cache hit versus miss moves 2K/4K cells by ±4-5%. Losslessness tests run with the cache off (a restored prefix changes how the prompt is chunked, which can change greedy text even without speculation); speed is measured with it on.
* **Keep proposer-side tweaks separate from the verifier.** A 32K-token draft vocabulary, a draft temperature and a pre-queued draft change which tokens are proposed, never the probabilities used to accept or reject them.
* **Plan GPU memory for long contexts.** 256K needs the wired-memory limit raised, the prefix-cache hot tier off, a stepwise prefill through the SSD cache, and a swap check beforehand.
* **Use independent auditors.** A clean-context agent that re-runs the frozen verifier, recomputes medians from raw rows and compares the code byte-for-byte with the certified patch found a real issue in every audit that found one (a losslessness gap, a verifier path that could not fail, a latent multimodal bug, an optimization that never engaged on the benchmark).

## 3. Where the time goes (final configuration)

| Context | 2K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| Cycle time (ms) | 21.6 | 22.9 | 25.3 | 28.7 | 36.0 |
| Tokens per cycle | 2.38 | 2.35 | 2.42 | 2.25 | 2.26 |

* The part of the cycle that does not depend on context is about 21 ms: expert weights, Gated-DeltaNet, projections and the `lm_head`, within about 20% of the weight-read bandwidth floor (18.6 ms measured vs 15.7 ms for a 32K cycle).
* The ten full-attention layers add the context-dependent part: about 15 ms of the 36 ms cycle at 128K. Their pass 1 runs at about 90% of the measured realistic matrix-multiply throughput of the GPU.
* One extra verify row costs about 18% of a cycle (EXP-089), which is why a draft depth of 2 (3 verify rows) is used throughout.
* KV cache: 2.716 GB + 20,480 bytes per cached token (bf16).

## 4. Other findings of interest

* An 8-bit KV cache gives no speed-up on the 3-row verify (128K: 1.684 vs 1.681 ms/layer): the attention is bound by matrix-multiply and barrier work, not by bytes. The KV cache therefore stays bf16.
* The draft-side window (4096) on the MTP head is inactive at the benchmarked contexts; the gain attributed to the kernel changes is entirely the kernel.
* Copy-heavy generations reach 3.2-3.4 tokens/cycle; typical code generation 2.2-2.7 (p10/p50/p90 = 2.19/2.35/2.67 over 583 generations).
