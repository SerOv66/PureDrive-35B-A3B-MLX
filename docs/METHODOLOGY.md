# Methodology

How the numbers were produced and why they can be trusted: what was measured, how losslessness was defined and proven, how the benchmark was frozen and audited, and how the
optimization loop was run. The measurements are in `docs/RESULTS.md`; the code that implements the protocol is in `bench/` and `quality/`.

## 1. Scope and definitions

**What is measured.** Single-stream decode throughput at 2K, 4K, 16K, 32K, 40K, 64K and 131,072 (128K) tokens of context, in **both** greedy mode (temperature 0) and the
model's **default sampling** (temperature 0.6, top-p 0.95, top-k 20), on one Apple M4 Pro (20-core GPU, 48 GB), for a 6-bit-expert checkpoint. Measurements at 262,144 tokens are made in a separate harness and reported separately. The benchmark also records stability (swap, thermal pressure after ≥ 10 minutes of continuous generation) and runs a quality gate (section 4).

**Constraints that define "lossless".** The weights are never reduced below their original precision (6-bit experts, 8-bit attention/shared/embedding/`lm_head` stay as they are). The KV cache
stays bf16. Allowed: anything that leaves the model's output distribution unchanged (speculative decoding with exact acceptance, kernels, layouts, scheduling).
Excluded (each would invalidate a result): fewer active experts (top-k < 8), skipped or pruned experts, shared expert off, layer skipping/early exit, replacing full attention, sliding-window/sparse attention, KV eviction,
truncating or summarising the context and calling it 128K, shrinking the vocabulary or `lm_head` of the target, speculative decoding with relaxed/typical/lossy acceptance.

## 2. Benchmark matrix (identical for every configuration)

| Item | Definition |
|---|---|
| Contexts | 2,048; 4,096; 16,384; 32,768; 40,960; 65,536; 131,072 tokens, **exact** token counts |
| Prompts | real code: the CPython 3.12 standard-library sources (asyncio, concurrent.futures, http, json, logging; 236,629 tokens, PSF license) truncated to the context length, followed by one of **three fixed real coding questions** (`bench/corpus/tasks.json`), wrapped in the model's chat template. No padding, no repetition, no synthetic text. The three runs of a cell use the three different questions |
| Generation | ≥ 512 new tokens per run (2,048 for the constancy test) |
| Runs | ≥ 3 per cell, **median** reported (minimum also reported); warm-up run discarded |
| Modes | greedy and default sampling, each in every cell |
| KV state | genuinely filled with that many real tokens (the SSD prefix cache may restore the prefix; the restored blocks are real KV, never an empty cache with faked positions) |
| Metrics | decode tok/s = (N−1)/(t_last − t_first) over streamed chunks; TTFT; swap; thermal pressure (via `powermetrics`); RSS; acceptance statistics from the server log |
| Constancy | 131,072 context × 2,048 generated tokens |
| Stability | after ≥ 10 min of *continuous* generation (not idle waiting), 2K × 3 runs × both modes again; swap growth < 1 GB; thermal pressure Nominal or Fair |

Decode and prefill are always reported separately; throughput of several concurrent streams is never aggregated (single stream, one request at a time).

**Forbidden benchmark shortcuts** (checked point by point by the auditor for every record): repetitive/padded/synthetic prompts that inflate acceptance; measuring only at temperature 0 when speculative gains collapse with sampling;
generations shorter than 512 tokens; a single run, best-of-N or discarding unfavourable outliers; counting proposed rather than accepted tokens; decoding at 128K without a really full KV cache; changing prompts, matrix or thresholds mid-project
(if needed: documented reason and re-measurement of everything); mixing up prefill and decode.

## 3. Losslessness: definition and how it was proven

**Definition.** For greedy decoding the output with speculation must be token-identical to the output without speculation, for the same server configuration and the same numerics.
For sampling the distribution of the emitted tokens must be exactly the target model's distribution.

**Why speculation can be exact.** Per cycle the target model scores the last accepted token and two draft tokens in one pass. Greedy: the emitted token at each position is the target's argmax, and a draft token is accepted only if it equals it.
Sampling: standard speculative sampling (Leviathan et al. 2023; Chen et al. 2023): accept a draft token with probability min(1, p/q), otherwise resample from the normalised residual max(p − q, 0), plus a bonus token from p when all drafts are accepted,
where q is *the distribution the draft was actually sampled from*; block verification (Sun et al., ICLR 2025, Algorithm 2) is the joint variant. The draft model (the checkpoint's own MTP head) may use any q, including one restricted to a 32K-token vocabulary or sampled at a lower temperature: only the acceptance rule matters.

**Why bits must match, and the "row-exact" design.** Greedy identity fails if the verify pass computes a row differently from a one-row decode (a different summation order flips near-ties). Therefore every kernel used in the verify pass is *row-exact*: row r of a multi-row call is bit-identical to the same row computed alone at the same position.
This is achieved by fixed reduction orders, absolute (position-based) split points, and per-column arithmetic independent of the tile width. When a kernel changes the numerics relative to the previous one (the final attention kernel does), losslessness is defined as MTP-on == MTP-off *with that same kernel*, and accuracy against an fp32 reference is reported separately (KL table in `docs/RESULTS.md`).

**Tests that were run (and re-run at each record):**

| Test | What it proves | Tool |
|---|---|---|
| Greedy generation, MTP on vs off, same config, prefix cache off, 6 short prompts + one 32K prompt | end-to-end identity in the verifier | inside `bench/verify_target.py`, `quality/run_quality.py --part identity` |
| **Coding identity, 57 problems**, MTP on vs off, cache off, verify windows of up to 16 rows engaged | the broadest end-to-end identity test; caught the 9-16-row window bug | `bench/identity_coding.sh`, `bench/audit/cmp_coding.py` |
| Long greedy generations (512 tokens) at 8,102 / 16,300 / 49,102 / 110,300 tokens of context | identity across kernel split-size thresholds | `bench/audit/identity_driver.sh` |
| Bitwise tests of each kernel against a one-row reference over many shapes (N from 300 to 131,072; L from 1 to 24), including *mutation tests*: deliberately wrong kernels must be detected | row-exactness, and that the test is sensitive | `bench/tests/*_test.py`, `bench/audit/v31_audit*.py` |
| **χ² tests** of the emitted-token distribution (two-sample, per position and per adjacent pair, ≥ 40,000 trials; toy Markov models with exact joint probabilities; 600 end-to-end completions MTP-off vs MTP-on), with deliberately broken samplers that must be rejected | exactness in distribution for the draft temperature, block verification, the compiled sampler, the pre-queued draft | `bench/audit/block_chi2.py`, `chi2_cmp.py`, `sample_dist.py` |
| KL divergence of decode attention against an exact fp32 reference, teacher-forced at 32K / 64K / 128K | accuracy of kernels whose numerics differ from stock | `quality/kl_decode.py`, `bench/audit/kl_decode_ab.py` |

Known limits, all stated by the auditors: identity is certified with the prefix cache **off** (with it on, the prompt is prefilled in different chunks, which changes greedy text even without speculation); the verifier's own identity check covers verify windows of 3 and 8 rows (other window lengths are covered by the external 57-problem test, repeated at every audit).

## 4. Quality gate

A fixed, repeatable subset of public benchmarks, run through an OpenAI-compatible HTTP client (`quality/run_quality.py`), greedy, in a sandbox for generated code:

* coding: HumanEval+ (28 problems) and MBPP+ (29), base+plus tests (EvalPlus semantics);
* reasoning: 30 GSM8K problems; tool calling: 14 cases with strict structured calls;
* long context: needle retrieval at 32K / 64K / 131,072 (and a RULER-like set — multi-key, multi-value, variable tracking — at 131K and 262K, `quality/longctx.py`);
* identity: 6 fixed prompts, informational against stock (different numerics) and mandatory (6/6) for MTP on vs off.

Thresholds are fixed in `quality/THRESHOLDS.md` ("lossless" profile: scores equal, no needle lost; "numeric" profile for kernels whose numerics differ: coding and reasoning may lose at most 2 problems, tools 1, no needle cell lost). A configuration that is faster but does not hold the gate is not an improvement.

## 5. Frozen verifier and independent audits

* **Frozen verifier** (`bench/verify_target.py`): the benchmark harness behind every certified number. It runs the whole matrix, the constancy and stability tests, the quick quality suite and the identity checks, each as a separate subprocess with a hard time budget (≤ 10 minutes per step, ≤ 45 minutes in total; measured 16-23 minutes), and writes a JSON report with the per-cell medians and the quality and identity results.
* **Freeze.** The verifier, the benchmark client and prompts, the corpus and tasks, the quality suite, the thresholds and the quality reference are pinned by sha256 (`results/FROZEN_original_sha256.txt`) and made read-only. A heartbeat process re-checked the hashes continuously; a change raised a tamper alarm. A real bug in the verifier could only be fixed by: a written explanation, approval by the auditor, new hashes, and **re-measurement of all previous configurations**. This happened three times in the first hours (versions v2, v2b, v2c: for example, a load phase that had been idle waiting, a thermal check that could never fail, and two fail-open paths found by the auditor in the first fix).
* **Adversarial auditor.** For every new record a separate agent with a clean context, not the author of the code, had to try to find the trick: re-run the frozen verifier, recompute medians from the raw rows and count server completions (no hidden runs), re-run the 57-problem identity test and the kernel bit tests, compare the code under test with the certified patch byte for byte, check the forbidden-shortcut list point by point, and compare greedy output with the baseline. The verdict was recorded as `APPROVED` or `REJECTED` with the reason. One audit rejected a candidate (EXP-030 → the 9-16-row window bug); another approved conditionally and forced a fix before certification (EXP-110, a latent multimodal bug). Eight records were approved this way (audits of EXP-035, 043, 056, 072, 086, 092, 105 and 110; `docs/RESULTS.md`, section 3).
* **Dossiers for exceptional steps.** Any change that could touch precision (for example quantizing the KV cache, or using a lower-bit variant) required a written dossier with profiling and recorded lossless alternatives, approved by the auditor. The KV cache and the weights stayed at their original precision.
* Before every claimed record, the list of forbidden shortcuts was re-read and the fact recorded in the lab notebook.

## 6. A/B protocol for optimizations

1. One variable at a time; every change behind an environment flag, default off.
2. A/B only inside **one GPU session** (alternating A/B/A/B, same server lifetime): numbers from different sessions are never compared, because clock, memory state and prefix-cache state drift.
3. Decide on **greedy** (identical content when the arithmetic is unchanged) and on **tokens per cycle**: with new numerics, greedy text changes, so compare cycle time at equal tokens/cycle (ANCOVA) or on identical content, never on medians of ms/cycle alone, and never on tok/s per cell when the cache state differs.
4. Microbenchmarks use dependent chains; the real effect is then confirmed online.
5. Screening runs (1 run, 512 tokens) may select candidates; only the frozen verifier certifies. Parameters of proposer-side knobs (e.g. the draft temperature) are not tuned on benchmark prompts; calibrations use separate non-benchmark prompts.
6. A hypothesis list with a falsifiable expected gain; three consecutive experiments without gain trigger a full re-profile and a change of technique family.

## 7. Safety and resource discipline

* **One model loaded at a time**, one request at a time. No second model process while a server is loaded (an incident with a concurrent training test pushed swap over 2 GB and killed the server three times).
* **Memory guard.** A background process polls swap and memory pressure every 5 s and kills the registered run if swap exceeds 2 GB or pressure turns critical. Every benchmark and server runs through a wrapper with a hard time budget (`scripts/run.sh`: ≤ 600 s per measurement step, long builds separately declared) that registers the process for the guard.
* **256K runs** need the GPU wired-memory limit raised (a platform-specific `sysctl`, to 43,008 MB, reset afterwards), the prefix-cache hot tier switched off, and swap checked beforehand.
* Original model files are never modified (work on copies/symlinks with recorded hashes); disk kept ≥ 40 GB free; profiler traces limited to a few seconds (a single recording once left ≈ 52 GB of temporary files).

## 8. How the work was carried out (autonomous AI-agent workflow)

The project was run almost entirely by AI coding agents (Claude Code) under a written mission specification, with a human owner who defined the rules.

* **Roles.** An *orchestrator/author* agent planned, wrote code and ran experiments; cheaper *executor* sub-agents ran benchmarks, sweeps and log reading (never two models on the GPU at once); *independent auditor* agents with clean context verified every record and every dossier (section 5).
* **Persistence.** A lab notebook (append-only, one entry per experiment with hypothesis, change, command, result, quality, conclusion and next step), a CSV with one row per measurement, a short state file with the best result, the gap and a numbered hypothesis queue, and a README updated at every record. These survive context compaction and agent hand-offs: a new agent re-reads them first and continues.
* **Working rules.** A queue of hypotheses, at least one recorded experiment per hour, a full re-profile after three consecutive experiments without gain, and no claim of a limit without profiling and benchmarks — estimates are not measurements.
* **Known risks of this setup, observed here:** a test agent run in a sandbox copy acted on the real workspace and wrote invalid notebook entries (declared invalid in errata); a shell word-splitting mistake produced a plausible but invalid configuration run; agents tend to over-credit microbenchmark gains. The frozen verifier, hash pinning and independent audits exist for these reasons.
