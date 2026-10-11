# Results

All numbers are single-stream decode throughput (tokens/s) on one Apple M4 Pro (14-core CPU, 20-core GPU, 48 GB unified memory, macOS 27.0.1),
MLX 0.32.3, oMLX 0.7.0 at commit `cc1fdc9` plus the certified patch, for a 6-bit-expert MLX checkpoint of a 35B-A3B mixture-of-experts coder model with a native MTP head.
"Greedy" = temperature 0; "default" = the model's default sampling (temperature 0.6, top-p 0.95, top-k 20). Decode tok/s = (generated tokens − 1) / (time of last token − time of first token) from the streamed response; it
excludes prefill. Sources: `results/verify/*_measurements.json` (extracts of the frozen-verifier output, see below), `results/RESULTS.csv` (one row per measurement), `quality/results/`. "EXP-nnn" are the project's internal experiment/audit numbers.

## 1. Speed: final certified configuration

Config `configs/q6_omlx_cand5.json`, independent audit "AUDIT EXP-110" (2026-10-10), frozen-verifier run `20261010-030533`
(`results/verify/20261010-030533_measurements.json`). Each cell is the **median of 3 runs × 512 generated tokens** on 3 different real coding prompts at an exact context length with the KV cache
actually filled with that many real tokens; min and max of the three runs are shown to make the spread visible.

| Context | Greedy median (min-max) | Default sampling median (min-max) |
|---|---|---|
| 2K (2,048) | **110.3** (106.0-113.4) | **111.9** (103.8-116.0) |
| 4K (4,096) | **107.3** (103.8-109.7) | **107.5** (100.9-125.0) |
| 16K (16,384) | **102.6** (96.6-102.9) | **100.9** (100.0-103.4) |
| 32K (32,768) | **95.8** (88.8-101.0) | **94.6** (89.8-98.2) |
| 40K (40,960) | **90.2** (89.7-94.6) | **86.7** (84.0-94.3) |
| 64K (65,536) | **78.3** (77.3-83.5) | **79.0** (76.1-87.6) |
| 128K (131,072) | **62.8** (60.9-66.2) | **62.2** (62.1-62.5) |
| 256K (262,144)* | 45.9 (45.8) | 44.8 (44.6) |

\* 256K: previous certified configuration `cand4` (EXP-105/106), measured with a separate harness, outside the frozen verifier (section 6).

* The published stored reports and quality JSONs omit the server's epoch timestamps (`created` fields from the model-list response); wall-clock times are kept. The files `results/verify/<ts>_measurements.json` are extracts of the verifier output (speed-threshold and verdict fields omitted; every other field, including the per-cell medians, config, quality, identity and stability data, is unchanged); `results/verify/extract.py` produces them and the per-run values are in the matching `*_matrix.csv`, `_constancy.csv` and `_stability.csv`.
* The three runs per cell use three different prompts (same corpus, three different coding questions), so the spread is largely content: how repetitive the generated code is determines how many draft tokens are accepted.
* **Constancy**: 128K context × 2,048 generated tokens, greedy: **63.2 tok/s**.
* **Stability**: after ≥ 10 minutes of continuous generation, 2K greedy / default: **110.8 / 112.2** tok/s; thermal pressure "Nominal" before and after; swap growth 0 MB; the whole verifier takes 16.2 minutes.
* Prefill speed and TTFT in the verifier CSVs are measured with a warm prefix cache (the KV is restored from SSD) and are **not** a prefill benchmark. Cold prefill measured separately: 128K ≈ 287 s (EXP-009); beyond 131K, ≈ 100-153 tok/s (EXP-106).

### Speed-up over the first lossless baseline

Baseline = oMLX main + Lightning MTP with row-exact verify forced (the first configuration that was both lossless and measured on the full matrix, EXP-012: 3 runs × 512 tokens per cell).

| | 2K | 4K | 16K | 32K | 40K | 64K | 128K |
|---|---|---|---|---|---|---|---|
| baseline greedy / default (EXP-012) | 85.8 / 82.8 | 85.1 / 78.7 | 70.8 / 70.0 | 62.8 / 62.0 | 60.4 / 57.9 | 50.6 / 49.1 | 34.6 / 34.2 |
| final greedy / default (EXP-110) | 110.3 / 111.9 | 107.3 / 107.5 | 102.6 / 100.9 | 95.8 / 94.6 | 90.2 / 86.7 | 78.3 / 79.0 | 62.8 / 62.2 |
| speed-up greedy | +29% | +26% | +45% | +53% | +49% | +55% | +82% |
| speed-up default | +35% | +37% | +44% | +53% | +50% | +61% | +82% |

## 2. Quality and losslessness

### Frozen quality gate (quick suite, profile "numeric")

Reference = the same model on the stock oMLX runtime, MTP off, no patches. Candidate = final configuration (verifier run `20261010-030533`).

| Part | Reference | Candidate | Allowed drop | Result |
|---|---|---|---|---|
| Coding (HumanEval+ every 6th: 28, MBPP+ every 13th: 29; base+plus tests) | 47 / 57 | 48 / 57 | 2 | PASS |
| Reasoning (30 GSM8K questions) | 28 / 30 | 28 / 30 | 2 | PASS |
| Tool calling (14 hand-written cases, strict structured calls) | 14 / 14 | 14 / 14 | 1 | PASS |
| Needle retrieval (50% depth at 32K, 64K, 131,072) | 3 / 3 | 3 / 3 | 0 | PASS |
| Identity vs stock (informational: the custom kernel changes summation order) | - | 0 / 6 identical | - | info |

Earlier certified configurations scored 26/30 on reasoning (v16, EXP-072, EXP-086) and 28/30 from v31 on; all passed.

### Losslessness (speculative decoding is exact)

| Test | Result | Source |
|---|---|---|
| Greedy output, MTP on vs MTP off, same server config and same numerics, 6 short prompts + 1 prompt at 32K, prefix cache off | 6/6 and 32K identical (inside the verifier) | `results/verify/20261010-030533_measurements.json` |
| Coding set (57 problems, up to 16-row verify windows engaged), MTP on vs off, prefix cache off | 57/57 identical texts at EXP-072, 078, 085, 086, 091, 092, 103, 105, 110, 114 | EXP notes |
| Long greedy generations (512 tokens) MTP on vs off at 8,102 (×3), 16,300, 49,102, 110,300 tokens of context (crossing every kernel split threshold) | all identical | AUDIT EXP-110 |
| Row-exactness of the attention kernel: every verify row vs a one-row decode at its position, including across split-size changes | 1149/1149 (author) and 1191/1191 (auditor, AUDIT EXP-092), mutation tests detect deliberately wrong kernels | `bench/tests/v31r_test.py`, `bench/audit/` |
| Sampling exactness, χ² tests of the emitted-token distribution (draft temperature, block verification, compiled sampler, pre-queued draft) | all p-values > 0.05 for the exact variants; deliberately broken variants rejected (p ≈ 1e-20 and below) | `bench/audit/*chi2*.py` |

Caveat stated by the auditors: with the prefix cache on (`CACHE=1`) greedy text can differ from cache-off runs *even with MTP off*, because the cache changes how the prompt is split into prefill chunks; losslessness is therefore certified with the cache off, speed is measured with it on.

### Kernel accuracy vs an fp32 reference (decode attention, teacher-forced, 256 steps; KL in nats)

KL(exact fp32 attention ‖ implementation). "Stock" is MLX's SDPA. The dominant error source common to all implementations is rounding of the output to bf16.

| Context | Stock SDPA | v16 | v31 + BRULE=1 | v31 + BRULE=2 (final) |
|---|---|---|---|---|
| 4K | 0.0095 | 0.0115 | 0.0106 | - |
| 32K | 0.0229 | 0.0150 | 0.0163 | 0.0260 |
| 64K | 0.0059 | 0.0036 | 0.0053 | 0.0044 |
| 128K | 0.0323 | 0.0365 | 0.0382 | 0.0541 |

Top-1 agreement with the exact reference is 95.7-98.7% for all kernels. The geometric-mean relative error against fp32 is 1.607e-3 (v31), 1.605e-3 (v16) and 1.782e-3 (stock); it is independent of the split size B (`bench/audit/berr.py`). The auditor judged the BRULE=2 differences to be metric noise from a few borderline steps (single-step KL up to 3.5) and re-measured on a fresh slice (124K, 300 steps): 0.0110 stock / 0.0121 BRULE=1 / 0.0123 BRULE=2 (paired difference +0.0002, 95% CI −0.0034...+0.0038, optimistic because the trajectory is autocorrelated). Raw files: `quality/results/kl_*.json`.

## 3. Comparison with other runtimes (same weights unless noted)

256-token, single-run smoke measurements (EXP-002/004/007/008), **not** matrix medians; given as reference points only. The last row is the matrix median from section 1.

| Setup | 2K | 16K | Source |
|---|---|---|---|
| mlx-vlm 0.7.6, 6-bit checkpoint, no speculation | 60.9 | 54.3 | EXP-002 |
| oMLX main, 6-bit checkpoint, speculation off | 72.95 | 63.74 | EXP-004 |
| oMLX main, 6-bit checkpoint, Lightning MTP with row-exact verify | 76.76 | 71.33 | EXP-006 |
| llama.cpp (master `de7fa0a`), 5-bit GGUF variant, MTP off | 52.27 | 47.03 | EXP-007 |
| llama.cpp, 5-bit GGUF variant, MTP on (n_max 3) | 55.83 | 59.89 | EXP-008 |
| **This work (final, matrix medians, greedy / default)** | **110.3 / 111.9** | **102.6 / 100.9** | EXP-110 |

## 4. Progression of adopted improvements

Every row is a run of the frozen verifier performed by an independent auditor on the configuration at that point (greedy / default medians, tok/s). Dates are 2026-10-09 (two last rows 2026-10-10). Per-technique gains, flags and exactness arguments: [`TECHNIQUES.md`](TECHNIQUES.md).

| Record | Verifier run | Time | Change added | 2K | 4K | 16K | 32K | 40K | 64K | 128K | 128K×2048 tok | 2K after 10 min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-016 | `20261009-011200` | 2026-10-09 01:37 | c35 v2 attention + row-exact MTP (EXP-014 config) | 80.8 / 81.6 | 83.3 / 83.3 | 74.8 / 73.9 | 68.1 / 67.3 | 63.3 / 63.7 | 57.7 / 58.1 | 45.2 / 43.0 | 42.6 | 80.6 / 79.9 |
| EXP-035 | `20261009-035434` | 2026-10-09 04:15 | + fixed depth 2, 32K draft vocab, shared rows, losslessness fix (EXP-033) | 91.9 / 87.2 | 90.3 / 84.6 | 81.9 / 76.8 | 73.4 / 72.2 | 69.0 / 68.6 | 64.6 / 59.8 | 47.2 / 48.8 | 46.2 | 93.1 / 86.5 |
| EXP-043 | `20261009-053039` | 2026-10-09 05:47 | + verify pipelining (flush every 2 layers) | 104.1 / 98.7 | 101.4 / 95.6 | 90.6 / 85.2 | 80.1 / 78.3 | 74.9 / 74.4 | 68.7 / 63.8 | 49.6 / 51.4 | 47.6 | 104.6 / 98.8 |
| EXP-056 | `20261009-072223` | 2026-10-09 07:41 | + attention kernel v16, MTP-head window 4096 | 105.7 / 99.9 | 101.9 / 96.2 | 93.4 / 87.7 | 83.8 / 82.0 | 78.8 / 78.3 | 73.4 / 68.2 | 54.5 / 56.8 | 52.7 | 105.9 / 99.9 |
| EXP-072 | `20261009-164933` | 2026-10-09 17:08 | + P2C merge, RPS=2, draft temperature 0.3 | 107.2 / 102.5 | 104.3 / 101.2 | 95.6 / 91.2 | 86.3 / 89.7 | 81.5 / 81.7 | 76.3 / 72.4 | 57.2 / 59.9 | 55.2 | 107.6 / 102.6 |
| EXP-086 | `20261009-191311` | 2026-10-09 19:38 | + fused add+RMSNorm, block verification, RPS6U, GDN 8-row recurrence | 109.6 / 105.0 | 106.3 / 105.5 | 97.3 / 92.2 | 87.7 / 84.3 | 82.7 / 80.7 | 77.4 / 73.2 | 57.7 / 56.8 | 55.8 | 109.9 / 105.0 |
| EXP-092 | `20261009-205630` | 2026-10-09 21:17 | + attention v31, position-based split (BRULE=1) | 107.7 / 108.3 | 105.0 / 104.4 | 98.6 / 89.8 | 90.3 / 92.8 | 86.5 / 84.3 | 80.1 / 76.4 | 61.0 / 61.3 | 58.5 | 107.9 / 107.2 |
| EXP-105 | `20261009-224835` | 2026-10-09 23:15 | + pre-queued draft, compiled draft sampler, fast commit, fused q/k prep | 109.2 / 110.7 | 105.9 / 106.2 | 99.1 / 98.5 | 90.7 / 89.3 | 88.8 / 82.0 | 81.7 / 76.3 | 60.6 / 63.5 | 58.3 | 109.6 / 110.9 |
| **EXP-110** | `20261010-030533` | 2026-10-10 04:03 | + wave-aligned split (BRULE=2), q/k prep on main model, fused attention output | **110.3 / 111.9** | **107.3 / 107.5** | **102.6 / 100.9** | 95.8 / 94.6 | 90.2 / 86.7 | 78.3 / 79.0 | 62.8 / 62.2 | 63.2 | 110.8 / 112.2 |

Notes from the auditors that qualify the table:

* Individual cells can move down between records for reasons unrelated to the change (e.g. default 32K/40K/128K at EXP-086, 64K greedy at EXP-110): the default-sampling output differs from run to run and some of the three prompts produce copy-heavy text that is accepted much more often. The real per-cycle gain of the last step (EXP-110) is −0.1...−0.3 ms at equal tokens per cycle (0.3-1.1%); an A/B on median ms/cycle had overstated it (−1.04 ms at 64K shrank to −0.15 ms at equal tokens/cycle).
* 2K and 4K cells depend on prefix-cache state (hit or miss) by ±4-5%; the configuration does not control this.
* The 16K default result (100.9) is partly content-driven; the greedy 16K median (102.6) has one run at 96.6.
* The MTP-head attention window (4096) is never active in the verifier matrix, so the gain credited to EXP-056 is entirely the attention kernel v16.

## 5. Tokens per cycle and cycle time

Speculation emits on average 2.2-2.4 tokens per cycle (draft depth 2, so 1 to 3 per cycle). Medians over the 6 requests per context of the certified verifier run (server log, `scripts/mtp_stats.py`):

| | 2K | 4K | 16K | 32K | 40K | 64K | 128K |
|---|---|---|---|---|---|---|---|
| tokens / cycle (median, min-max) | 2.38 (2.23-2.56) | 2.34 (2.18-3.24) | 2.35 (2.21-2.40) | 2.42 (2.20-2.71) | 2.32 (2.16-2.51) | 2.25 (2.19-2.53) | 2.26 (2.25-2.60) |
| accepted per draft token (median) | 0.76 | 0.73 | 0.74 | 0.71 | 0.73 | 0.69 | 0.70 |
| cycle time, ms (derived: 1000 × tok/cycle ÷ greedy tok/s) | 21.6 | 21.8 | 22.9 | 25.3 | 25.7 | 28.7 | 36.0 |

Other acceptance statistics: per-token acceptance of the first draft 0.78 (greedy) / 0.77 (default); second given first ≈ 0.70-0.72; over 583 non-benchmark generations the tokens/cycle p10 / p50 / p90 were 2.19 / 2.35 / 2.67. Copy-heavy generations reach 3.2-3.4 tokens/cycle.
The server-log "backbone ms" in this configuration double-counts host time (pre-queued draft), which is why the cycle time is derived from wall-clock throughput. At 256K the cycle is ≈ 50 ms (EXP-106). An extra verify row costs about 18% of a cycle (EXP-089); see [`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md).

## 6. Long context: 262,144 tokens

Separate measurement, outside the frozen verifier (which covers 2K-131,072). Harness `bench/prompts_256k.py` + `bench/run_256k.sh`: the frozen corpus (236,629 tokens) extended with the CPython 3.12.13 `multiprocessing/` sources; prompts up to 236K are identical to the frozen ones. **These rows were measured with the previous certified configuration `cand4` (AUDIT EXP-105; measured in EXP-106) in a separate harness, outside the frozen verifier**; the final configuration (`cand5`) was not re-measured at 256K. Setup: bf16 KV completely filled with real tokens, 3 runs × 512 tokens per mode, hot prefix-cache tier off, wired-memory limit raised to 43,008 MB during the runs.

| Context | Greedy median (min) | Default median (min) | Wired memory |
|---|---|---|---|
| 131,072 (same session) | 63.4 | 64.3 | 39.5 GB |
| 196,608 | 53.2 (50.6) | 50.1 (48.0) | 41.9-42.3 GB |
| 229,376 | 46.2 | 47.6 | - |
| 262,144 | **45.9** (45.8) | **44.8** (44.6) | 44.7 GB |

At 262,144: 2.38 tokens/cycle, per-draft acceptance 0.72, cycle ≈ 50 ms, thermal Nominal, swap flat (1.6 GB, the baseline of the machine). KV cache at 256K in bf16 ≈ 5.4 GB. Cold prefill of a 262K prompt was done in steps through the SSD prefix cache (0→131K 416 s, →196K 430 s, →229K 277 s, →262K 308 s).

Long-context quality at 262,144 tokens (`quality/longctx.py`, third-party Python sources as haystack, greedy): needle 3/3 (depths 10/50/90%), multi-key 3/3, multi-value 3/3, variable tracking 0/3 on the strict criterion (scores 0.83, 1.00 with additional variables listed, 0.33). Variable tracking is also 0/3 at 131,072 (0.33, 0.33, 0.83), so it is a property of the model on that task and not specific to 256K. Raw: `quality/results/longctx_131k_cand4.json`, `longctx_256k_cand4.json`.


## 7. Agentic coding harness (Phase 4)

Pi 0.73.1 is the recommended command-line coding agent on top of the certified server (`configs/q6_omlx_cand5.json`, prefix cache on). It was evaluated together with Hermes Agent v0.20.4 on 4 short tasks (bug fix, feature with tests, multi-file refactor, debugging from terminal errors) x 3 repetitions, plus a long session (>= 64K context). Medians per cell, sampling T 0.6 / top_p 0.95 / top_k 20 set server-side for both:

| Task | Harness | Success | Wall (s) | Requests | Prompt / completion tokens | Cache hit | Decode tok/s* | TTFT (s) |
|---|---|---|---|---|---|---|---|---|
| t1 bugfix | **Pi** | 3/3 | **15.4** | 5 | 18.3K / 1.0K | 83% | 110.5 | 1.02 |
| | Hermes Agent | 3/3 | 40.7 | 6 | 46.5K / 3.3K | 89% | 120.1 | 1.39 |
| t2 feature | **Pi** | 3/3 | **25.8** | 6 | 26.3K / 1.9K | 78% | 104.4 | 1.04 |
| | Hermes Agent | 3/3 | 45.9 | 10 | 86.4K / 3.5K | 92% | 108.8 | 0.81 |
| t3 refactor | **Pi** | 3/3 | **22.6** | 15 | 50.2K / 1.6K | 91% | 112.0 | 0.44 |
| | Hermes Agent | 3/3 | 39.9 | 7 | 57.9K / 2.8K | 85% | 117.3 | 2.11 |
| t4 debug | **Pi** | 3/3 | **15.5** | 6 | 24.4K / 0.9K | 77% | 116.1 | 0.83 |
| | Hermes Agent | 3/3 | 47.7 | 21 | 232K / 3.2K | 95% | 126.2 | 1.24 |

\* Decode tok/s are server-reported counters (completion tokens / generation time) over short agentic completions at 2K-30K context: **indicative only, not comparable to the frozen-verifier matrix** in section 1.

* **Pi completed 12/12 short-task runs** (24/24 across both harnesses) without a tool-call parsing error. Its median wall time is **1.8-3.1x lower** (13.7-68.5 s over all runs): a system prompt of about 2K tokens (4 tools) versus about 5.6K, and fewer turns. Prefix-cache hit 77-95%.
* **Long session** (nine CPython standard-library modules: fix a planted bug and document every public name by reading each file in full) **completed with Pi in both runs**, with the prefix cache serving 91-96% of prompt tokens:

| Pi, long session | Declared window | Wall | Requests | Max context | Cache hit |
|---|---|---|---|---|---|
| run 1 | 262,144 | 302 s | 48 | 65.5K | 96% |
| run 2 | 98,304 | 469 s | 31 | 103.9K | 91% |

* **Launch:** `harness/server_up.sh logs/harness_server.log`, then `PORT=8000 harness/pi.sh /path/to/project`. Tasks, scripts, configs and per-run summaries are in [`../harness/`](../harness/README.md).

## 8. Terminal-Bench 2.1 (agentic benchmark)

| Model | Setup | Terminal-Bench 2.1 (Terminus-2) |
|---|---|---|
| **This work** (6-bit, one M4 Pro) | 30-task pre-registered subset, 1 run per task | **73.3%** (22/30, 95% CI 55.6–85.8) |
| Ornith-1.5-35B-A3B (original, vendor-reported) | 89 tasks, mean of 5 runs | 67.8% |
| Qwen3.6-35B-A3B (reported on the Ornith-1.5 card) | 89 tasks | 52.5% |

Harbor 0.24.0 + Terminus-2 on the certified server; 30-task pre-registered subset, one run per task, agent time limits × 3 (max 90 min), containers under x86-64 emulation. Decode during the run: 109.6 tok/s median at 0–8K context, 85.0 at 32–64K, 62.9 at 96–132K. Protocol, differences from the vendor's setup and per-task results: [`TERMINAL_BENCH.md`](TERMINAL_BENCH.md).
