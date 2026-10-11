# Terminal-Bench 2.1 on a single Apple M4 Pro

The optimized 6-bit model, served by the certified configuration (`configs/q6_omlx_cand5.json`, lossless MTP speculative decoding), was evaluated on [Terminal-Bench 2.1](https://www.tbench.ai/) with the same agent harness the model vendor used: Harbor 0.24.0 with the Terminus-2 agent (JSON parser). Every task ran end to end on the Mac: the model on the GPU, the task containers in a local Linux VM.

## Result

| Model | Setup | Terminal-Bench 2.1 (Terminus-2) |
|---|---|---|
| **This project**: Coder-35B-A3B, 6-bit MLX, M4 Pro 48 GB | 30-task pre-registered subset, 1 run per task | **73.3%** (22/30, Wilson 95% CI 55.6–85.8) |
| Ornith-1.5-35B-A3B (original model, vendor-reported) | 89 tasks, mean of 5 runs | 67.8% |
| Qwen3.6-35B-A3B (as reported on the Ornith-1.5 model card) | 89 tasks | 52.5% |

**Reading.** The local 6-bit model with its lossless speed-ups scores in line with the vendor's published number for the original model: 67.8 lies inside our confidence interval. This is a subset evaluation, not a leaderboard submission and not a full reproduction; the differences are listed below.

Without `custom-memory-heap-crash` (its debugging tools cannot work under x86-64 emulation, see below): 21/29 = 72.4% (CI 54.3–85.3).

## Speed during the benchmark

884 model requests, 2.61 M generated tokens, single stream, sampling T 1.0 / top_p 1.0 / top_k 20. Decode speed is the server's counter (completion tokens / generation time) per request, grouped by the request's context length:

| Context | Requests | Decode tok/s, median (p25–p75) |
|---|---|---|
| 0–8K | 179 | **109.6** (102.7–117.9) |
| 8–16K | 123 | **104.2** (97.2–109.7) |
| 16–32K | 142 | **94.7** (89.0–101.1) |
| 32–64K | 212 | **85.0** (78.4–92.0) |
| 64–96K | 104 | **72.3** (68.0–77.1) |
| 96–132K | 86 | **62.9** (60.1–66.9) |
| 132–157K | 30 | **58.4** (56.0–61.6) |

These agentic numbers match the frozen-verifier matrix in [RESULTS.md](RESULTS.md) at the same context lengths. Between turns the prefix cache re-uses the conversation, so re-reading the prompt typically takes a few seconds even above 100K tokens.

## Protocol

* **Harness:** Harbor 0.24.0, agent `terminus-2`, `parser_name=json`, one trial at a time (`-n 1`, `-k 1`).
* **Model server:** the certified configuration, unchanged, behind a logging proxy ([`../harness/proxy.py`](../harness/proxy.py)) that records sampling parameters, token usage and timing for every request.
* **Sampling:** temperature 1.0 and top_p 1.0 (vendor values); top_k 20 from the model's `generation_config.json`, which a vLLM server applies by default when the client does not send it; max_tokens 32,768.
* **Chat template:** the model's own template with no injected system prompt; every past assistant turn keeps its reasoning (`<think>…</think>`) in the history, as the vendor's published template renders it (Terminus-2 `interleaved_thinking=true`, template kwarg `preserve_thinking=true`). The rendered history was checked byte for byte against the vendor's template.
* **Context:** 131,072 tokens declared to the agent, which summarizes the conversation when it approaches the limit.
* **Timeouts:** each task's agent timeout × 3, capped at 90 minutes (the tasks run under x86-64 emulation).
* **Task selection, fixed before any model call:**
  1. 17 of the 89 tasks need resources this machine cannot provide (≥ 8 GB RAM, 4 CPUs, images > 1.4 GB, or a nested qemu VM) and were excluded.
  2. The remaining 72 were shuffled with a fixed seed ([`make_order.py`](../results/terminal-bench/make_order.py), [`tb21_order.txt`](../results/terminal-bench/tb21_order.txt)).
  3. An oracle pre-flight (the tasks' reference solutions) removed 2 tasks the reference solution itself cannot solve in this environment ([`tb21_oracle_exclusions.txt`](../results/terminal-bench/tb21_oracle_exclusions.txt)).
  4. The first 30 in order form the evaluation set ([`tb21_selected.txt`](../results/terminal-bench/tb21_selected.txt)).
* **Rules:** one attempt per task; a single re-run only for an infrastructure failure (none occurred); every task scored by the benchmark's own verifier, unmodified. A task that reaches the timeout is still verified on its final state, as in the benchmark.
* **Review:** before and during the run, independent reviewers audited the whole pipeline (sampling actually applied, prompt rendering, verifier results, driver logic, resources). All 30 results come from one run under the final audited protocol; earlier pilot runs with a different sampling and template configuration were discarded before scoring.

## Differences from the vendor's setup

| Difference | Ours | Vendor | Likely effect on the score |
|---|---|---|---|
| Tasks | 30-task random subset of the 72 feasible | all 89 | The 17 excluded heavy tasks may be harder than average: possibly favourable to us |
| Runs per task | 1 | 5 (mean) | Wider uncertainty (see CI), no bias |
| Agent time limit | task timeout × 3, max 90 min | 4 h (model card) | Unfavourable: 7 of the 8 failures ended at the time limit |
| Hardware | Apple M4 Pro, ~60–110 tok/s; containers 1 CPU / 2 GB under x86-64 emulation | vLLM server (GPU not stated); 32 CPU / 48 GB | Unfavourable: slower generation and slower, emulated tools |
| Weights | 6-bit MLX derivative (abliterated, MTP head grafted) | original BF16 | Unknown |
| Sampling extras | top_k 20, max_tokens 32,768 | not stated (vLLM defaults would apply top_k 20) | Probably neutral |

## Per-task results

| Task | Result | Wall (s) | Turns | Decode tok/s | Note |
|---|---|---|---|---|---|
| sanitize-git-repo | fail | 344 | 28 | 93.8 | One of two secrets missed (search filtered by file type) |
| build-pmars | **pass** | 2730 | 58 | 54.5 | Finished early, kept verifying until the time limit |
| custom-memory-heap-crash | **pass** | 4053 | 70 | 71.8 | Solved although gdb cannot attach under emulation |
| extract-elf | **pass** | 2745 | 26 | 74.1 | Verified on final state at the time limit |
| chess-best-move | fail | 2741 | 55 | 73.2 | Board given as an image; no answer file written |
| crack-7z-hash | **pass** | 396 | 24 | 100.3 | |
| pytorch-model-cli | **pass** | 1051 | 31 | 93.0 | |
| sqlite-db-truncate | fail | 2737 | 7 | 77.2 | Very long reasoning turns (13–33K tokens each) |
| prove-plus-comm | **pass** | 157 | 5 | 105.9 | |
| polyglot-c-py | **pass** | 596 | 9 | 89.0 | |
| log-summary-date-ranges | **pass** | 116 | 7 | 111.0 | |
| count-dataset-tokens | **pass** | 653 | 17 | 95.2 | |
| largest-eigenval | fail | 2732 | 27 | 73.6 | 26/27 tests; one speed test missed by 0.4% (83.02 vs 82.73 µs) |
| large-scale-text-editing | **pass** | 1156 | 27 | 86.0 | |
| git-multibranch | **pass** | 1348 | 67 | 82.2 | |
| kv-store-grpc | **pass** | 217 | 21 | 109.3 | |
| cobol-modernization | fail | 2738 | 24 | 71.4 | Required files missing at the time limit |
| break-filter-js-from-html | **pass** | 810 | 13 | 82.8 | |
| tune-mjcf | fail | 2762 | 48 | 73.8 | Simulation at 103.8% of the original time (60% required) |
| nginx-request-logging | **pass** | 229 | 17 | 106.6 | |
| merge-diff-arc-agi-task | **pass** | 581 | 24 | 97.1 | |
| constraints-scheduling | **pass** | 371 | 8 | 97.7 | |
| fix-code-vulnerability | **pass** | 491 | 23 | 101.8 | |
| bn-fit-modify | **pass** | 1602 | 42 | 84.4 | |
| git-leak-recovery | **pass** | 144 | 10 | 104.7 | |
| raman-fitting | fail | 2737 | 41 | 74.8 | Result file missing at the time limit |
| password-recovery | **pass** | 1799 | 26 | 72.7 | |
| feal-differential-cryptanalysis | **pass** | 4302 | 38 | 72.0 | |
| fix-git | **pass** | 250 | 12 | 103.4 | |
| circuit-fibsqrt | fail | 5444 | 45 | 70.6 | Time limit; 2/3 tests passed |

Decode tok/s here is the per-task average over all its requests (long-context turns pull it down). Median wall time 1104 s, median 25 turns.

## Files

[`../results/terminal-bench/`](../results/terminal-bench/): `tb21_results.csv` (one row per task: reward, turns, tokens, wall time, decode speed, memory), `analyze_tb21.py` (pass@1 with Wilson CI; `python3 analyze_tb21.py`), `make_order.py` + `tb21_order.txt` (seeded order, reproducible from the task set), `tb21_selected.txt`, `tb21_oracle_exclusions.txt`.
