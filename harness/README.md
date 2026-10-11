# Phase 4 - agentic coding with the certified decode server

Pi 0.73.1 is the recommended command-line coding agent on top of the certified configuration (`configs/q6_omlx_cand5.json`, MTP d=2, prefix cache on). It was evaluated together with Hermes Agent v0.20.4 on five repeatable tasks. This directory holds the measurement tooling and the summarized results; raw per-request logs and agent transcripts are not published.

**Highlights.** 24/24 short-task runs succeeded with both harnesses; Pi's median wall time is 1.8-3.1x lower (15-26 s per task); Pi completed the long session (>= 64K context) in both runs; the prefix cache served 77-95% of prompt tokens.

> **Caveat on throughput numbers.** Decode tok/s in this directory are server-reported counters (completion tokens / `generation_duration`) over short agentic completions at 2K-30K context. They are indicative only and are **not comparable** to the frozen-verifier matrix in `docs/RESULTS.md` (3 x 512 generated tokens, real full-context prompts, median of runs). Only the verifier numbers are certified.

## Harnesses evaluated

| Harness | Version | Notes |
|---|---|---|
| **Pi** (recommended) | 0.73.1 (npm package `@mariozechner/pi-coding-agent`, local prefix) | non-interactive mode `pi -p`; 4 tools (read, bash, edit, write); system prompt about 2.5K characters (about 2K tokens with tool schemas) |
| Hermes Agent | v0.20.4 | `hermes chat -q ... -Q --yolo`; with `-t terminal,file,todo` the system prompt is 8.1K characters (about 5.6K tokens); compaction in mid-run at a 0.5 threshold of the window |

Global harness configurations were not touched: each harness runs with its own config directory under `harness/home/` generated from `configs/`.

## Method

* **Tasks** (`tasks/<task>/{seed,ref,reset.sh,check.sh,prompt.txt}`; Python; nothing from `bench/`):
  * `t1_bugfix` - two bugs in a ledger library, pytest;
  * `t2_feature` - implement an LRU cache with TTL (8 tests);
  * `t3_refactor` - rename and move `calc_total` across three files, grep-checked;
  * `t4_debug` - a script that crashes three times on a dirty CSV; the agent must run it in the terminal and fix it from the errors;
  * `t5_long` - long session, see below.
  * Each `check.sh` works on a copy and restores the seed's tests/data, so an agent cannot win by editing them. Every check was verified to fail on `seed/` and pass on `ref/` (the reference was never shown to the agent).
  * The original task prompts and seed texts were not in English; the published prompts, docstrings, comments, labels and test data are **English translations** (logic unchanged, checks re-run: fail on `seed/`, pass on `ref/`). The timings in this directory were measured with the original-language prompts. The prompts ask, in short: t1 "tests in `tests/` fail; run `python -m pytest -q`, find the cause in `ledger.py` and fix it, do not touch the tests, answer with a one-line summary"; t2 "implement the `LRUCache` skeleton in `cache.py` (stdlib only) so that all tests in `tests/test_cache.py` pass"; t3 "rename `calc_total` to `compute_total` and move it to a new `shop/totals.py`, keep `VAT_RATE` and `apply_discount` in `pricing.py`, update all uses, leave no reference to `calc_total`"; t4 "run `python report.py data/sales.csv`, read the errors, fix `report.py` (not the CSV) until the report follows every rule of its docstring, then run pytest"; t5 below.
* **Runs.** `run_one.sh <harness> <task> <rep>`: reset (git init + commit), non-interactive harness launch with a 900 s cap, check. `run_matrix.sh` runs the interleaved matrix and restarts the server before the 1500 s budget of `scripts/run.sh`.
* **Measurement.** `proxy.py` (port 8002 to 8000) records per request: TTFT, server usage (prompt/completion tokens, `cached_tokens`, `generation_tokens_per_second`), tool calls, and tool-call text leaks (= parsing errors). Decode tok/s = completion tokens / server `generation_duration` (excludes prefill). `summarize.py` aggregates (`results/RUNS.txt`, `results/SUMMARY.txt`); `parse_server_log.py` reads prefix-cache hits and MTP statistics from the server log.
* **Same sampling for both.** Neither harness sends temperature/top_p/top_k, so `server_up.sh` sets T 0.6, top_p 0.95, top_k 20 server-side (the benchmark's "default" profile). Max 16,384 tokens per reply, no thinking.
* **Isolation.** `--no-context-files` (Pi) and `--ignore-rules` (Hermes); otherwise they load the workspace's own instruction files (15.8K extra tokens; one early Pi run was discarded for that reason).
* **Setup.** One machine (Apple M4 Pro, 48 GB), one model loaded at a time, measured on 2026-10-10.

## Results

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

More detail on the tables above:

* Success 24/24 and **0** tool-call text leaks (no Qwen3-Coder tool-format parsing error) with either harness; Pi's wall-time ranges: 13.7-68.5 s over all 12 runs (Hermes 20.8-150.9 s).
* Decode is about 105-126 tok/s with both (contexts 2K-30K); the small differences follow the generated content and request lengths, not the harness.
* TTFT on tool requests: Pi median 0.54 s (p95 2.0 s), Hermes 0.94 s (p95 3.2 s). The prefix cache works with both (stable prefix, append-only turns).
* Long session (`t5_long`): nine CPython 3.13.14 standard-library modules (argparse, optparse, configparser, difflib, textwrap, shlex, glob, fnmatch, getopt; about 370 KB) in `lib/`; fix a planted bug in `textwrap.py` and write `API.md` by reading every file in full. Check: pytest + at least 90% of each module's public names (`__all__`) in `API.md`. With Pi, decode falls from 99 to 67 tok/s as the context grows to 65K (median TTFT 2.2 s); with a 98,304 window the run reached 103.9K tokens of context, and in `-p` mode compaction only runs at the end of the run. The modules are not shipped: `tasks/t5_long/build_seed.sh` copies them from a local CPython 3.13.14 and plants the one-line bug (sha256-checked). Per-run lines for every harness are in `results/RUNS.txt`.
* System-level observation: server decode falls from about 100 to 70 tok/s between 2K and 65K-100K context (consistent with `docs/RESULTS.md`); the prefix cache makes every turn pay only for new tokens.

## Why Pi is recommended

1. Full success on the short tasks (12/12) at 1.8x-3.1x lower total time: fewer prompt tokens (4 tools, a system prompt of about 2K tokens versus about 5.6K), fewer turns, terser output.
2. It completed the long session in both runs, including with a 96K declared window.
3. Lower median TTFT, a stable prefix, and a minimal isolated configuration (two JSON files).

Usage notes for Pi: in `-p` mode it does not compact mid-run (keep the declared window at or below the server's real window, 262,144); it has no sandbox and runs arbitrary bash, so use a dedicated working directory.

## How to launch

Prerequisites: the patched oMLX server of `docs/REPRODUCE.md`, Node.js for Pi, Python with `pytest` for the checks, `gtimeout` (GNU coreutils) for `run_one.sh`.

```bash
export REPO_DIR=$PWD                       # repository root
export MODEL_DIR=/path/to/your/mlx/checkpoint
export HARNESS_PYTHON=/path/to/python-with-pytest
export PI_BIN=/path/to/node_modules/.bin/pi          # npm install @mariozechner/pi-coding-agent@0.73.1 --prefix <dir>
harness/server_up.sh logs/harness_server.log         # cand5 + sampling T0.6/top_p0.95/top_k20; budget 1500 s (scripts/run.sh)
PORT=8000 harness/pi.sh /path/to/project             # interactive TUI
PORT=8000 harness/pi.sh /path/to/project "prompt"    # one non-interactive pass
scripts/kill_run.sh server-omlx                      # stop the server
```

To measure: `python harness/proxy.py 8002 8000 &`, then `harness/run_one.sh pi t1_bugfix 1` (or `harness/run_matrix.sh "pi hermes" "t1_bugfix t2_feature t3_refactor t4_debug" "1 2 3"`), then `python harness/summarize.py --runs`. `CTXWIN=<n>` sets the declared context window (default 262,144). The `t5_long` seed needs `CPYTHON313_STDLIB=/path/to/lib/python3.13 harness/tasks/t5_long/build_seed.sh` once.

## Files

| Path | Content |
|---|---|
| `server_up.sh` | start the certified server with identical server-side sampling |
| `pi.sh`, `launch.sh` | launch Pi (interactive/non-interactive), or either harness in isolation for the matrix |
| `run_one.sh`, `run_matrix.sh` | one run / the interleaved matrix |
| `proxy.py` | measuring proxy (TTFT, usage, cache hits, tool calls, leaks) |
| `capture_server.py` | fake server (no model) to inspect what a harness sends |
| `parse_server_log.py`, `summarize.py` | server-log extraction and aggregation |
| `configs/` | Pi `models.json`/`settings.json` templates and the Hermes config; the API key `local` is a dummy string required by the clients, not a secret |
| `tasks/` | the five task sandboxes (seed, reference, reset/check scripts, prompt) |
| `results/SUMMARY.txt`, `results/RUNS.txt` | per-cell medians and one line per run (28 runs) followed by the summary table |

Not published: raw `proxy.jsonl` request logs, agent transcripts and diffs.
