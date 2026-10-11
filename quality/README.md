# quality/ — quality suite

Quality gate for every candidate configuration: capability (coding, reasoning, tool calling, long context) and **exactness**
of speculative decoding. Pure HTTP client (stdlib) for any OpenAI-compatible server (oMLX, llama.cpp `llama-server`). It loads
no model and starts no server: it only talks to `--url`. Greedy decoding everywhere (`temperature 0, top_k 1, top_p 1`, `seed 1234`).

## Commands

Run from the repository root; `PY` = the project Python (see `docs/REPRODUCE.md`); server already running at `$URL`.

```
$PY quality/run_quality.py --fetch-data                      # download the public datasets, verify against quality/data/SHA256SUMS
$PY quality/run_quality.py --url $URL --model NAME --part coding    --suite quick --out quality/results/<tag>.json
$PY quality/run_quality.py --url $URL --model NAME --part reasoning --suite quick --out quality/results/<tag>.json
$PY quality/run_quality.py --url $URL --model NAME --part tools     --suite quick --out quality/results/<tag>.json
$PY quality/run_quality.py --url $URL --model NAME --part needle    --suite quick --out quality/results/<tag>.json
$PY quality/run_quality.py --url $URL --model NAME --part identity  --suite quick --out quality/results/<tag>.json [--ref <ref>.json]
$PY quality/run_quality.py --url $URL --model NAME --part all       --suite quick --out quality/results/<tag>.json [--ref ...]
$PY quality/compare.py <ref>.json <cand>.json --profile {lossless|numeric}   # PASS/FAIL per part
$PY -m unittest quality/test_quality.py -v                   # tests with a local mock server, no model
$PY quality/run_quality.py --selftest-sandbox --suite full   # the 541 canonical solutions must pass the sandbox
```

- Several `--part` runs with the same `--out` (same suite and `--model`) are **merged** into the same JSON.
- `--suite quick` = fixed subset (below); `--suite full` = the whole dataset. Reference and candidate must use the same suite.
- Exit codes: `run_quality.py` 0 ok, 2 server unreachable. `compare.py` 0 PASS, 1 FAIL, 2 not comparable.
- The reference is measured **once** per variant on a faithful runtime without experimental optimizations, with the same harness;
  every candidate is then run with `--part all --ref <ref>` and `compare.py`.

### Thinking off and tools

Chat requests (coding/reasoning/tools) send `chat_template_kwargs: {"enable_thinking": false}`; tools use the OpenAI
`tools` + `tool_choice: "auto"` schema and the result is read from `message.tool_calls[]`. Long-context and identity prompts use
`/v1/completions` with the chat template applied locally (needle with `enable_thinking=False`, identity with the template default
as the speed-bench prompts). The server context must be >= 131072 + 64 tokens for the 128K needle.

## What each part measures

| part | measures | score | typical failure caught |
|---|---|---|---|
| `coding` | HumanEval+ and MBPP+ (EvalPlus), greedy pass@1, **base+plus** tests in a sandbox | problems passed / total (quick 57, full 541) | kernels that change the trajectory, numeric errors breaking syntax/logic |
| `reasoning` | GSM8K test, exact numeric final answer (`#### <number>`, robust parsing) | correct / total (quick 30, full 200) | multi-step reasoning degradation |
| `tools` | 14 hand-written cases (OpenAI `tools` schema): structured call present, right name, valid JSON arguments (strings, ints, bool, enum, arrays of objects, parallel calls, choice between similar tools, no arguments) | correct / 14. **Strict**: the call must be in structured `tool_calls` | server tool-call parser, template, JSON errors |
| `needle` | long-context retrieval: a unique needle sentence inserted into the **real corpus** (`bench/corpus/corpus.txt`) at depth 10/50/90%, **exact** total context 32768/65536/131072 tokens, question at the end; found = exact value in the answer | cells found / total (quick 3, full 9) | long-context attention/KV/RoPE regressions |
| `identity` | 6 greedy generations of 256 tokens via `/v1/completions` (3 = speed-bench prompts at ctx 2048, 3 = short hand-written coding prompts); with `--ref` compares text with the reference: `IDENTICAL` or `DIVERGE at char N` | identical / 6 | non-lossless speculative decoding / MTP |

## Quick subset (deterministic)

- **HumanEval+** (164 tasks, v0.1.10): every 6th (positions 0, 6, 12, ...) -> 28 tasks.
- **MBPP+** (378 tasks, v0.2.0, minus 1 exclusion -> 377): every 13th -> 29 tasks. Documented exclusion: **Mbpp/255** (its canonical solution alone exceeds ~1.5 GB RSS).
- **GSM8K test** (1319): indices `int(i*1319/30)`, i = 0..29 (quick) / `int(i*1319/200)` (full).
- **Tools**: 14 fixed cases (`TOOL_CASES` in `run_quality.py`).
- **Needle**: quick = depth 50% x {32768, 65536, 131072}; full = {10, 50, 90%} x 3 contexts. Each cell's needle is derived from `random.Random("needle-1234-<ctx>-<depth>")`.
- **Identity**: 6 fixed prompts (`part_identity`), the same in quick and full.

## Coding sandbox

Each solution runs in a separate `python -I -B` subprocess in a throw-away temp directory, closed stdin, minimal environment, its own
process group (killed at the end/timeout). Network and writes outside the temp dir are denied with macOS `sandbox-exec`
(if unavailable only the Python-level guard remains: sufficient for model-generated code under test, not a sandbox against hostile code).
Limits: `RLIMIT_CPU`, `RLIMIT_FSIZE` 16 MB, `RLIMIT_NOFILE` 256, memory 1536 MB enforced by a parent RSS watchdog, candidate timeout 10 s
(extended up to 60 s at 3x the canonical solution time where the canonical one is slow, as EvalPlus does).
EvalPlus 0.3.1 semantics (Apache-2.0): expected outputs come from running the canonical solution on the same inputs; exact comparison, `atol` for floats,
special oracles (MBPP 164/295/581/558, set-eq, not-none; HumanEval/32). Declared deviations: HumanEval/32 also accepts output equal to the canonical
one; Mbpp/255 excluded; base+plus tests run in one process with stop at first failure; code extraction is regex-based.
Validation: `--selftest-sandbox --suite full` passes all 541 canonical solutions; tests cover wrong solutions, plus-only failures, infinite loops, `10**10**10`, memory, network, writes outside the temp dir, crashes, noisy output, temp-dir cleanup.

## Datasets (`quality/data/SHA256SUMS`, verified at every run; the data files are not redistributed here)

| file | origin | sha256 | license (as recalled by the authors, re-check before redistributing) |
|---|---|---|---|
| `HumanEvalPlus.jsonl.gz` | github.com/evalplus/humanevalplus_release (v0.1.10) | `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101` | EvalPlus Apache-2.0; original HumanEval MIT |
| `MbppPlus.jsonl.gz` | github.com/evalplus/mbppplus_release (v0.2.0) | `af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63` | EvalPlus Apache-2.0; original MBPP CC BY 4.0 |
| `gsm8k_test.jsonl` | github.com/openai/grade-school-math (`test.jsonl`) | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` | MIT |

## Output JSON

`schema, created, suite, url, model, harness{script, sha256, thresholds_sha256, python, platform}, datasets{file: sha256}, sampling, server{...}, timings{part: s}, total_seconds, parts{<part>: {score, passed, n, seconds, items[]}}`.

## Known limits

- Greedy decoding with a non-deterministic server (batching, parallel reductions) can differ between two identical runs: measure the reference twice before trusting tight thresholds.
- Thinking OFF is a template/server contract; if `chat_template_kwargs` is ignored the model thinks and runs out of tokens (`finish: length`, low scores).
- No KL/perplexity here: see `quality/kl_decode.py` and `quality/results/kl_*.json`.

## Files

`run_quality.py` (suite) · `compare.py` (PASS/FAIL) · `THRESHOLDS.md` (thresholds, JSON block) · `test_quality.py` (mock-server tests) ·
`longctx.py` (long-context cases up to 256K) · `kl_decode.py` (KL of decode attention vs stock) · `kv_quant_err.py` (8-bit KV format study) ·
`reference/` (frozen reference run) · `results/` (JSONs) · `data/SHA256SUMS`.
