# Reproducing the results

The repository contains the **patch**, the **kernel**, the **configurations**, the **benchmark/verifier code** and the **raw results**. It does **not** contain model weights, the oMLX source tree, or the prefix cache.
Everything below is written for the setup the numbers were measured on; other Apple-silicon machines will give different absolute numbers (the kernel tuning, e.g. the wave-aligned split, assumes a 20-core GPU).

## 1. Requirements

| | Used for the published results |
|---|---|
| Machine | Apple M4 Pro, 14-core CPU, 20-core GPU, 48 GB unified memory (48 GB is sufficient up to 256K context with the settings below) |
| OS | macOS 27.0.1 (Metal GPU family of the M4) |
| Python | 3.12.13 |
| MLX | `mlx==0.32.3` (pinned by oMLX; the custom kernels are ABI-coupled to this version) |
| Other packages | `mlx-lm` (git `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`, 0.31.4.dev), `mlx-vlm` 0.7.7, `transformers` 5.17.0, `numpy` 2.3.5, `safetensors` 0.8.0 (all installed by oMLX's `pyproject.toml`) |
| Runtime | oMLX 0.7.0, git commit **`cc1fdc9`** of github.com/jundot/omlx (Apache-2.0) |
| Tools | `git`, [`uv`](https://github.com/astral-sh/uv), GNU `timeout` (`brew install coreutils` → `gtimeout`) used by `scripts/run.sh` and the verifier, `curl`, `shasum` |
| Free disk | ≥ 40 GB (prefix cache up to 40 GB; checkpoint ≈ 28 GB) |
| Model | [peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP) (MIT), **not distributed here** — see "The model" |

### The model

Architecture (read from the checkpoint's `config.json`): `Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`), 40 decoder layers = 30 Gated-DeltaNet linear-attention layers + 10 full-attention layers (every 4th),
hidden size 2048, full attention with 16 query heads / 2 KV heads (GQA 16:2), head dim 256, partial rotary 0.25 (M-RoPE sections 11/11/10), 256 routed experts (top-8) + 1 shared expert,
expert intermediate size 512, vocabulary 248,320, 1 native MTP layer, context up to 262,144, plus a 27-layer vision tower (unused here).
Quantization (MLX affine): **6 bit, group size 64 for the routed experts**; 8 bit for attention/GDN input projections, shared expert, embeddings and `lm_head`; a few output projections 6 bit at group size 128 ("oQ6e");
the MTP layer carries 6/8-bit tensors under `language_model.mtp.*`. The checkpoint used for the measurements is **[peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP)** (MIT), and **no weights are redistributed** here.
Lineage: [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (ornith-ai, MIT) → [Huihui-Ornith-1.5-35B-A3B-abliterated](https://huggingface.co/huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated) (huihui-ai, MIT) → Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP (peculiar-ragdoll, MIT; Ornith's MTP head grafted back, oMLX oQ6e requantization); the architecture is the Qwen3.5/3.6-35B-A3B family ([Qwen](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Apache-2.0).
**Download source:** that Hugging Face repo (for example `hf download peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP --local-dir "$MODEL_DIR"`); the sha256 of its 6 weight shards matches the checkpoint measured here.
The only local difference is the system identity line of the chat template (`chat_template.jinja` and `tokenizer_config.json`): the measured copy says "You are Coder.", the repo says "You are Cyber-Tiel-Coder, a variant of Ornith-1.5-35B-A3B."; the weights are identical (generated text can differ slightly with the repo's line, as the system prompt differs).
To reproduce on another model you need an MLX checkpoint with this architecture, this quantization layout and an MTP head stored under `language_model.mtp.*`; results on a different fine-tune will differ in acceptance rate.

## 2. Build the runtime

```bash
git clone <this repository> repo && cd repo
export REPO_DIR=$PWD
scripts/setup_omlx.sh          # clones oMLX, checks out cc1fdc9, applies the patch, installs the kernel, creates runtimes/venv-omlx
```

`scripts/setup_omlx.sh` does:

1. `git clone https://github.com/jundot/omlx runtimes/omlx && git -C runtimes/omlx checkout cc1fdc9`
2. `git -C runtimes/omlx apply ../../patches/omlx_c35_EXP110.patch` (the certified patch, 14 files: 13 under `omlx/patches/` plus `omlx/engine/vlm.py`; `git apply --check` is run first). The patch was verified to apply cleanly to `cc1fdc9`.
3. `cp kernels/gqa_attn.py bench/kernels/gqa_attn.py` — the patch loads the Metal kernel module from `<repo>/bench/kernels/` (four directories above `omlx/patches/`). Certified kernel sha256 starts with `e61d6d11ad7e9b4c`; patch sha256 starts with `9d916294656a670d`.
4. `uv venv --python 3.12 runtimes/venv-omlx && uv pip install -e runtimes/omlx`.

A second environment is used by the benchmark client and the quality suite (they need only `transformers` and the standard library):

```bash
uv venv --python 3.12 runtimes/venv-mlx
uv pip install --python runtimes/venv-mlx/bin/python transformers==5.17.0 numpy
export PYTHON=$PWD/runtimes/venv-mlx/bin/python    # used by the scripts; defaults to runtimes/venv-mlx/bin/python
```

The tests under `bench/tests/` that import oMLX modules (`qk_prep_test.py`, `attn_out_test.py`) and the kernel tests that import `mlx` need `runtimes/venv-omlx/bin/python`.

## 3. Point the scripts at the model

```bash
export MODEL_DIR=/path/to/Coder-35B-A3B-MLX-MTP-6q       # the checkpoint directory (used by bench/prompts.py and several tests)
mkdir -p models/omlx-models
ln -s "$MODEL_DIR" models/omlx-models/coder-q6           # oMLX serves a directory of models; the served name is "coder-q6"
```

`configs/omlx_q6.sh` writes oMLX's `model_settings.json`, selects the SSD prefix-cache directory `models/omlx-cache` when `CACHE=1`, and starts the server on port 8000 with `--max-concurrent-requests 1`.
`C35_DRAFT_VOCAB` points to `configs/draft_vocab_32k.json` (the exact 32K draft vocabulary used; it can be rebuilt with `scripts/build_draft_vocab.py`, which takes token frequencies from an independent corpus, never from the benchmark text).

## 4. Start the server (certified configuration)

For everyday use, `./serve.sh` starts this configuration in the foreground, without a time budget, adding server-side sampling defaults (T 0.6, top_p 0.95, top_k 20) that a request's own values override (see `AGENTS.md`). The configuration is `configs/q6_omlx_cand5.json`; its `env` block lists every flag. Manually:

```bash
MTP=true OMLX_MTP_ROW_EXACT_FORCE=1 C35_GQA_ATTN=1 C35_GQA_VERSION=31 C35_GQA_BRULE=2 C35_GQA_P2=3 C35_RPS=2 \
C35_DRAFT_TEMP=0.3 C35_FUSE_GLUE=1 C35_BLOCK_VERIFY=1 C35_RPS6U=4 C35_GDN_REC_ROWS=8 C35_FAST_COMMIT=1 C35_QK_PREP=1 \
C35_QK_PREP_MAIN=1 C35_ATTN_OUT=1 C35_DRAFT_COMPILE=1 C35_MTP_PREDRAFT=1 CACHE=1 C35_SHARED_ROWS_MIN_TILES=0 \
C35_VERIFY_FLUSH=2 C35_MTP_WINDOW=4096 C35_DRAFT_VOCAB=configs/draft_vocab_32k.json \
OMLX_EXTRA='"mtp_fixed_depth": 2' configs/omlx_q6.sh
```

Use explicit `VAR=value` arguments, not a string variable: in `zsh` a string of assignments is not word-split and silently starts a stock server (this produced one invalid run, recorded as such).
For measurements wrap the server in `scripts/run.sh server <budget_s> <logfile> -- configs/omlx_q6.sh` so that it has a hard time budget and is registered for a memory guard.

## 5. Run the benchmark and the verifier

```bash
CPYTHON_STDLIB=/path/to/cpython-3.12.13/lib/python3.12 python bench/corpus/build_corpus.py   # rebuilds the frozen benchmark corpus byte-exactly (sha256-checked)
python quality/run_quality.py --fetch-data          # downloads HumanEval+, MBPP+, GSM8K; verifies sha256 (quality/data/SHA256SUMS)
scripts/freeze.sh                                   # writes FROZEN.sha256 for the verifier/prompts/quality suite, makes them read-only
python bench/verify_target.py configs/q6_omlx_cand5.json     # full verification: ~16-23 min, writes logs/verify/<ts>.json
```

`verify_target.py` starts and stops the servers itself (candidate, then candidate/baseline without speculation for the identity check), so **no other server may be running on port 8000**, and the machine should be otherwise idle.
Thermal pressure is read with `powermetrics`; the command is configurable (`POWERMETRICS_CMD`, default `sudo -n powermetrics --samplers thermal -n 1 -i 2000`, which needs passwordless sudo for that command or the script is run as root).
The verifier prints and stores the measurement matrix and the quality, identity and stability checks; an optional per-cell speed threshold can be added with `--speed-target <tok/s>` (default off). It writes a JSON report with the per-cell medians, constancy, stability, quality and identity results. **Reference result:** `results/verify/20261010-030533_measurements.json` (summarised in `docs/RESULTS.md`); expect ±4-5% on the 2K/4K cells depending on prefix-cache state.

Single measurements (this is what the verifier calls):

```bash
python bench/bench_http.py --url http://127.0.0.1:8000 --model coder-q6 --ctx 2048 32768 131072 --runs 3 --modes greedy default --gen 512 \
       --exp MY-EXP --variant Q6-oQ6e --runtime cand5 --mtp on --spec lightning-mtp   # appends rows to RESULTS.csv
bench/screen.sh MY-EXP label VAR=val ...      # quick screening: server + 1 greedy run at 2K/32K/128K
bench/ab_screen.sh MY-EXP configs/A.json configs/B.json   # A/B of two configurations in one session
```

Interpreting a run: the three runs of a cell use three different prompts, so min-max spread is content; compare configurations on medians in the *same session* and prefer greedy plus tokens/cycle (`scripts/mtp_stats.py <server log>`), as described in `docs/METHODOLOGY.md`.

## 6. Quality and losslessness checks

```bash
python quality/run_quality.py --url http://127.0.0.1:8000 --model coder-q6 --part all --suite quick --out quality/results/mine.json
python quality/compare.py quality/reference/Q6-oQ6e_quick.json quality/results/mine.json --profile numeric     # PASS/FAIL
python -m unittest quality/test_quality.py -v            # unit tests of the suite (mock server, no model)
bench/identity_coding.sh configs/q6_omlx_cand5.json out/identity    # 57-problem coding identity, MTP on vs off, cache off (starts servers itself)
python quality/kl_decode.py --help                        # KL of decode attention vs exact fp32 (needs the model; teacher-forced)
python quality/longctx.py --help                          # needle / multi-key / multi-value / variable tracking up to 262K
```

Kernel tests (they need the GPU but not the model): `python bench/tests/v31r_test.py` (row-exactness across split sizes), `bench/tests/qk_prep_test.py`, `bench/tests/attn_out_test.py`, `bench/tests/block_verify_test.py`, and the auditors' tools in `bench/audit/`
(`v31_audit_b2.py` mutates the kernel to prove the test is sensitive; with `C35_GQA_BRULE=2` the older `v31_audit.py` is **not** sensitive and must not be used).
Do not run GPU tests while a model is loaded: with a server running, even a few-second test can push swap past the memory guard.

## 7. The 256K measurements

```bash
export CPYTHON_STDLIB=/path/to/cpython-3.12.13/lib/python3.12       # needed to build the extension corpus (sha256-checked)
python bench/prompts_256k.py build                                  # writes bench/corpus/corpus_ext_256k.txt
python bench/prompts_256k.py check                                  # exact lengths, prefix identity with the frozen prompts
sudo sysctl iogpu.wired_limit_mb=43008                              # raise the GPU wired limit during the run only; set back to 0 afterwards
bench/run_256k.sh configs/q6_omlx_cand4.json MY-256K 196608 262144  # stepwise prefill through the SSD prefix cache, then 3 x 512 tokens per mode
bench/quality_256k.sh MY-Q256K prefill:131072 prefill:262144 cases  # long-context quality at 262,144 tokens
```

The 256K runs were made with the `cand4` configuration (the best certified configuration at that time) and the hot tier of the prefix cache switched off (`HOT_MAX=0`, done by the scripts).

## 8. Layout of this repository

| Path | Content |
|---|---|
| `patches/` | `omlx_c35_EXP110.patch` (certified) against oMLX `cc1fdc9` |
| `kernels/` | `gqa_attn.py` (certified Metal GQA attention kernel module), `microbench/` (MMA throughput, occupancy, dispatch cost probes) |
| `configs/` | server launcher `omlx_q6.sh`; JSON configurations of every certified step (`q6_omlx_v16_win4k`, `q6_omlx_cand_p2c_rps2_dt03`, `cand2`...`cand5`); `draft_vocab_32k.json` |
| `bench/` | `verify_target.py` (frozen verifier), `bench_http.py`, `prompts.py`, `prompts_256k.py`, `corpus/`, run scripts; `tests/` kernel/identity tests; `audit/` the auditors' tools |
| `quality/` | quality suite, thresholds, frozen reference run, KL and long-context tools, result JSONs |
| `scripts/` | `setup_omlx.sh`, `freeze.sh`, `run.sh` / `kill_run.sh` (time-budgeted, registered runs), `mtp_stats.py`, `build_draft_vocab.py` |
| `results/` | `RESULTS.csv` (every measurement), `verify/` (frozen-verifier outputs of every certified run; the final one with all its sub-results), `FROZEN_original_sha256.txt` |
| `docs/` | methodology, techniques, results, engineering notes, reproduction |

## 9. Known limitations of this repository

* The published copies of the frozen files were sanitised (absolute paths removed, messages and comments in English, server epoch timestamps removed from the stored reports, `verify_target.py` with an optional speed threshold and a configurable interpreter and power-metrics command); their sha256 therefore differ from `results/FROZEN_original_sha256.txt`. `scripts/freeze.sh` pins the published copies instead. Logic is unchanged.
* `results/RESULTS.csv` rows refer to log files that are not published; configuration names in its `flags` column are the project's internal labels (for example `cand4`); paths that appear in some rows were replaced by `$REPO/...`.
* Some helper scripts in `bench/audit/` refer to `logs/...` locations of the original workspace for their *inputs* (server logs and generated texts of an earlier run); they are provided as evidence of the audit method rather than as turnkey tools.
* Prefix-cache state changes 2K/4K results by ±4-5%; start measurements from a defined cache state if you need to resolve differences smaller than that.
