# AGENTS.md — install and run PureDrive-35B-A3B-MLX

Instructions for an AI coding agent (Claude Code, Codex, Cursor, Copilot, Gemini CLI, Pi, …) asked to "install this" or "get it running" on a Mac. A human can follow the same steps. All paths are relative to the cloned GitHub repository.

**Result when done:** a local, OpenAI-compatible server at `http://127.0.0.1:8000/v1`, model id `coder-q6`, serving a 35B-A3B mixture-of-experts coder model with exact (lossless) multi-token-prediction speculative decoding and custom Metal kernels. Measured on an Apple M4 Pro (48 GB): 110 tok/s at 2K context, 96 tok/s at 32K, 63 tok/s at 128K, single stream.

## Rules for the agent

* Run every step from the repository root and check its expected output before going on. If a check fails, use the Troubleshooting table; if that does not solve it, stop and show the user the exact error.
* **Ask the user first** before: downloading the model weights (about 28 GB), installing anything system-wide (Homebrew packages, Xcode command-line tools), and raising the prefix cache above its 10 GB default. Never use `sudo`.
* If the user already has the checkpoint on disk, use it (step 3, option A) instead of downloading it again. Never modify the checkpoint files.
* Only one model server at a time: a second large model in memory causes swapping. Start the server in the background (step 4) and tell the user how to stop it.
* `logs/serve.log` contains the machine's host name and local IP addresses: redact them before showing or sharing the log.
* Report what you did, with the numbers from step 5.

## 0. Check the machine

```bash
uname -m                      # must print arm64 (Apple silicon)
sysctl -n hw.memsize          # bytes; measured configuration: 48 GB = 51539607552
sw_vers -productVersion       # measured on macOS 27.0.1
df -h .                       # free space: ~28 GB weights + ~1.1 GB runtime + prefix cache (10 GB by default)
```

* Not `arm64` → stop: this project needs Apple silicon (MLX, Metal).
* Less than 48 GB of unified memory → tell the user this is untested (the weights alone take about 28 GB of memory) and ask whether to continue.
* Other Apple-silicon chips work, but speed differs: the kernels were tuned on a 20-core-GPU M4 Pro.

## 1. Tools

```bash
git --version                 # Xcode command-line tools; if missing ask the user to run: xcode-select --install
uv --version                  # if missing, ask the user, then: brew install uv   or   curl -LsSf https://astral.sh/uv/install.sh | sh
```

`uv` downloads Python 3.12 by itself. Nothing else is needed to run the server.

## 2. Clone and build the runtime

```bash
git clone https://github.com/SerOv66/PureDrive-35B-A3B-MLX PureDrive-35B-A3B-MLX && cd PureDrive-35B-A3B-MLX
scripts/setup_omlx.sh
```

What it does: clones oMLX (github.com/jundot/omlx) into `runtimes/omlx`, pins commit `cc1fdc9`, applies `patches/omlx_c35_EXP110.patch`, copies the Metal attention kernel to `bench/kernels/`, creates `runtimes/venv-omlx` and installs oMLX with its pinned dependencies (`mlx==0.32.3`, about 1.1 GB). First run: a few minutes (downloads); with a warm `uv` cache: about 15 s. Safe to re-run.

Expected: a line starting with the kernel hash `e61d6d11ad7e9b4c`, then `done: runtime in runtimes/omlx, interpreter runtimes/venv-omlx/bin/python`.

## 3. Model weights

The checkpoint is [peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP) (MIT license, 6 safetensors shards, about 28 GB). It is not part of this repository.

**Option A — already on disk** (for example in LM Studio's models folder):

```bash
mkdir -p models/omlx-models
ln -sfn "/absolute/path/to/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP" models/omlx-models/coder-q6
```

**Option B — download** (ask the user first; about 28 GB):

```bash
uvx --from huggingface_hub hf download peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP --local-dir models/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP
mkdir -p models/omlx-models
ln -sfn "$PWD/models/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP" models/omlx-models/coder-q6
```

Check (either option):

```bash
ls models/omlx-models/coder-q6/                    # config.json, model-00001-of-00006.safetensors … model-00006-of-00006.safetensors, tokenizer files
grep -c '"language_model.mtp' models/omlx-models/coder-q6/model.safetensors.index.json    # 42: the MTP head is present
```

Another MLX checkpoint of the same architecture (`qwen3_5_moe`, 35B-A3B), quantization layout and an MTP head under `language_model.mtp.*` may also work, but it is untested and its speed depends on how often its MTP head's drafts are accepted.

## 4. Start the server

```bash
mkdir -p logs
nohup ./serve.sh > logs/serve.log 2>&1 &
echo $! > logs/serve.pid
for i in $(seq 100); do
  grep -q 'Loaded model: coder-q6' logs/serve.log && { echo "ready after $((i*3)) s"; break; }
  kill -0 "$(cat logs/serve.pid)" 2>/dev/null || { echo "server exited, see logs/serve.log"; tail -20 logs/serve.log; break; }
  sleep 3
done
```

* Expected: `ready after` about 10–20 s. `/v1/models` answers a few seconds *before* the model is loaded: wait for the `Loaded model: coder-q6` log line, as above.
* `serve.sh` starts oMLX with the certified configuration `configs/q6_omlx_cand5.json` (every flag of the optimized runtime) and an SSD prefix cache in `models/omlx-cache` (up to 10 GB), which makes long multi-turn sessions fast.
* Options go **before** `nohup` (or before `./serve.sh` in a terminal): `CACHE_MAX=20GB nohup ./serve.sh …` (bigger cache; ask the user), `CACHE=0 …` (no cache), `PORT=8001 …` (other port).
* Interactive use: run `./serve.sh` in its own terminal; Ctrl-C stops it.
* Stop the background server: `kill "$(cat logs/serve.pid)"` (port 8000 is free again within a second or two). The process renames itself `omlx-server`: use the pid, not a name pattern.

## 5. Smoke test

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"coder-q6","messages":[{"role":"user","content":"Write a Python function that returns the n-th prime number, with a docstring and three tests."}],"max_tokens":1024}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]; m=c["message"]; print("reasoning:", (m.get("reasoning_content") or "")[:300]); print("answer:", (m.get("content") or "")[:600]); print("finish:", c.get("finish_reason")); print("usage:", r.get("usage"))'
```

Expected: a coherent answer, `finish: stop`, and `usage` with `completion_tokens`, `generation_tokens_per_second` and `time_to_first_token`. On an M4 Pro, `generation_tokens_per_second` is about 100–125 tok/s; the first request after a cold start can be slower (weights are paged in from disk), so repeat it once before judging speed. The server log also prints one line per request (`Chat completion: … tokens in …s (… tok/s)`, prefill included) and an `MTP[…]` line with the draft acceptance rate.

## 6. Use it

Any OpenAI-compatible client works:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | any non-empty string, e.g. `local` |
| Model | `coder-q6` |
| Context window | 262,144 tokens (measured up to 256K) |
| Tool calling | yes, OpenAI `tools` format (`tool_calls` with JSON-string arguments) |
| Reasoning | the model may think before answering; the thinking is returned in `message.reasoning_content`, the answer in `message.content` |
| Sampling defaults | temperature 0.6, top_p 0.95, top_k 20 (set by `serve.sh`; a request's own values win). Greedy (temperature 0) output is byte-identical to non-speculative decoding. |

Tested coding agents: **Pi 0.73.1** (recommended; `npm install @mariozechner/pi-coding-agent@0.73.1 --prefix runtimes/harness/pi`, then `PORT=8000 harness/pi.sh /path/to/project`), Hermes Agent 0.20.4 (`harness/configs/hermes_config.yaml`), Terminus-2 via Harbor (Terminal-Bench). Configuration examples are in `harness/configs/`.

## 7. Optional: reproduce the published measurements

The full frozen benchmark (7 context lengths, greedy and sampling, quality and identity checks, about 20 minutes) needs GNU coreutils (`brew install coreutils`), CPython 3.12.13 sources to rebuild the benchmark corpus, and an idle machine. Follow `docs/REPRODUCE.md`; compare with `results/verify/20261010-030533_measurements.json`. The verifier's thermal reading calls `sudo -n powermetrics` by default: do not set up sudo; run it with `POWERMETRICS_CMD=true` (the thermal field is then left empty) unless the user has configured passwordless `powermetrics` themselves.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `git apply --check` fails in step 2 | `runtimes/omlx` was edited or is broken: `rm -rf runtimes/omlx` and re-run `scripts/setup_omlx.sh` |
| `port 8000 is already in use` | another server is running: `lsof -nP -iTCP:8000 -sTCP:LISTEN`; stop it or use `PORT=8001` |
| `model missing` | the link `models/omlx-models/coder-q6` must point to the checkpoint directory (the one containing `config.json`) |
| `server exited` in step 4 | read the last lines of `logs/serve.log`; most often the runtime is missing (re-run step 2) or memory is short (close other large apps) |
| Slow, or heavy swap | close other memory-heavy apps; run only one model server; check `sysctl vm.swapusage` |
| Disk filling up | the prefix cache grows up to `CACHE_MAX` (10 GB by default) in `models/omlx-cache`; lower it or use `CACHE=0`; deleting the folder is safe while the server is stopped |

## Uninstall

Stop the server and delete the repository folder: runtime, virtual environment, prefix cache and (option B) the weights all live inside it. Outside it: oMLX writes a one-line launcher shim, `~/.omlx/bin/omlx-cluster-python`, on every server start (remove `~/.omlx` if no other oMLX installation uses it), and `uv` keeps a package cache in `~/.cache/uv` (`uv cache clean` removes it).
