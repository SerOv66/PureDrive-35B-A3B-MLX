#!/bin/bash
# Serve the model with the certified configuration: OpenAI-compatible API at http://127.0.0.1:${PORT:-8000}/v1, model id "coder-q6".
# Foreground, no time limit (Ctrl-C stops). Prerequisites: scripts/setup_omlx.sh, checkpoint linked at models/omlx-models/coder-q6.
# Env: PORT (8000), CONFIG (configs/q6_omlx_cand5.json), CACHE (1 = SSD prefix cache in models/omlx-cache, 0 = off), CACHE_MAX (10GB), HOT_MAX (4GB).
# A variable of the config's "env" block that is already set in the environment overrides the config value.
# Server-side sampling defaults: temperature 0.6, top_p 0.95, top_k 20; values sent in a request take precedence.
# Works with macOS's bash 3.2.
set -euo pipefail
cd "$(dirname "$0")"
PY=runtimes/venv-omlx/bin/python
CONFIG=${CONFIG:-configs/q6_omlx_cand5.json}
export CACHE_MAX=${CACHE_MAX:-10GB}
[ -x "$PY" ] || { echo "runtime missing: run scripts/setup_omlx.sh first" >&2; exit 1; }
[ -f models/omlx-models/coder-q6/config.json ] || { echo "model missing: ln -s /path/to/checkpoint models/omlx-models/coder-q6" >&2; exit 1; }
if curl -s -m 2 "http://127.0.0.1:${PORT:-8000}/v1/models" >/dev/null; then echo "port ${PORT:-8000} is already in use" >&2; exit 1; fi
lines=$("$PY" scripts/config_env.py "$CONFIG")
args=()
while IFS= read -r l; do args+=("$l"); done <<< "$lines"
echo "starting oMLX with $CONFIG (${#args[@]} settings, prefix cache up to $CACHE_MAX); ready when the log shows 'Loaded model: coder-q6'" >&2
exec env "${args[@]}" configs/omlx_q6.sh
