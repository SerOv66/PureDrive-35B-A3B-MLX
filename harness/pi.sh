#!/bin/bash
# The chosen harness (Pi) on the certified cand5 configuration. Usage: harness/pi.sh [workdir=.] [prompt...]  (no prompt = interactive TUI; with prompt = one non-interactive pass)
# Prerequisites: server on 8000 (harness/server_up.sh) and, for measurements, the proxy on 8002 (python harness/proxy.py); PORT=8000 skips the proxy.
# Env: PI_BIN (default: runtimes/harness/pi/node_modules/.bin/pi), CTXWIN (declared context window, default 262144)
R="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$R/.." && pwd)"; WD=$(cd "${1:-.}" && pwd); shift
PORT=${PORT:-8000}; HM="$R/home/pi"; mkdir -p "$HM"
sed "s/__PORT__/$PORT/; s/__CTX__/${CTXWIN:-262144}/" "$R/configs/pi_models.json" > "$HM/models.json"; cp "$R/configs/pi_settings.json" "$HM/settings.json"
export PI_CODING_AGENT_DIR="$HM" PI_OFFLINE=1
cd "$WD" && exec "${PI_BIN:-$ROOT/runtimes/harness/pi/node_modules/.bin/pi}" --provider omlx --model coder-q6 --no-context-files ${1:+-p "$*"}
