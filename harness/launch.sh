#!/bin/bash
# Runs a harness non-interactively on a workdir with the prompt from a file, isolated from the user's global config
# (config/data/cache of the harness point to harness/home/<harness>).
# Usage: launch.sh <pi|hermes> <workdir> <prompt.txt> [port=8002]
# Env: PI_BIN (default: runtimes/harness/pi/node_modules/.bin/pi), HERMES_BIN (default: hermes on PATH), HARNESS_PYTHON (python with pytest), CTXWIN (declared context window, default 262144)
H=$1; WD=$2; PF=$3; PORT=${4:-8002}
R="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$R/.." && pwd)"
PROMPT="$(cat "$PF")"
export PYTHONDONTWRITEBYTECODE=1 NO_COLOR=1
export TERM=dumb
cd "$WD" || exit 99
case $H in
  pi)
    HM="$R/home/pi"; mkdir -p "$HM"
    sed "s/__PORT__/$PORT/; s/__CTX__/${CTXWIN:-262144}/" "$R/configs/pi_models.json" > "$HM/models.json"; cp "$R/configs/pi_settings.json" "$HM/settings.json"
    export PI_CODING_AGENT_DIR="$HM" PI_OFFLINE=1
    exec "${PI_BIN:-$ROOT/runtimes/harness/pi/node_modules/.bin/pi}" --provider omlx --model coder-q6 --no-context-files -p "$PROMPT" ;;
  hermes)
    HM="$R/home/hermes"; mkdir -p "$HM"
    sed "s/__PORT__/$PORT/; s/__CTX__/${CTXWIN:-262144}/" "$R/configs/hermes_config.yaml" > "$HM/config.yaml"
    export HERMES_HOME="$HM"
    exec "${HERMES_BIN:-hermes}" chat -q "$PROMPT" -Q --yolo --ignore-rules -t terminal,file,todo --provider omlx -m coder-q6 ;;
  *) echo "unknown harness"; exit 98 ;;
esac
