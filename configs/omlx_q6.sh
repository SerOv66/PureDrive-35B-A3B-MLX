#!/bin/bash
# oMLX main (runtimes/omlx @ cc1fdc9) serving Q6 oQ6e, Lightning MTP via env MTP=true|false (default false). Settings home: runtimes/omlx-home.
# Env: OMLX_MODEL_ROOT (default models/omlx-models), MTP=true|false, OMLX_MTP_ROW_EXACT_FORCE=1 (lossless verify), CACHE=1, PORT. Usage: configs/omlx_q6.sh  (foreground; wrap in scripts/run.sh server <budget> <log> -- ...)
W="$(cd "$(dirname "$0")/.." && pwd)"
V=$(grep -n "^SETTINGS_VERSION" "$W/runtimes/omlx/omlx/model_settings.py" | sed 's/.*= *//')
H="$W/runtimes/omlx-home"; mkdir -p "$H"
cat > "$H/model_settings.json" <<J
{"version": $V, "models": {"coder-q6": {"mtp_enabled": ${MTP:-false}, "is_pinned": true, "max_context_window": 262144 ${OMLX_EXTRA:+, $OMLX_EXTRA}}}}
J
# CACHE=1: paged SSD prefix cache in models/omlx-cache (needed to reuse long prefixes); default off
CACHE_ARGS=(--no-cache); [ "${CACHE:-0}" = 1 ] && CACHE_ARGS=(--paged-ssd-cache-dir "$W/models/omlx-cache" --paged-ssd-cache-max-size ${CACHE_MAX:-40GB} --hot-cache-max-size ${HOT_MAX:-4GB})
exec "$W/runtimes/venv-omlx/bin/omlx" serve --model-dir "${OMLX_MODEL_ROOT:-$W/models/omlx-models}" --base-path "$H" --port ${PORT:-8000} \
  --max-concurrent-requests 1 --memory-guard off "${CACHE_ARGS[@]}" ${OMLX_ARGS:-}
