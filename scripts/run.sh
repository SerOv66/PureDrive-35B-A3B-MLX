#!/bin/bash
# Run a command under a hard budget and register it for a memory-guard / heartbeat monitor.
# Usage: scripts/run.sh <tag> <budget_s> <logfile> -- cmd args...
#   tag: bench|server|build|dl|train|... (the memory guard kills everything except build*/dl*)
# gtimeout puts the command in its own process group (pgid = gtimeout pid) -> kill -- -pid kills the tree.
set -u
W="$(cd "$(dirname "$0")/.." && pwd)"
tag=$1; budget=$2; log=$3; shift 3; [ "${1:-}" = "--" ] && shift
mkdir -p "$(dirname "$log")" "$W/logs/run"
echo "[run.sh $(date '+%F %T')] tag=$tag budget=${budget}s cmd: $*" >>"$log"
gtimeout -k 30 "$budget" "$@" >>"$log" 2>&1 &
pid=$!
echo "$pid $(date +%s) $budget $tag $log" > "$W/logs/run/$pid"
wait $pid; rc=$?
rm -f "$W/logs/run/$pid"
[ $rc -eq 124 ] || [ $rc -eq 137 ] && echo "[run.sh] TIMEOUT/KILL rc=$rc after budget ${budget}s" >>"$log"
echo "[run.sh $(date '+%F %T')] rc=$rc tag=$tag" >>"$log"
exit $rc
