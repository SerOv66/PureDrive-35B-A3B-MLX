#!/bin/bash
# One harness x task run: reset, non-interactive launch with a time cap, check, metadata. The proxy (port 8002) must be running.
# Usage: run_one.sh <pi|hermes> <task> <rep> [cap_s=900] [prompt_file]   -> harness/results/<id>/{meta.json,agent.out,proxy.jsonl,check.out,diff.txt}
# RUN_TAG=<s> appends a suffix to the run id (the long-session runs used ctx96k / ctx256k with CTXWIN=...).
R="$(cd "$(dirname "$0")" && pwd)"; H=$1; T=$2; REP=$3; CAP=${4:-900}; PF=${5:-$R/tasks/$T/prompt.txt}
ID="$(date +%m%d-%H%M%S)-$H-$T-r$REP${RUN_TAG:+-$RUN_TAG}"; O="$R/results/$ID"; mkdir -p "$O"
WD="$("$R/tasks/$T/reset.sh")" || exit 1
curl -s "localhost:8002/__log?file=$O/proxy.jsonl" >/dev/null || { echo "proxy is down"; exit 2; }
S0=$(date '+%F %T'); t0=$(date +%s.%N)
gtimeout -k 20 "$CAP" "$R/launch.sh" "$H" "$WD" "$PF" 8002 > "$O/agent.out" 2>&1 < /dev/null; rc=$?
t1=$(date +%s.%N); S1=$(date '+%F %T'); curl -s "localhost:8002/__log?file=/dev/null" >/dev/null
"$R/tasks/$T/check.sh" "$WD" > "$O/check.out" 2>&1; crc=$?
(cd "$WD" && git add -A && git diff --cached --stat > "$O/diff.txt" 2>&1)
python3 -c "import json,sys; print(json.dumps(dict(id='$ID',harness='$H',task='$T',rep=$REP,cap_s=$CAP,agent_rc=$rc,timeout=($rc==124 or $rc==137),wall_s=round($t1-$t0,1),check_rc=$crc,success=($crc==0),start='$S0',end='$S1')))" > "$O/meta.json"
cat "$O/meta.json"
