#!/bin/bash
# Stop registered runs (logs/run/*) whose tag starts with $1: TERM the process group, KILL after 15 s.
W="$(cd "$(dirname "$0")/.." && pwd)"
for f in "$W"/logs/run/*; do
  [ -f "$f" ] || continue; read -r pid _ _ tag _ < "$f"
  [[ "$tag" == $1* ]] || continue
  kill -TERM -- -"$pid" 2>/dev/null; for i in $(seq 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$pid" 2>/dev/null; echo "stopped $tag pid $pid"
done
