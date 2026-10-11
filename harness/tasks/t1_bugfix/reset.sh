#!/bin/bash
# Recreates a clean harness/work/<task>: copy seed/, git init + initial commit. Prints the work dir.
T="$(cd "$(dirname "$0")" && pwd)"; W="$(cd "$T/../../work" 2>/dev/null && pwd || (mkdir -p "$T/../../work" && cd "$T/../../work" && pwd))"
D="$W/$(basename "$T")"; [ -d "$D" ] && rm -rf "$D"
mkdir -p "$D" && cp -R "$T/seed/." "$D/" && cd "$D" && git init -q && git add -A && git -c user.name=h -c user.email=h@h commit -qm seed && echo "$D"
