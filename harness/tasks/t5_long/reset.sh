#!/bin/bash
# Recreates a clean harness/work/t5_long (seed/ + lib/ built by build_seed.sh), git init + initial commit. Prints the work dir.
T="$(cd "$(dirname "$0")" && pwd)"; [ -d "$T/seed/lib" ] || "$T/build_seed.sh" || exit 1; W="$T/../../work"; mkdir -p "$W"; W="$(cd "$W" && pwd)"
D="$W/$(basename "$T")"; [ -d "$D" ] && rm -rf "$D"
mkdir -p "$D" && cp -R "$T/seed/." "$D/" && cd "$D" && git init -q && git add -A && git -c user.name=h -c user.email=h@h commit -qm seed && echo "$D"
