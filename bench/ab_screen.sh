#!/bin/bash
# A/B screening of two server configs (JSON in configs/) in one GPU session: each config starts its own server via
# bench/screen.sh with the config's env, same CTXS/RUNS/MODES. Usage: CTXS=.. RUNS=.. MODES=.. bench/ab_screen.sh <exp> <cfgA.json> <cfgB.json> [extra VAR=val for B]
W="$(cd "$(dirname "$0")/.." && pwd)"; cd "$W"
exp=$1 a=$2 b=$3; shift 3
envargs() { ${PYTHON:-runtimes/venv-mlx/bin/python} -c "import json,sys; e=json.load(open(sys.argv[1]))['env']; e.pop('OMLX_EXTRA',None); print('\n'.join(f'{k}={v}' for k,v in e.items()))" "$1"; }
extra() { ${PYTHON:-runtimes/venv-mlx/bin/python} -c "import json,sys; print(json.load(open(sys.argv[1]))['env'].get('OMLX_EXTRA',''))" "$1"; }
i=0
for cfg in "$a" "$b"; do
  tag=$([ $i = 0 ] && echo A || echo B); i=$((i+1))
  args=(); while IFS= read -r l; do [ -n "$l" ] && args+=("$l"); done < <(envargs "$cfg")
  [ "$tag" = B ] && args+=("$@")
  echo "=== ${exp}${tag} $(basename "$cfg") ${*:+(+ $*)} $(date +%T)"
  OMLX_EXTRA="$(extra "$cfg")" bench/screen.sh "${exp}${tag}" "$(basename "$cfg" .json)" "${args[@]}"
done
echo "=== DONE $(date +%T)"
