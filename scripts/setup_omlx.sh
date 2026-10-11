#!/bin/bash
# Build the certified runtime: oMLX at commit cc1fdc9 + the certified patch + the custom Metal attention kernel.
# Run from the repository root. Needs git, uv (https://github.com/astral-sh/uv) and Python 3.12.
set -euo pipefail
PATCH=${PATCH:-patches/omlx_c35_EXP110.patch}
KERNEL=${KERNEL:-kernels/gqa_attn.py}
mkdir -p runtimes bench/kernels
[ -d runtimes/omlx ] || git clone https://github.com/jundot/omlx runtimes/omlx
# reset to the pinned commit first, so that re-running this script re-applies the patch cleanly
git -C runtimes/omlx reset -q --hard cc1fdc9 && git -C runtimes/omlx clean -fdq
git -C runtimes/omlx apply --check "../../$PATCH"
git -C runtimes/omlx apply "../../$PATCH"
# the patch looks for the kernel at <repo>/bench/kernels/gqa_attn.py (4 directories above omlx/patches/)
cp "$KERNEL" bench/kernels/gqa_attn.py
sha256sum bench/kernels/gqa_attn.py 2>/dev/null || shasum -a 256 bench/kernels/gqa_attn.py
[ -x runtimes/venv-omlx/bin/python ] || uv venv --python 3.12 runtimes/venv-omlx
uv pip install --python runtimes/venv-omlx/bin/python -e runtimes/omlx
echo "done: runtime in runtimes/omlx, interpreter runtimes/venv-omlx/bin/python"
