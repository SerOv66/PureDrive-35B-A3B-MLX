#!/bin/bash
# Pin the verifier, the benchmark prompts and the quality suite with sha256 (FROZEN.sha256) and make them read-only.
# bench/verify_target.py refuses to run if `shasum -c FROZEN.sha256` fails. Run once from the repository root, after
# `python bench/corpus/build_corpus.py` and `python quality/run_quality.py --fetch-data` have produced the corpus and the datasets. The hashes of the ORIGINAL
# (pre-publication) files are in results/FROZEN_original_sha256.txt for reference; the published copies are sanitized
# (paths, translated messages), so their hashes differ.
set -euo pipefail
cd "${REPO_DIR:-$PWD}"
FILES="bench/verify_target.py bench/bench_http.py bench/prompts.py bench/corpus/corpus.txt bench/corpus/tasks.json
quality/run_quality.py quality/compare.py quality/THRESHOLDS.md quality/reference/Q6-oQ6e_quick.json quality/data/SHA256SUMS"
shasum -a 256 $FILES > FROZEN.sha256
chmod a-w $FILES FROZEN.sha256
shasum -a 256 -c FROZEN.sha256
