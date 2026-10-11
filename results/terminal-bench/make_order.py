#!/usr/bin/env python3
"""Pre-registration of the Terminal-Bench 2.1 subset order. Deterministic: sorted pool, random.Random(SEED).shuffle.
Pool = the 89 Terminal-Bench 2.1 tasks minus 17 tasks excluded before any model call (RAM >= 8 GB, 4 CPUs, image > 1.4 GB, qemu).
Usage: python3 make_order.py <terminal-bench-2.1 tasks dir>"""
import os, random, sys
SEED = 20261010
EXCL = """caffe-cifar-10 filter-js-from-html gpt2-codegolf hf-model-inference install-windows-3.11 make-doom-for-mips
make-mips-interpreter mcmc-sampling-stan mteb-leaderboard mteb-retrieve pytorch-model-recovery qemu-alpine-ssh qemu-startup
reshard-c4-data rstan-to-pystan torch-pipeline-parallelism torch-tensor-parallelism""".split()
T = sys.argv[1]
tasks = sorted(d for d in os.listdir(T) if os.path.isfile(os.path.join(T, d, "task.toml")))
assert len(tasks) == 89 and all(e in tasks for e in EXCL) and len(EXCL) == 17, (len(tasks), len(EXCL))
pool = [t for t in tasks if t not in EXCL]
random.Random(SEED).shuffle(pool)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb21_order.txt")
open(out, "w").write("\n".join(pool) + "\n")
print(len(pool), "tasks ->", out)
