# per-run ms/cycle of an A/B screen (same pairing as ab_mscyc.py): ctx mode run tok/s tok/cyc cycles ms/cyc
import re, subprocess, sys
W = __import__("os").environ.get("REPO_DIR", ".")
for exp in sys.argv[1:]:
    for s in "AB":
        runs = [(int(m[1]), m[2], float(m[3])) for m in re.finditer(r"^ctx (\d+) (\w+) run\d+: .*? decode ([\d.]+) t/s", open(f"{W}/logs/{exp}{s}_screen.log").read(), re.M)]
        out = subprocess.run([f"{W}/runtimes/venv-mlx/bin/python", f"{W}/scripts/mtp_stats.py", f"{W}/logs/{exp}{s}_server.log"], capture_output=True, text=True).stdout
        tc = [(int(m[1]), int(m[2]), int(m[3])) for m in re.finditer(r"^uid\s+(\d+) tok\s+(\d+) cyc\s+(\d+)", out, re.M)][-len(runs):]
        print(f"{exp}{s}: " + "  ".join(f"{c//1024}K r{i%3} {t:.1f}t/s {tok/cyc:.2f}t/c {cyc}cyc {1000*tok/cyc/t:.2f}ms" for i, ((c, m, t), (u, tok, cyc)) in enumerate(zip(runs, tc))))
