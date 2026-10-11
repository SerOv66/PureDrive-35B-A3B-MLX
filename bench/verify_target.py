"""FROZEN VERIFIER — the benchmark harness behind the certified numbers. Published copy: the original is sha256-pinned (see docs/METHODOLOGY.md);
this copy differs from it only in translated messages/comments and the lines marked "PUBLISH" (configurable interpreter/powermetrics command, optional speed threshold).
Usage: python bench/verify_target.py configs/<candidate>.json [--speed-target <tok/s>]
  --speed-target X (or env SPEED_TARGET=X): optional per-cell decode threshold in tok/s; default OFF (measurements only).
Config JSON: {"name","variant","runtime","cmd":[...],"env":{...},"baseline_env":{...}|null,"url","model","mtp","spec","quality_profile","identity_env_override"}
  baseline_env = same server without speculation (for the token-identity check); null if the config has no speculation.
Steps (every step is a separate subprocess, <=10 min each; total <=45 min):
  0. integrity: FROZEN.sha256 must verify.
  1. speed matrix 6.1: ctx 2K,4K,16K,32K,40K,64K,128K x 3 runs x {greedy, default sampling} x 512 generated tokens,
     real fixed prompts (bench/prompts.py), KV really filled (exact-length prompts, prefix cache allowed = real tokens).
  2. constancy: 1 run at 131072 generating 2048 tokens (greedy).
  3. stability: after >=10 min of continuous generation, ctx 2048 x 3 runs x both modes again; swap growth < 1 GB;
     thermal pressure (powermetrics) Nominal/Fair.
  4. quality quick (quality/run_quality.py parts, compared by quality/compare.py with quality/THRESHOLDS.md).
  5. identity: greedy outputs of the candidate == the same server without speculation (if baseline_env):
     6 quick prompts (<=2K) + 1 bench prompt at 32K, both servers without prefix cache.
Verdict (written to logs/verify/<ts>.json and logs/verify/LATEST): PASS only if integrity holds, every cell has >= 3 runs and every run produced >= 512
tokens, stability (swap, thermal) holds, quality PASS and identity PASS. If a speed threshold is set, every median decode (every ctx, both modes,
constancy, stability) must also reach it. Without a threshold the speed medians are reported but never affect the verdict."""
import json, os, subprocess, sys, time, csv, statistics, urllib.request

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("PYTHON") or os.path.join(W, "runtimes", "venv-mlx", "bin", "python")  # PUBLISH: configurable interpreter
CTXS = [2048, 4096, 16384, 32768, 40960, 65536, 131072]
TARGET = float(os.environ["SPEED_TARGET"]) if os.environ.get("SPEED_TARGET") else None  # PUBLISH: optional speed threshold (tok/s), default OFF; --speed-target overrides
TS = time.strftime("%Y%m%d-%H%M%S")
VD = os.path.join(W, "logs", "verify"); os.makedirs(VD, exist_ok=True)
LOG = os.path.join(VD, f"{TS}.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def sh(args, budget=600, env=None):
    """One step = one subprocess under gtimeout -k 30 <budget> (hard per-step time budget)."""
    r = subprocess.run(["gtimeout", "-k", "30", str(budget)] + args, cwd=W, env=env, capture_output=True, text=True)
    open(LOG, "a").write(r.stdout[-20000:] + r.stderr[-5000:])
    return r.returncode, r.stdout


def swap_mb():
    out = subprocess.run("sysctl -n vm.swapusage", shell=True, capture_output=True, text=True).stdout.replace(",", ".")
    p = out.split(); i = p.index("used")
    v = p[i + 2]; return float(v[:-1]) * (1024 if v.endswith("G") else 1)


def thermal():
    """Thermal pressure level from powermetrics; '' if unavailable. PUBLISH: the command is configurable via POWERMETRICS_CMD
    (default: passwordless `sudo -n powermetrics --samplers thermal -n 1 -i 2000`)."""
    cmd = os.environ.get("POWERMETRICS_CMD", "sudo -n powermetrics --samplers thermal -n 1 -i 2000").split()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    lv = [l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if "Current pressure level" in l]
    return lv[-1] if lv else ""


def server(cfg, env_over, tag):
    env = dict(os.environ, **env_over)
    subprocess.Popen([os.path.join(W, "scripts", "run.sh"), tag, "3000", os.path.join(VD, f"{TS}_{tag}.log"), "--"] + cfg["cmd"],
                     cwd=W, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(120):
        try:
            urllib.request.urlopen(cfg["url"] + "/v1/models", timeout=2); return True
        except Exception:
            time.sleep(2)
    return False


def stop(tag):
    """Stop the server and wait until its memory is really released before the next server loads (no double residency)."""
    subprocess.run([os.path.join(W, "scripts", "kill_run.sh"), tag], cwd=W, capture_output=True)
    for _ in range(60):
        if subprocess.run(["pgrep", "-f", "omlx-server|llama-server"], capture_output=True).returncode != 0: break
        time.sleep(1)
    time.sleep(5)


def speed(cfg, ctxs, runs, gen, modes, csvf, warm):
    args = [PY, "bench/bench_http.py", "--url", cfg["url"], "--model", cfg["model"], "--ctx", *map(str, ctxs), "--runs", str(runs),
            "--gen", str(gen), "--modes", *modes, "--warmup", str(int(warm)), "--exp", f"VERIFY-{TS}", "--variant", cfg["variant"],
            "--runtime", cfg["runtime"], "--mtp", cfg.get("mtp", "off"), "--spec", cfg.get("spec", "none"),
            "--flags", f"verify {cfg['name']}", "--results", csvf]
    return sh(args)[0]


def main():
    global TARGET
    import argparse
    ap = argparse.ArgumentParser(description="Frozen benchmark harness: speed matrix, constancy, stability, quality and identity checks.")
    ap.add_argument("config", help="candidate configuration JSON (configs/<candidate>.json)")
    ap.add_argument("--speed-target", type=float, default=None, metavar="TOK_S",
                    help="optional per-cell decode threshold in tok/s (default: off; env SPEED_TARGET also works)")
    args = ap.parse_args()
    if args.speed_target is not None: TARGET = args.speed_target
    cfg = json.load(open(args.config))
    res = {"ts": TS, "config": cfg, "fail": [], "speed": {}, "quality": {}, "identity": None}
    fail = res["fail"]
    rc, out = sh(["shasum", "-a", "256", "-c", "FROZEN.sha256"], 60)
    if rc != 0: fail.append("integrity: FROZEN.sha256 does not verify")
    # a speculative config must be checked for losslessness; the override may only disable the prefix cache
    if (cfg.get("mtp", "off") != "off" or cfg.get("spec", "none") != "none") and cfg.get("baseline_env") is None:
        fail.append("speculative config without baseline_env: lossless identity cannot be checked")
    if cfg.get("identity_env_override", {}) not in ({}, {"CACHE": "0"}):
        fail.append("identity_env_override allowed only as {} or {\"CACHE\": \"0\"}")
    t0 = time.time(); s0 = swap_mb(); res["thermal_start"] = thermal()
    hdr = "exp_id,timestamp,variant,runtime,commit,flags,mtp,spec_method,ctx,prompt_tok,gen_tok,prefill_tok_s,ttft_s,decode_tok_s,accept_rate,rss_gb,gpu_mem_gb,swap_gb,thermal,quality_score,notes\n"
    csvs = {k: os.path.join(VD, f"{TS}_{k}.csv") for k in ("matrix", "constancy", "stability")}
    for f in csvs.values(): open(f, "w").write(hdr)
    try:
        urllib.request.urlopen(cfg["url"] + "/v1/models", timeout=2)
        fail.append("port already in use by another server: stop it before running the verifier"); return finish(res)
    except Exception:
        pass
    if not server(cfg, cfg.get("env", {}), "server-verify"):
        fail.append("candidate server did not start"); return finish(res)
    for i, ctx in enumerate(CTXS):
        log(f"speed ctx {ctx}")
        if speed(cfg, [ctx], 3, 512, ["greedy", "default"], csvs["matrix"], i == 0) != 0: fail.append(f"speed ctx {ctx}: run failed/TIMEOUT")
    log("constancy 128K x 2048 tok")
    if speed(cfg, [131072], 1, 2048, ["greedy"], csvs["constancy"], False) != 0: fail.append("constancy: run failed/TIMEOUT")
    # >= 10 min of CONTINUOUS generation before the stability point (not idle waiting)
    loadf = os.path.join(VD, f"{TS}_load.csv"); open(loadf, "w").write(hdr)
    while time.time() - t0 < 600:
        log("load: continuous generation ctx 2048 x 1024 tok")
        if speed(cfg, [2048], 1, 1024, ["greedy"], loadf, False) != 0:
            fail.append("load: continuous generation failed/TIMEOUT (stability invalid)"); break
    log("stability ctx 2048 after load")
    if speed(cfg, [2048], 3, 512, ["greedy", "default"], csvs["stability"], False) != 0: fail.append("stability: run failed/TIMEOUT")
    for step, f in csvs.items():
        groups = {}
        for r in csv.DictReader(open(f)):
            mode = r["flags"].split("mode=")[1].split()[0]
            groups.setdefault((int(r["ctx"]), mode), []).append(float(r["decode_tok_s"]))
            need = 2048 if step == "constancy" else 512
            if int(r["gen_tok"]) < need: fail.append(f"{step} ctx {r['ctx']}: generated {r['gen_tok']} < {need} tokens")
        for (ctx, mode), v in sorted(groups.items()):
            name = f"{ctx}/{mode}" if step == "matrix" else f"{step}/{ctx}/{mode}"
            res["speed"][name] = statistics.median(v)
            if step == "matrix" and len(v) < 3: fail.append(f"{name}: only {len(v)} runs")
    if TARGET is not None:
        for name, med in res["speed"].items():
            if not med >= TARGET: fail.append(f"{name}: median {med:.1f} < {TARGET:.0f} tok/s")
    for mode in ("greedy", "default"):
        for ctx in CTXS:
            if f"{ctx}/{mode}" not in res["speed"]: fail.append(f"{ctx}/{mode}: no measurement")
    ds = swap_mb() - s0
    if ds > 1024: fail.append(f"swap grew by {ds:.0f} MB")
    th = thermal()
    if th.lower() not in ("nominal", "fair"): fail.append(f"thermal: pressure level '{th or 'unreadable'}' after the load")
    res["swap_delta_mb"] = ds; res["thermal"] = th
    # quality quick: candidate vs frozen reference (variant on its reference runtime, no experimental
    # optimizations) with the config's profile ("numeric" if kernels/numerics differ, "lossless" otherwise)
    ref = os.path.join(W, "quality", "reference", f"{cfg['variant']}_quick.json")
    qout = os.path.join(VD, f"{TS}_quality.json")
    for part in ("coding", "reasoning", "tools"):
        log(f"quality {part}")
        rc, _ = sh([PY, "quality/run_quality.py", "--url", cfg["url"], "--model", cfg["model"], "--part", part, "--suite", "quick", "--out", qout])
        if rc != 0: fail.append(f"quality {part}: run failed/TIMEOUT")
    # needle: one step (<=10 min) per context, items merged into the candidate's needle part
    items = []
    for nc in (32768, 65536, 131072):
        log(f"quality needle ctx {nc}")
        nf = qout.replace(".json", f"_needle{nc}.json")
        rc, _ = sh([PY, "quality/run_quality.py", "--url", cfg["url"], "--model", cfg["model"], "--part", "needle", "--suite", "quick",
                    "--needle-ctxs", str(nc), "--out", nf])
        if rc != 0: fail.append(f"quality needle {nc}: run failed/TIMEOUT")
        try: items += json.load(open(nf))["parts"]["needle"]["items"]
        except Exception: pass
    try:
        q = json.load(open(qout))
        q["parts"]["needle"] = {"score": sum(i["passed"] for i in items) / max(1, len(items)), "passed": sum(i["passed"] for i in items),
                                "n": len(items), "seconds": sum(i.get("seconds") or 0 for i in items), "items": items}
        json.dump(q, open(qout, "w"), indent=1)
    except Exception as e:
        fail.append(f"quality needle merge: {e}")
    stop("server-verify")
    # lossless: greedy identity of the candidate vs the same server without speculation, both without prefix cache
    # (a shared cache could mix KV blocks computed under different configs); identity_env_override e.g. {"CACHE": "0"}
    over = cfg.get("identity_env_override", {})
    for tag, env, outp in (("server-verify-id", cfg.get("env", {}), qout),
                           ("server-verify-base", cfg.get("baseline_env"), os.path.join(VD, f"{TS}_identity_baseline.json"))):
        if env is None: continue
        log(f"identity {tag}")
        if server(cfg, dict(env, **over), tag):
            rc, _ = sh([PY, "quality/run_quality.py", "--url", cfg["url"], "--model", cfg["model"], "--part", "identity", "--suite", "quick", "--out", outp])
            if rc != 0: fail.append(f"identity {tag}: run failed/TIMEOUT")
            rc, _ = sh([PY, "bench/bench_http.py", "--url", cfg["url"], "--model", cfg["model"], "--ctx", "32768", "--runs", "1", "--modes", "greedy",
                        "--gen", "256", "--warmup", "0", "--exp", f"VERIFY-{TS}-id", "--variant", cfg["variant"], "--runtime", cfg["runtime"],
                        "--results", os.devnull, "--save", outp.replace(".json", "_32k.json")])
            if rc != 0: fail.append(f"identity 32K {tag}: run failed/TIMEOUT")
        else:
            fail.append(f"identity {tag}: server did not start")
        stop(tag)
    rc, out = sh([PY, "quality/compare.py", ref, qout, "--profile", cfg.get("quality_profile", "numeric")], 120)
    res["quality"] = out.strip()[-1500:]
    if rc != 0: fail.append(f"quality gate FAIL vs {os.path.basename(ref)} (profile {cfg.get('quality_profile', 'numeric')})")
    if cfg.get("baseline_env") is not None:
        rc, out = sh([PY, "quality/compare.py", os.path.join(VD, f"{TS}_identity_baseline.json"), qout, "--profile", "lossless"], 120)
        res["identity"] = out.strip()[-1500:]
        idl = [l for l in out.splitlines() if "identity" in l.lower()]
        if rc != 0 or not any("PASS" in l for l in idl): fail.append("identity: greedy output differs from the baseline without speculation")
        try:
            a32 = json.load(open(qout.replace(".json", "_32k.json"))); b32 = json.load(open(os.path.join(VD, f"{TS}_identity_baseline_32k.json")))
            res["identity_32k"] = "IDENTICAL" if a32 == b32 and a32 and all(a32.values()) else "DIVERGE/EMPTY"
        except Exception as e:
            res["identity_32k"] = f"missing: {e}"
        if res["identity_32k"] != "IDENTICAL": fail.append(f"identity 32K: {res['identity_32k']}")
    res["minutes"] = (time.time() - t0) / 60
    if res["minutes"] > 45: fail.append(f"verifier over budget: {res['minutes']:.1f} min")
    finish(res)


def finish(res):
    verdict = "PASS" if not res["fail"] else "FAIL"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(VD, f"{TS}.json"), "w"), indent=1)
    summary = f"{verdict} {TS} {res['config']['name']}: " + ("all criteria met" if verdict == "PASS" else "; ".join(res["fail"][:12]))
    open(os.path.join(VD, "LATEST"), "w").write(summary + "\n")
    log(summary)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
