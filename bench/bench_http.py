"""Speed matrix client for any OpenAI-compatible server (oMLX, llama-server, mlx_vlm.server).
Prompts: bench/prompts.py (real code + real task, template applied, exact ctx in tokens; sent as text to /v1/completions).
Per run: TTFT, decode tok/s = (N-1)/(t_last-t_first) from streamed chunks, completion tokens from usage.
Writes one RESULTS.csv row per run immediately (a killed suite keeps what it measured).
Usage: python bench/bench_http.py --url http://127.0.0.1:8000 --model M --ctx 2048 32768 --runs 3 --modes greedy default \
         --exp EXP-004 --variant Q6-oQ6e --runtime omlx-main --flags "..." --mtp off [--server-pid PID] [--save out.json]"""
import argparse, json, os, sys, time, csv, subprocess, statistics, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = "exp_id,timestamp,variant,runtime,commit,flags,mtp,spec_method,ctx,prompt_tok,gen_tok,prefill_tok_s,ttft_s,decode_tok_s,accept_rate,rss_gb,gpu_mem_gb,swap_gb,thermal,quality_score,notes".split(",")
SAMPLING = {"greedy": dict(temperature=0.0, top_p=1.0, top_k=1, min_p=0.0),
            "default": dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)}


def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""


def snapshot(pid):
    rss = sh(f"ps -o rss= -p {pid}") if pid else ""
    wired = sh("vm_stat | awk '/wired down/{gsub(/\\./,\"\",$4); print $4*16384/1e9}'")
    swap = sh("sysctl -n vm.swapusage | tr ',' '.' | awk '{for(i=1;i<=NF;i++) if($i==\"used\") print $(i+2)}'")
    therm = sh("pmset -g therm | grep -i 'speed_limit' | awk '{print $NF}'") or "nominal"
    return (f"{int(rss)/1e6:.2f}" if rss.isdigit() else ""), swap.rstrip("MG"), therm, (f"{float(wired.replace(',', '.')):.2f}" if wired else "")


def run_one(url, model, text, gen, samp, seed, extra, tok=None):
    body = dict(model=model, prompt=text, max_tokens=gen, stream=True, seed=seed,
                stream_options={"include_usage": True}, **SAMPLING[samp], **extra)
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); stamps = []; out = []; usage = {}; timings = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: j = json.loads(d)
            except json.JSONDecodeError: continue
            if j.get("usage"): usage = j["usage"]
            if j.get("timings"): timings = j["timings"]
            for c in j.get("choices", []):
                tx = c.get("text") or ""
                if tx:
                    stamps.append(time.perf_counter()); out.append(tx)
    n = usage.get("completion_tokens") or len(stamps)
    ttft = stamps[0] - t0 if stamps else float("nan")
    # servers may batch several tokens per SSE chunk: tokens already emitted at the first chunk are excluded
    n_first = 1
    if tok is not None and out:
        est_total = max(1, len(tok.encode("".join(out), add_special_tokens=False)))
        n_first = max(1, round(len(tok.encode(out[0], add_special_tokens=False)) * n / est_total))
    dec = (n - n_first) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else float("nan")
    return dict(text="".join(out), n=n, n_first=n_first, prompt_tokens=usage.get("prompt_tokens"), ttft=ttft, decode=dec,
                chunks=len(stamps), timings=timings, usage=usage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048]); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gen", type=int, default=512); ap.add_argument("--modes", nargs="+", default=["greedy", "default"])
    ap.add_argument("--exp", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--runtime", required=True)
    ap.add_argument("--commit", default=""); ap.add_argument("--flags", default=""); ap.add_argument("--mtp", default="off")
    ap.add_argument("--spec", default="none"); ap.add_argument("--server-pid", type=int, default=0)
    ap.add_argument("--extra", default="{}", help="extra JSON fields for the request body")
    ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--save", default=None)
    ap.add_argument("--results", default=os.path.join(W, "RESULTS.csv")); ap.add_argument("--notes", default="")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    tok = prompts.tokenizer(); saved = {}
    if a.warmup:
        run_one(a.url, a.model, tok.decode(prompts.prompt_ids(2048, 2)[:1024]), 32, "greedy", 0, extra)
    for ctx in a.ctx:
        for mode in a.modes:
            decs = []
            for r in range(a.runs):
                text = tok.decode(prompts.prompt_ids(ctx, r))
                res = run_one(a.url, a.model, text, a.gen, mode, 1000 + r, extra, tok)
                rss, swap, therm, wired = snapshot(a.server_pid)
                pt = res["prompt_tokens"] or ctx
                acc = ""
                t = res["timings"]
                if t.get("draft_n"): acc = f"{t.get('draft_n_accepted',0)}/{t['draft_n']}"
                row = dict(exp_id=a.exp, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), variant=a.variant, runtime=a.runtime,
                           commit=a.commit, flags=a.flags + f" mode={mode} task={r}", mtp=a.mtp, spec_method=a.spec, ctx=ctx,
                           prompt_tok=pt, gen_tok=res["n"], prefill_tok_s=f"{pt/res['ttft']:.0f}", ttft_s=f"{res['ttft']:.2f}",
                           decode_tok_s=f"{res['decode']:.2f}", accept_rate=acc, rss_gb=rss, gpu_mem_gb=wired, swap_gb=swap,
                           thermal=therm, quality_score="", notes=(a.notes + (f" SHORT_GEN" if res["n"] < a.gen else "")).strip())
                with open(a.results, "a", newline="") as f: csv.DictWriter(f, COLS).writerow(row)
                decs.append(res["decode"])
                saved[f"{ctx}/{mode}/{r}"] = res["text"]
                print(f"ctx {ctx} {mode} run{r}: prompt {pt} gen {res['n']} ttft {res['ttft']:.2f}s prefill {pt/res['ttft']:.0f} t/s "
                      f"decode {res['decode']:.2f} t/s chunks {res['chunks']} first {res['n_first']} {acc} wired {wired} swap {swap}", flush=True)
            print(f"== ctx {ctx} {mode}: median {statistics.median(decs):.2f} min {min(decs):.2f} tok/s", flush=True)
    if a.save:
        json.dump(saved, open(a.save, "w"))


if __name__ == "__main__":
    main()
