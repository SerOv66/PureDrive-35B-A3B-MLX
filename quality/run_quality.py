#!/usr/bin/env python3
"""Quality suite  for any OpenAI-compatible server (oMLX, llama-server).

Parts: coding (HumanEval+/MBPP+), reasoning (GSM8K), tools (tool calling), needle (long-context retrieval),
identity (token-exactness of speculative decoding vs a reference run).  Greedy everywhere (temperature 0, top_k 1,
fixed seed); thinking OFF via chat_template_kwargs for coding/reasoning/tools.  Stdlib only (+ bench/prompts.py,
i.e. transformers, for needle/identity prompts).

Usage: run_quality.py --url http://127.0.0.1:8000 --model NAME --part {coding,reasoning,tools,needle,identity,all}
                      --suite {quick,full} --out quality/results/X.json [--ref ref.json]
Only talks to the server given by --url.  Never loads a model.
"""
import argparse, concurrent.futures as cf, datetime, gzip, hashlib, json, os, platform, random, re, shutil
import subprocess, sys, tempfile, time, urllib.error, urllib.request
from decimal import Decimal, InvalidOperation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
SEED = 1234
GREEDY = dict(temperature=0, top_p=1.0, top_k=1, seed=SEED)
NO_THINK = {"enable_thinking": False}  # honoured by oMLX (chat_template_kwargs) and llama-server (--jinja)
DATA_FILES = {  # name -> url (sha256 pinned in data/SHA256SUMS)
    "HumanEvalPlus.jsonl.gz": "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
    "MbppPlus.jsonl.gz": "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}
# quick-suite subsets (deterministic; documented in README)
HE_STEP, MBPP_STEP, GSM_QUICK, GSM_FULL = 6, 13, 30, 200
MAXTOK = dict(humaneval=512, mbpp=320, gsm8k=400, tools=200, needle=64, identity=256)
NEEDLE_CTXS, NEEDLE_DEPTHS_QUICK, NEEDLE_DEPTHS_FULL = [32768, 65536, 131072], [0.5], [0.1, 0.5, 0.9]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- data
def fetch_data():
    """Download missing dataset files; write SHA256SUMS on first download, otherwise verify against it."""
    os.makedirs(DATA, exist_ok=True)
    for name, url in DATA_FILES.items():
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            log("download", url)
            urllib.request.urlretrieve(url, p)
    sums = os.path.join(DATA, "SHA256SUMS")
    if not os.path.exists(sums):
        with open(sums, "w") as f:
            for name in DATA_FILES:
                f.write(f"{sha256_file(os.path.join(DATA, name))}  {name}\n")
    return verify_data()


def verify_data():
    """Return {file: sha256}; raise if a dataset file differs from the pinned SHA256SUMS."""
    pinned = {}
    with open(os.path.join(DATA, "SHA256SUMS")) as f:
        for line in f:
            if line.strip():
                h, n = line.split()
                pinned[n] = h
    out = {}
    for name in DATA_FILES:
        out[name] = sha256_file(os.path.join(DATA, name))
        if pinned.get(name) != out[name]:
            raise RuntimeError(f"dataset {name} sha256 {out[name]} != pinned {pinned.get(name)}")
    return out


EXCLUDE = {  # tasks whose canonical solution cannot run safely inside the sandbox limits (model resident => RAM is tight)
    "Mbpp/255": "reference alone peaks at ~1.5 GB RSS (combinations_with_replacement on large inputs)",
}


def load_plus(name):
    """{task_id: task} sorted by numeric id, minus EXCLUDE."""
    with gzip.open(os.path.join(DATA, name), "rt") as f:
        rows = [json.loads(l) for l in f]
    rows = [r for r in rows if r["task_id"] not in EXCLUDE]
    rows.sort(key=lambda r: int(r["task_id"].split("/")[-1]))
    return {r["task_id"]: r for r in rows}


def load_gsm8k():
    with open(os.path.join(DATA, "gsm8k_test.jsonl")) as f:
        return [json.loads(l) for l in f]


def subset_ids(ids, step, suite):
    return list(ids) if suite == "full" else list(ids)[::step]


def gsm_indices(n_total, suite):
    n = GSM_FULL if suite == "full" else GSM_QUICK
    return [int(i * n_total / n) for i in range(n)]


# ----------------------------------------------------------------------------------------------- http
class Client:
    def __init__(self, url, model, timeout=900):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def _req(self, path, body=None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} on {path}: {e.read().decode(errors='replace')[:500]}")

    def info(self):
        """Best-effort lightweight GETs (model list, llama.cpp /props, /health), truncated."""
        out = {}
        for p in ("/v1/models", "/props", "/health"):
            try:
                out[p] = self._req(p, timeout=10)
            except Exception as e:
                out[p] = f"unavailable: {type(e).__name__}"
        s = json.dumps(out)
        return out if len(s) < 6000 else {"truncated": s[:6000]}

    def chat(self, messages, max_tokens, tools=None, timeout=None):
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, stream=False,
                    chat_template_kwargs=NO_THINK, **GREEDY)
        if tools:
            body["tools"], body["tool_choice"], body["parallel_tool_calls"] = tools, "auto", True  # oMLX ignores the last
        t0 = time.time()
        j = self._req("/v1/chat/completions", body, timeout)
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        return dict(content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
                    tool_calls=msg.get("tool_calls") or [], finish=ch.get("finish_reason"),
                    usage=j.get("usage") or {}, seconds=round(time.time() - t0, 2))

    def complete(self, prompt, max_tokens, timeout=None):
        body = dict(model=self.model, prompt=prompt, max_tokens=max_tokens, stream=False, **GREEDY)
        t0 = time.time()
        j = self._req("/v1/completions", body, timeout)
        ch = j["choices"][0]
        return dict(text=ch.get("text") or "", finish=ch.get("finish_reason"), usage=j.get("usage") or {},
                    seconds=round(time.time() - t0, 2))


def strip_think(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return s.split("</think>")[-1]  # unmatched closing tag (template already opened <think>)


# ----------------------------------------------------------------------------------------------- sandbox
# Worker executed as `python -I -B worker.py spec.json result.json` inside a temp dir (cwd), under sandbox-exec
# (no network, writes only in the temp dir) when available.  It first runs the canonical solution on every input
# to get the expected outputs (EvalPlus differential testing), then the candidate with a per-input time limit
# max(1 s, 4 x canonical time) and 10 s in total.  Comparison semantics follow EvalPlus 0.3.1 (Apache-2.0):
# exact match, float atol, special oracles, MBPP input deserialisation.  NOT a hardened sandbox against hostile code.
WORKER = r'''
import sys, json, time, copy, math, signal, os, socket, subprocess
spec = json.load(open(sys.argv[1])); OUT = sys.argv[2]
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (int(spec["cpu_limit"]),) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 24, 1 << 24))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except Exception:
    pass  # memory is enforced by the parent RSS watchdog (RLIMIT_AS is not enforceable on macOS)
RES = {"status": "error", "msg": ""}

def finish(**kw):
    RES.update(kw)
    with open(OUT, "w") as f:
        json.dump(RES, f)
    os._exit(0)

def _blocked(*a, **k):
    raise OSError("disabled in sandbox")

socket.socket = socket.create_connection = socket.getaddrinfo = socket.gethostbyname = _blocked

TUP1 = {2,116,132,143,222,261,273,394,399,421,424,429,470,560,579,596,616,630,726,740,744,809}
TUP2 = {63,64,70,94,120,237,272,299,400,409,417,438,473,614,780}

def deser(tid, inputs):  # EvalPlus mbpp_deserialize_inputs
    if tid in TUP1: return [[tuple(l) for l in i] for i in inputs]
    if tid in TUP2: return [[[tuple(l) for l in ll] for ll in i] for i in inputs]
    if tid in (75, 413, 444, 753): return [[[tuple(l) for l in i[0]]] + [i[1]] for i in inputs]
    if tid in (106, 750): return [[i[0]] + [tuple(i[1])] for i in inputs]
    if tid == 115: return [[[set(it) if isinstance(it, list) and len(it) else {} for it in i[0]]] for i in inputs]
    if tid == 124: return [(float(i[0]), complex(i[1])) for i in inputs]
    if tid in (250, 405, 446, 617, 720, 763, 808): return [[tuple(i[0])] + [i[1]] for i in inputs]
    if tid in (259, 401, 445):
        m = [[[tuple(l) for l in ll] for ll in i] for i in inputs]
        return [[tuple(l) for l in i] for i in m]
    if tid == 278:
        m = [[[tuple(it) if isinstance(it, list) else it for it in i[0]]] for i in inputs]
        return [[tuple(l) for l in i] for i in m]
    if tid == 307: return [[tuple(i[0])] + [i[1], i[2]] for i in inputs]
    if tid == 722: return [[{k: tuple(v) for k, v in i[0].items()}] + i[1:] for i in inputs]
    if tid == 252: return [[complex(i[0])] for i in inputs]
    if tid in (580, 615, 791):
        def t(x): return tuple(t(y) for y in x) if isinstance(x, list) else x
        return [t(i) for i in inputs]
    return inputs

NOT_NONE = ("check_str", "text_match_three", "text_starta_endb")
SET_EQ = ("similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly", "larg_nnum",
          "intersection_array", "find_dissimilar", "Diff")

def _poly(xs, x): return sum([c * math.pow(x, i) for i, c in enumerate(xs)])

def _surface(b, h):
    sl = math.sqrt((b / 2) ** 2 + h ** 2)
    return round(b ** 2 + 4 * (b * sl) / 2)

def _digit_distance(a, b):
    a, b = str(a), str(b); n = max(len(a), len(b)); a, b = a.zfill(n), b.zfill(n)
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))

def is_floats(x):
    if isinstance(x, float): return True
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(i, float) for i in x)

def close(a, b, atol, rtol=1e-7):  # numpy.allclose semantics, recursive over sequences
    if isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, atol, rtol) for x, y in zip(a, b))
    return a == b or abs(a - b) <= atol + rtol * abs(b)

def check(ds, entry, out, exp, inp, atol):
    exact = out == exp
    if ds == "mbpp":
        if entry == "are_equivalent": exact = True
        elif entry == "sum_div": exact = exact or out == 0
        elif entry == "surface_Area": exact = exact or abs(out - _surface(*inp)) <= atol
        elif entry == "digit_distance_nums": exact = exact or out == _digit_distance(*inp)
        elif entry in SET_EQ: exact = set(out) == set(exp)
        elif entry in NOT_NONE: exact = (out == exp) if isinstance(out, bool) else (exp == (out is not None))
    if ds == "humaneval" and entry == "find_zero":
        # EvalPlus: root check only; here also accept the canonical's answer (it fails the strict check on 6/888 inputs)
        assert abs(_poly(*inp, out)) <= atol or close(out, exp, max(atol, 1e-6))
        return
    a = 1e-6 if (atol == 0 and is_floats(exp)) else atol
    if not exact and a != 0:
        assert type(out) == type(exp)
        if isinstance(exp, (list, tuple)): assert len(out) == len(exp)
        assert close(out, exp, a)
    else:
        assert exact

class TO(BaseException): pass
def _alarm(s, f): raise TO()
signal.signal(signal.SIGALRM, _alarm)

ds, entry, tid = spec["dataset"], spec["entry_point"], spec["task_num"]
binp, pinp = list(spec["base_input"]), list(spec["plus_input"] or [])  # Mbpp/793 has an empty dict
if ds == "mbpp": binp, pinp = deser(tid, binp), deser(tid, pinp)
inputs = binp + pinp; nb = len(binp)
try:  # reference phase (trusted)
    g = {"__name__": "reference"}; exec(spec["ref_code"], g); rfn = g[entry]
    expected, rtimes = [], []
    for inp in inputs:
        a = copy.deepcopy(inp); t = time.perf_counter(); expected.append(rfn(*a)); rtimes.append(time.perf_counter() - t)
    if ds == "mbpp" and entry in NOT_NONE: expected = [e is not None for e in expected]
except BaseException as e:
    finish(status="ref_error", msg=repr(e)[:300])

# total candidate budget: base limit, extended (max 60 s) when the reference itself is slow; EvalPlus allows up to 60 s
TOTAL = min(60.0, max(spec["cand_timeout"], 3 * sum(rtimes)))
try:  # C-level hangs in the candidate die by SIGXCPU shortly after the budget
    ru = resource.getrusage(resource.RUSAGE_SELF)
    lim = int(ru.ru_utime + ru.ru_stime + TOTAL + 3)
    resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
except Exception:
    pass
for n in ("system", "fork", "forkpty", "kill", "killpg", "popen", "execv", "execve", "execvp", "execl", "execlp",
          "posix_spawn", "posix_spawnp", "spawnv", "spawnl"):
    if hasattr(os, n): setattr(os, n, _blocked)
subprocess.Popen = _blocked
sys.stdout = sys.stderr = open(os.devnull, "w")
try:
    signal.setitimer(signal.ITIMER_REAL, TOTAL)
    g = {"__name__": "candidate"}; exec(compile(spec["code"], "<candidate>", "exec"), g); fn = g[entry]
    signal.setitimer(signal.ITIMER_REAL, 0)
except TO:
    finish(status="timeout", msg="timeout while loading candidate")
except BaseException as e:
    signal.setitimer(signal.ITIMER_REAL, 0)
    finish(status="load_error", msg=repr(e)[:300])
t0 = time.monotonic(); first_fail = None; err = ""; status = "ok"
for i, inp in enumerate(inputs):
    left = TOTAL - (time.monotonic() - t0)
    if left <= 0:
        first_fail, err, status = i, "total time limit", "timeout"; break
    try:
        signal.setitimer(signal.ITIMER_REAL, min(max(spec["min_limit"], spec["factor"] * rtimes[i]), left))
        try: out = fn(*copy.deepcopy(inp))
        finally: signal.setitimer(signal.ITIMER_REAL, 0)
        check(ds, entry, out, expected[i], inp, spec["atol"])
    except TO:
        first_fail, err, status = i, "per-input timeout", "timeout"; break
    except BaseException as e:
        first_fail, err = i, repr(e)[:200]; break
finish(status=status, first_fail=first_fail, msg=err, n_base=nb, n_plus=len(pinp),
       base_pass=first_fail is None or first_fail >= nb, plus_pass=first_fail is None,
       cand_seconds=round(time.monotonic() - t0, 3))
'''
_SB_OK = None


def _sandbox_prefix(td):
    """sandbox-exec prefix (no network, writes only under td) or [] when unavailable."""
    global _SB_OK
    exe = shutil.which("sandbox-exec")
    if not exe or _SB_OK is False:
        return []
    prof = (f'(version 1)(allow default)(deny network*)(deny process-fork)(deny file-write* (require-not '
            f'(require-any (subpath "{os.path.realpath(td)}") (literal "/dev/null") (literal "/dev/dtracehelper"))))')
    if _SB_OK is None:
        _SB_OK = subprocess.run([exe, "-p", prof, sys.executable, "-I", "-B", "-c", "pass"], capture_output=True,
                                cwd=td).returncode == 0
    return [exe, "-p", prof] if _SB_OK else []


def _rss_mb(pid):
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout) / 1024
    except ValueError:
        return 0


def check_solution(dataset, task, code, cand_timeout=10.0, ref_timeout=40.0, mem_mb=1536):
    """Run `code` against base+plus inputs of `task` in an isolated subprocess. Returns a result dict."""
    spec = dict(dataset=dataset, entry_point=task["entry_point"], task_num=int(task["task_id"].split("/")[-1]),
                ref_code=task["prompt"] + task["canonical_solution"], code=code, base_input=task["base_input"],
                plus_input=task["plus_input"] or [], atol=task.get("atol") or 0, cand_timeout=cand_timeout,
                min_limit=1.0, factor=4.0, cpu_limit=ref_timeout + 60 + 10)
    with tempfile.TemporaryDirectory(prefix="qsuite_") as td:
        sp, rp, wp = (os.path.join(td, n) for n in ("spec.json", "result.json", "worker.py"))
        with open(sp, "w") as f: json.dump(spec, f)
        with open(wp, "w") as f: f.write(WORKER)
        env = {"PATH": "/usr/bin:/bin", "HOME": td, "TMPDIR": td, "LANG": "en_US.UTF-8"}
        cmd = _sandbox_prefix(td) + [sys.executable, "-I", "-B", wp, sp, rp]
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=td, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        reason, tick = None, 0
        while p.poll() is None:
            time.sleep(0.05); tick += 1
            if time.time() - t0 > ref_timeout + 60 + 10:
                reason = "timeout"
            elif tick % 5 == 0 and _rss_mb(p.pid) > mem_mb:  # RSS watchdog every 0.25 s
                reason = "memory"
            if reason:
                try: os.killpg(p.pid, 9)
                except ProcessLookupError: pass
                p.wait(); break
        try:
            with open(rp) as f: res = json.load(f)
        except Exception:
            res = dict(status=reason or ("timeout" if p.returncode == -24 else "crash"), msg=f"no result (exit {p.returncode})")
        if reason:
            res.update(status=reason)
        res["passed"] = bool(res.get("status") == "ok" and res.get("plus_pass") and res.get("base_pass"))
        res["total_seconds"] = round(time.time() - t0, 2)
        return res


# ----------------------------------------------------------------------------------------------- coding
FENCE = re.compile(r"```[ \t]*([\w+.-]*)[ \t]*\n(.*?)(?:```|\Z)", re.S)


def extract_code(text, entry):
    text = strip_think(text)
    blocks = [m.group(2) for m in FENCE.finditer(text)]
    if not blocks:
        return text
    for b in blocks:
        if re.search(rf"def\s+{re.escape(entry)}\b", b):
            return b
    return max(blocks, key=len)


def build_program(dataset, task, response):
    code = extract_code(response, task["entry_point"])
    if dataset == "humaneval":  # prompt = imports + helpers + stub; the model's def overrides the stub
        if re.search(rf"^\s*def\s+{re.escape(task['entry_point'])}\s*\(", code, re.M):
            return task["prompt"] + "\n" + code
        return task["prompt"] + code  # body-only completion
    return code


def coding_prompt(dataset, task):
    if dataset == "humaneval":
        return ("Complete the following Python function. Reply with the complete function (including any needed "
                "imports) in a single ```python code block and nothing else.\n\n```python\n" + task["prompt"] + "```")
    p = task["prompt"].strip().removeprefix('"""').removesuffix('"""').strip()
    return ("Write a Python function for the following task. Use exactly the function name and signature used in the "
            "example assertion. Reply with the complete code (including any needed imports) in a single ```python "
            "code block and nothing else.\n\n" + p)


def part_coding(c, suite, **_):
    jobs, pool, items = [], cf.ThreadPoolExecutor(2), []
    for ds, fname, step in (("humaneval", "HumanEvalPlus.jsonl.gz", HE_STEP), ("mbpp", "MbppPlus.jsonl.gz", MBPP_STEP)):
        tasks = load_plus(fname)
        for tid in subset_ids(tasks, step, suite):
            t = tasks[tid]
            try:
                r = c.chat([{"role": "user", "content": coding_prompt(ds, t)}], MAXTOK[ds])
            except Exception as e:
                items.append(dict(task_id=tid, dataset=ds, passed=False, status="request_error", msg=str(e)[:300]))
                continue
            prog = build_program(ds, t, r["content"])
            fut = pool.submit(check_solution, ds, t, prog)  # sandbox runs while the next generation is decoded
            jobs.append((tid, ds, r, fut, len(items))); items.append(None)
            log(f"coding {tid} generated ({r['usage'].get('completion_tokens')} tok, {r['finish']})")
    for tid, ds, r, fut, idx in jobs:
        try:
            res = fut.result()
        except Exception as e:  # harness failure on one item must not lose the others
            res = dict(passed=False, status="harness_error", msg=repr(e)[:300], total_seconds=0)
        items[idx] = dict(task_id=tid, dataset=ds, passed=res["passed"], base_pass=res.get("base_pass"),
                          plus_pass=res.get("plus_pass"), status=res["status"], msg=res.get("msg", ""),
                          completion_tokens=r["usage"].get("completion_tokens"), finish=r["finish"],
                          gen_seconds=r["seconds"], check_seconds=res["total_seconds"], response=r["content"][:3000])
    pool.shutdown()
    out = {}
    for ds, key in (("humaneval", "humaneval_plus"), ("mbpp", "mbpp_plus")):
        its = [i for i in items if i["dataset"] == ds]
        out[key] = dict(n=len(its), passed=sum(i["passed"] for i in its), base_passed=sum(bool(i.get("base_pass")) for i in its))
    n, p = sum(v["n"] for v in out.values()), sum(v["passed"] for v in out.values())
    return dict(score=p / n if n else None, passed=p, n=n, by_dataset=out, items=items)


# ----------------------------------------------------------------------------------------------- reasoning
def parse_num(s):
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    try:
        return Decimal(m[0]) if m else None
    except InvalidOperation:
        return None


def parse_answer(text):
    """Final numeric answer: last '#### x', else \\boxed{x}, else 'answer is x', else last number. Returns (Decimal, how)."""
    text = strip_think(text)
    for how, rx in (("####", r"####[ \t]*([^\n]*)"), ("boxed", r"\\boxed\{([^}]*)\}"),
                    ("answer_is", r"answer\s+is[:\s]*([^\n]*)")):
        found = re.findall(rx, text, flags=re.I)
        if found:
            v = parse_num(found[-1])
            if v is not None:
                return v, how
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return (parse_num(nums[-1]), "last_number") if nums else (None, None)


def part_reasoning(c, suite, **_):
    rows = load_gsm8k()
    items = []
    for idx in gsm_indices(len(rows), suite):
        q, gold_s = rows[idx]["question"], rows[idx]["answer"].split("####")[-1]
        gold = parse_num(gold_s)
        prompt = ("Solve the following math word problem. Reason step by step, concisely. Then write the final answer "
                  "on the last line exactly as `#### <number>` (digits only, no units or symbols).\n\nProblem: " + q)
        try:
            r = c.chat([{"role": "user", "content": prompt}], MAXTOK["gsm8k"])
        except Exception as e:
            items.append(dict(index=idx, passed=False, status="request_error", msg=str(e)[:300])); continue
        pred, how = parse_answer(r["content"])
        ok = pred is not None and abs(pred - gold) < Decimal("1e-6")
        items.append(dict(index=idx, gold=str(gold), pred=None if pred is None else str(pred), parse=how, passed=ok,
                          completion_tokens=r["usage"].get("completion_tokens"), finish=r["finish"],
                          seconds=r["seconds"], response=r["content"][-1200:]))
        log(f"gsm8k {idx} gold={gold} pred={pred} ({how}) {'OK' if ok else 'X'}")
    p = sum(i["passed"] for i in items)
    return dict(score=p / len(items) if items else None, passed=p, n=len(items), items=items)


# ----------------------------------------------------------------------------------------------- tools
class L:
    """Loose matcher: predicate + a sample value that satisfies it (used by the mock server in the tests)."""
    def __init__(self, pred, sample): self.pred, self.sample = pred, sample
    def __call__(self, v): return self.pred(v)


def T(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
has = lambda *subs: L(lambda v: isinstance(v, str) and all(s in v.lower() for s in subs), " ".join(subs))
WEATHER = T("get_weather", "Get the current weather for a city.",
            {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"])
FILES = [T("read_file", "Read a text file and return its contents.", {"path": S}, ["path"]),
         T("write_file", "Write text to a file, replacing it.", {"path": S, "content": S}, ["path", "content"]),
         T("list_dir", "List the entries of a directory.", {"path": S}, ["path"])]
TOOL_CASES = [
    dict(id="weather", user="What's the weather like in Paris right now? Please use celsius.", tools=[WEATHER],
         expect=[("get_weather", dict(city="Paris", unit="celsius"))]),
    dict(id="calc", user="What is 1234 multiplied by 5678? Use the calculator tool.",
         tools=[T("calculate", "Evaluate an arithmetic expression.", {"expression": S}, ["expression"])],
         expect=[("calculate", dict(expression=L(lambda v: isinstance(v, str) and "1234" in v and "5678" in v, "1234 * 5678")))]),
    dict(id="email", user="Send an email to anna.rossi@example.com with the subject 'Quarterly report' telling her the report is attached.",
         tools=[T("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"])],
         expect=[("send_email", dict(to="anna.rossi@example.com", subject=has("quarterly report"),
                                     body=L(lambda v: isinstance(v, str) and len(v.strip()) > 0, "Hi Anna, the report is attached.")))]),
    dict(id="event", user="Schedule a 45 minute meeting titled 'Design review' on 2026-11-03.",
         tools=[T("create_event", "Create a calendar event.", {"title": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                                              "duration_minutes": I}, ["title", "date", "duration_minutes"])],
         expect=[("create_event", dict(title=has("design review"), date="2026-11-03", duration_minutes=45))]),
    dict(id="stock", user="What is the current share price of NVDA?",
         tools=[WEATHER, T("search_web", "Search the web.", {"query": S}, ["query"]),
                T("get_stock_price", "Get the latest stock price for a ticker symbol.", {"symbol": S}, ["symbol"])],
         expect=[("get_stock_price", dict(symbol="NVDA"))]),
    dict(id="read_file", user="Show me the contents of /etc/hostname.", tools=FILES,
         expect=[("read_file", dict(path="/etc/hostname"))]),
    dict(id="write_file", user="Create the file notes/todo.txt containing exactly the text: buy milk", tools=FILES,
         expect=[("write_file", dict(path="notes/todo.txt", content=has("buy milk")))]),
    dict(id="flights", user="Find direct flights from Rome to Tokyo for 2 passengers.",
         tools=[T("search_flights", "Search flights.", {"origin": S, "destination": S, "passengers": I, "direct_only": B},
                  ["origin", "destination", "passengers"])],
         expect=[("search_flights", dict(origin="Rome", destination="Tokyo", passengers=2, direct_only=True))]),
    dict(id="timer", user="Set a timer for 5 minutes.",
         tools=[T("set_timer", "Start a countdown timer.", {"seconds": I}, ["seconds"])],
         expect=[("set_timer", dict(seconds=300))]),
    dict(id="sql", user="How many rows are in the table `users`? Use the SQL tool.",
         tools=[T("run_sql", "Run a read-only SQL query.", {"query": S}, ["query"])],
         expect=[("run_sql", dict(query=L(lambda v: isinstance(v, str) and re.search(r"select\s+count\s*\(", v, re.I) is not None
                                          and "users" in v.lower(), "SELECT COUNT(*) FROM users")))]),
    dict(id="cart_array", user="Add 2 units of SKU A-100 and 1 unit of SKU B-200 to my cart.",
         tools=[T("add_to_cart", "Add items to the shopping cart.",
                  {"items": {"type": "array", "items": {"type": "object", "properties": {"sku": S, "qty": I},
                                                         "required": ["sku", "qty"]}}}, ["items"])],
         expect=[("add_to_cart", dict(items=L(lambda v: isinstance(v, list) and sorted((i.get("sku"), i.get("qty")) for i in v if isinstance(i, dict))
                                              == [("A-100", 2), ("B-200", 1)], [{"sku": "A-100", "qty": 2}, {"sku": "B-200", "qty": 1}])))]),
    dict(id="parallel", user="Tell me the current weather in Rome and in Berlin (celsius).", tools=[WEATHER],
         expect=[("get_weather", dict(city="Rome")), ("get_weather", dict(city="Berlin"))]),
    dict(id="disambiguation", user="Will it rain in Oslo tomorrow?",
         tools=[WEATHER, T("get_forecast", "Get the multi-day weather forecast for a city.", {"city": S, "days": I}, ["city"])],
         expect=[("get_forecast", dict(city="Oslo"))]),
    dict(id="no_args", user="List all my open tasks.",
         tools=[T("list_tasks", "List the user's tasks.", {}, []), T("create_task", "Create a task.", {"title": S}, ["title"])],
         expect=[("list_tasks", {})]),
]


def ideal_calls(case):
    """Correct tool calls for a case (concrete values) -- used by the mock server in the tests."""
    return [(n, {k: (v.sample if isinstance(v, L) else v) for k, v in a.items()}) for n, a in case["expect"]]


def _match(v, m):
    if callable(m): return bool(m(v))
    if isinstance(m, bool): return v is m
    if isinstance(m, str): return isinstance(v, str) and v.strip().lower() == m.lower()
    if isinstance(m, (int, float)): return isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v - m) < 1e-9
    return v == m


def norm_tool_calls(raw):
    out = []
    for tc in raw:
        f = tc.get("function") or {}
        a, ok = f.get("arguments"), True
        if isinstance(a, str):
            try: a = json.loads(a)
            except ValueError: a, ok = None, False
        out.append(dict(name=f.get("name"), arguments=a, valid_json=ok and isinstance(a, dict)))
    return out


def score_tool_case(case, calls):
    """(passed, reason). Strict: structured tool_calls only, count must match, names/args must match."""
    if len(calls) != len(case["expect"]):
        return False, f"expected {len(case['expect'])} call(s), got {len(calls)}"
    if not all(c["valid_json"] for c in calls):
        return False, "invalid JSON arguments"
    left = list(calls)
    for name, args in case["expect"]:
        for c in left:
            if c["name"] == name and all(k in c["arguments"] and _match(c["arguments"][k], m) for k, m in args.items()):
                left.remove(c); break
        else:
            return False, f"no matching call for {name}({', '.join(args)})"
    return True, "ok"


def part_tools(c, suite, **_):
    items = []
    for case in TOOL_CASES:
        try:
            r = c.chat([{"role": "user", "content": case["user"]}], MAXTOK["tools"], tools=case["tools"])
            calls = norm_tool_calls(r["tool_calls"])
            ok, why = score_tool_case(case, calls)
            if not calls and re.search(r"<tool_call>|<function=", r["content"]):
                why = "tool call emitted as text in content (server did not parse it)"
            items.append(dict(id=case["id"], passed=ok, reason=why, calls=calls, content=r["content"][:400],
                              finish=r["finish"], seconds=r["seconds"]))
        except Exception as e:
            items.append(dict(id=case["id"], passed=False, reason=f"request_error: {str(e)[:300]}"))
        log(f"tools {case['id']}: {'OK' if items[-1]['passed'] else 'X ' + items[-1]['reason']}")
    p = sum(i["passed"] for i in items)
    return dict(score=p / len(items), passed=p, n=len(items), items=items)


# ----------------------------------------------------------------------------------------------- needle
ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"
INSTR = ("Below is a large collection of Python source files, concatenated. Somewhere inside it a single sentence "
         "states the secret access code of a named vault. Read everything carefully; you will be asked about that "
         "code at the end.\n\n")


def make_needle(ctx, depth):
    r = random.Random(f"needle-{SEED}-{ctx}-{depth}")
    label = "".join(r.choice(ALPHA) for _ in range(3)) + "-" + "".join(r.choice("23456789") for _ in range(3))
    value = "".join(r.choice(ALPHA) for _ in range(4)) + "-" + "".join(r.choice("23456789") for _ in range(4)) \
        + "-" + "".join(r.choice(ALPHA) for _ in range(4))
    return label, value, f"The secret access code for vault {label} is {value}."


def build_needle_prompt(ctx, depth, tok, corpus):
    """Return (text, label, value, n_tokens): exactly `ctx` tokens before the server re-tokenises the text."""
    label, value, sentence = make_needle(ctx, depth)
    question = f"\n\nQuestion: What is the secret access code for vault {label}? Reply with the code only."
    ph = "<<HAY>>"
    s = tok.apply_chat_template([{"role": "user", "content": ph}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    head, tail = s.split(ph)
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    h, t, ins, q, nd = enc(head), enc(tail), enc(INSTR), enc(question), enc("\n\n" + sentence + "\n\n")
    n_hay = ctx - len(h) - len(t) - len(ins) - len(q) - len(nd)
    p = int(depth * n_hay)
    for k in range(p, min(p + 300, n_hay)):  # move the insertion point to a line boundary
        if "\n" in tok.decode([corpus[k]]):
            p = k + 1; break
    hay = list(corpus[:p]) + nd + list(corpus[p:n_hay])
    ids = h + ins + hay + q + t
    assert len(ids) == ctx, (len(ids), ctx)
    return tok.decode(ids), label, value, len(ids)


def part_needle(c, suite, ctxs=None, depths=None, **_):
    import prompts  # bench/prompts.py (frozen); tokenizer from the model dir, read-only
    tok, corpus = prompts.tokenizer(), prompts.corpus_ids()
    ctxs = ctxs or NEEDLE_CTXS
    depths = depths or (NEEDLE_DEPTHS_FULL if suite == "full" else NEEDLE_DEPTHS_QUICK)
    items = []
    for ctx in ctxs:
        for d in depths:
            text, label, value, n = build_needle_prompt(ctx, d, tok, corpus)
            try:
                r = c.complete(text, MAXTOK["needle"], timeout=3600)
                found = value in r["text"]
                items.append(dict(ctx=ctx, depth=d, passed=found, value=value, answer=r["text"][:200],
                                  prompt_tokens_local=n, prompt_tokens_retok=len(tok.encode(text, add_special_tokens=False)),
                                  prompt_tokens_server=r["usage"].get("prompt_tokens"),
                                  seconds=r["seconds"], finish=r["finish"]))
            except Exception as e:
                items.append(dict(ctx=ctx, depth=d, passed=False, value=value, status="request_error", msg=str(e)[:300]))
            log(f"needle ctx={ctx} depth={d}: {'FOUND' if items[-1]['passed'] else 'MISSED'} ({items[-1].get('seconds')} s)")
    p = sum(i["passed"] for i in items)
    return dict(score=p / len(items), passed=p, n=len(items), items=items)


# ----------------------------------------------------------------------------------------------- identity
IDENT_CODING = [
    "Write a Python function that merges overlapping intervals given as a list of (start, end) tuples, and explain its time complexity.",
    "Implement an LRU cache class in Python using collections.OrderedDict with get and put methods, and show a short usage example.",
    "Explain the difference between asyncio.gather and asyncio.wait in Python, with a minimal runnable example of each.",
]


def first_div(a, b):
    """Index of the first differing character, or None when identical."""
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    return None if i == n and len(a) == len(b) else i


def part_identity(c, suite, ref=None, **_):
    import prompts
    tok = prompts.tokenizer()
    plist = [(f"bench_ctx2048_task{t}", tok.decode(prompts.prompt_ids(2048, t))) for t in range(3)]
    for i, q in enumerate(IDENT_CODING):  # chat template default (thinking on), same as the bench prompts
        plist.append((f"coding{i}", tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                            add_generation_prompt=True)))
    refitems = {i["id"]: i for i in (ref or {}).get("parts", {}).get("identity", {}).get("items", [])}
    items = []
    for pid, text in plist:
        try:
            r = c.complete(text, MAXTOK["identity"])
        except Exception as e:
            items.append(dict(id=pid, status="request_error", msg=str(e)[:300], vs_ref="ERROR")); continue
        it = dict(id=pid, text=r["text"], completion_tokens=r["usage"].get("completion_tokens"), finish=r["finish"],
                  seconds=r["seconds"], vs_ref=None)
        if pid in refitems:
            d = first_div(refitems[pid]["text"], r["text"])
            if d is None:
                it["vs_ref"] = "IDENTICAL"
            else:
                it["vs_ref"] = f"DIVERGE at char {d} (~token {len(tok.encode(r['text'][:d], add_special_tokens=False))})"
        items.append(it)
        log(f"identity {pid}: {it['vs_ref'] or 'generated'}")
    comp = [i for i in items if i["vs_ref"]]
    ident = sum(i["vs_ref"] == "IDENTICAL" for i in comp)
    return dict(score=(ident / len(comp)) if comp else None, passed=ident if comp else None, n=len(items),
                compared=len(comp), items=items)


# ----------------------------------------------------------------------------------------------- main
PARTS = dict(identity=part_identity, tools=part_tools, reasoning=part_reasoning, coding=part_coding, needle=part_needle)


def selftest_sandbox(suite):
    """Canonical solutions must pass the sandbox, a stub body must fail -- validates the EvalPlus port."""
    bad = 0
    for ds, fname, step in (("humaneval", "HumanEvalPlus.jsonl.gz", HE_STEP), ("mbpp", "MbppPlus.jsonl.gz", MBPP_STEP)):
        tasks = load_plus(fname)
        for tid in subset_ids(tasks, step, suite):
            t = tasks[tid]
            code = (t["prompt"] if ds == "humaneval" else "") + t["canonical_solution"]
            r = check_solution(ds, t, code)
            if not r["passed"]:
                bad += 1; print("CANONICAL FAIL", tid, r.get("status"), r.get("msg"))
        print(f"{ds}: {len(subset_ids(tasks, step, suite))} canonical checked")
    print("selftest:", "OK" if not bad else f"{bad} failures")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", default=None)
    ap.add_argument("--part", default="all", choices=list(PARTS) + ["all"])
    ap.add_argument("--suite", default="quick", choices=["quick", "full"])
    ap.add_argument("--out", default=None); ap.add_argument("--ref", default=None, help="reference result JSON (identity)")
    ap.add_argument("--timeout", type=int, default=900, help="per-request timeout in s (needle uses 3600)")
    ap.add_argument("--needle-ctxs", type=int, nargs="+", default=None, help="override contexts (tests only)")
    ap.add_argument("--needle-depths", type=float, nargs="+", default=None)
    ap.add_argument("--fetch-data", action="store_true"); ap.add_argument("--selftest-sandbox", action="store_true")
    a = ap.parse_args()
    if a.fetch_data:
        print(json.dumps(fetch_data(), indent=1)); return 0
    data_hashes = verify_data()
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    if a.selftest_sandbox:
        return 0 if selftest_sandbox(a.suite) else 1
    if not a.model or not a.out:
        ap.error("--model and --out are required")
    c = Client(a.url, a.model, a.timeout)
    try:
        c._req("/v1/models", timeout=15)
    except urllib.error.URLError as e:
        print(f"server not reachable at {a.url}: {e}", file=sys.stderr); return 2
    except Exception:
        pass  # /v1/models missing is fine
    ref = None
    if a.ref:
        with open(a.ref) as f: ref = json.load(f)
    me = os.path.abspath(__file__)
    thr = os.path.join(HERE, "THRESHOLDS.md")
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(schema=1, created=datetime.datetime.now().isoformat(timespec="seconds"), suite=a.suite, url=a.url,
                model=a.model, harness=dict(script=os.path.relpath(me, ROOT), sha256=sha256_file(me),
                thresholds_sha256=sha256_file(thr) if os.path.exists(thr) else None, git_commit=git or None,
                python=sys.version.split()[0], platform=platform.platform()),
                datasets=data_hashes, sampling=dict(GREEDY, thinking="off (coding/reasoning/tools)"),
                server=c.info())
    doc = meta
    if os.path.exists(a.out):  # merge parts run separately into one file when suite+model match
        with open(a.out) as f: old = json.load(f)
        if old.get("suite") == a.suite and old.get("model") == a.model:
            doc = dict(old, **{k: v for k, v in meta.items() if k not in ("parts", "timings")})
    doc.setdefault("parts", {}); doc.setdefault("timings", {})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t_all = time.time()
    for name in (list(PARTS) if a.part == "all" else [a.part]):
        t0 = time.time(); log(f"=== part {name} ({a.suite})")
        res = PARTS[name](c, a.suite, ref=ref, ctxs=a.needle_ctxs, depths=a.needle_depths)
        res["seconds"] = round(time.time() - t0, 1)
        doc["parts"][name], doc["timings"][name] = res, res["seconds"]
        doc["total_seconds"] = round(time.time() - t_all, 1)
        with open(a.out, "w") as f: json.dump(doc, f, indent=1)  # incremental: a killed run keeps finished parts
        log(f"part {name}: score={res.get('score')} passed={res.get('passed')}/{res['n']} in {res['seconds']} s")
    print(json.dumps({k: {"score": v.get("score"), "passed": v.get("passed"), "n": v["n"], "seconds": v["seconds"]}
                      for k, v in doc["parts"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
