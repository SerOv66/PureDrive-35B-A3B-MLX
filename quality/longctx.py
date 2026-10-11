#!/usr/bin/env python3
"""Long-context retrieval suite for the KV8 gate (DOSSIER_KV, condition C1): needle 3 depths x 32K/64K/128K plus a
RULER-like set (multi-key, multi-value, variable tracking) at 64K/128K.  Greedy, thinking OFF, /v1/completions with the chat
template applied by hand (same convention as run_quality.py --part needle).  Never loads a model: talks to --url only.

Haystack = third-party Python code from runtimes/venv-mlx site-packages (NOT the bench corpus, NOT the CPython stdlib), concatenated
in a fixed order.  Everything is deterministic (seed 1234 + fixed package list); the sha256 of the haystack and of every prompt is
stored in the result file, and --compare refuses to compare runs whose prompts differ.

Layouts (--layout)
  shared    (default) ONE document per context holding the facts of every case; each case only changes the final question.
            A cold prefill per context (32K/64K/128K = 3 in total); with the server prefix cache the other cases cost a few s.
            The 3 needles of a context coexist in the same document (multi-needle: the question names the vault).
  separate  one document per task set (a needle per depth, one multi-key, one multi-value, one variable-tracking): classic
            single-needle purity, but a cold prefill per document (128K: 6 x ~5 min).

Usage (workspace root, PY=${PYTHON:-runtimes/venv-mlx/bin/python})
  $PY quality/longctx.py --dry-run [--only 131072/vartrack]                     # cases + token statistics, no network
  $PY quality/longctx.py --url URL --model coder-q6 --label bf16 --only 131072 --out quality/results/longctx_bf16.json
  $PY quality/longctx.py --compare quality/results/longctx_bf16.json quality/results/longctx_kv8.json   # exit 1 = regressions
--only CTX[/TYPE[/ID]] (repeatable) selects cases (types: needle multikey multivalue vartrack); the document does not depend on --only.
The result file is merged and written atomically after every case: re-running skips finished cases (--redo repeats them), so the
suite can be split in invocations of <=10 min (--budget S stops cleanly before a case that would not fit; exit 3 = partial).
Exit codes: run 0 ok / 2 server unreachable or result file incompatible / 3 stopped by --budget; compare 0 / 1 regressions / 2 not comparable.
"""
import argparse, functools, hashlib, json, os, random, re, sys, time, urllib.error
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "bench")]
from run_quality import ALPHA, GREEDY, SEED, Client, log, make_needle, sha256_file, strip_think  # noqa: E402,F401

SITE = os.path.join(ROOT, "runtimes", "venv-mlx", "lib", "python3.12", "site-packages")
PACKAGES = ["anyio", "click", "fastapi", "h11", "httpcore", "httpx", "jinja2", "markdown_it", "starlette", "packaging",
            "filelock", "fsspec"]
HAYSTACK_SHA256 = "57fb6b7254dfd59ba01033b0b5ec4659719935496bcabfa0b4f0e910137ee3d8"   # of the concatenated text (1163935 chars)
PER_PKG = 90_000                      # chars of source per package (sorted paths, size 2..120 KB, no tests)
NEEDLE_CTXS, RULER_CTXS, DEPTHS = [32768, 65536, 131072], [65536, 131072], [0.1, 0.5, 0.9]
TYPES = ("needle", "multikey", "multivalue", "vartrack")
MAXTOK = dict(needle=64, multikey=64, multivalue=128, vartrack=128)
RESERVE, GAP = 160, 300               # prompt <= ctx-160 tokens (answer <=128 fits); min tokens between two inserted facts
# Measured oMLX prefill speed (notebook EXP-009: 32K 687 t/s, 64K 749, 128K 457 -> TTFT 287 s) and a guess for one warm case.
PREFILL_TPS, WARM_S = {32768: 687, 65536: 749, 131072: 457}, 8
ADJ = "amber brave calm dusty eager faint gentle hollow icy jolly keen lucid mellow noble pale quiet rusty silent tidy upbeat vivid witty young zesty".split()
NOUN = "falcon harbor lantern meadow nebula orchard pebble quartz river summit thicket valley willow anchor beacon canyon dune ember fjord glacier harvest island jungle kettle lagoon".split()
CONS = "BCDFGHJKLMNPQRSTVWXZ"          # no vowels: a variable name can never be an English word in the answer
# ponytail: one generic preface for every layout (it names all fact formats even when a 'separate' document holds only some)
INSTR = ("Below is a large collection of Python source files, concatenated. Scattered among the code there are a few plain "
         "lines inserted by hand: sentences stating the secret access code of a named vault, sentences stating a special magic "
         "number of a named key, and variable lines such as `VAR_ABCDE = 48213` (a value) or `VAR_FGHIJ = VAR_ABCDE` (a copy "
         "of another variable). Read everything carefully; you will be asked about them at the end.\n\n")


# ----------------------------------------------------------------------------------------------- haystack
def load_haystack():
    sp, chunks = Path(SITE), []
    for pkg in PACKAGES:
        if not (sp / pkg).is_dir():
            log("haystack: package missing, skipped:", pkg); continue
        got = 0
        for f in sorted((sp / pkg).rglob("*.py")):
            rel = f.relative_to(sp)
            if {"tests", "test"} & set(rel.parts) or f.name.startswith("test_") or not 2048 < f.stat().st_size < 120_000:
                continue
            try:
                body = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            chunks.append(f"# ===== file: {rel} =====\n{body}\n"); got += len(body)
            if got >= PER_PKG:
                break
    text = "".join(chunks)
    return text, hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- cases
def gen_tasksets(ctx, hay_text):
    """Task sets of a context: {id, type, facts[{frac,text,tag}], cases[{id,type,question,answers,forbid,tags,maxtok}]}.
    `forbid` = values of the other facts of the same task (wrong-but-plausible answers: their presence fails the case)."""
    r = random.Random(f"longctx-{SEED}-{ctx}")
    used, T = set(), []

    def uniq(fn):
        while True:
            v = fn()
            if v not in used and v not in hay_text:
                used.add(v); return v
    num = lambda: uniq(lambda: str(r.randrange(10**6, 10**7)))
    key = lambda: uniq(lambda: f"{r.choice(ADJ)}-{r.choice(NOUN)}")
    var = lambda: uniq(lambda: "VAR_" + "".join(r.choices(CONS, k=5)))
    strat = lambda n, lo, hi: [lo + (hi - lo) * (j + r.uniform(0.2, 0.8)) / n for j in range(n)]
    case = lambda typ, name, q, ans, forbid, tags: dict(id=f"{ctx}/{typ}/{name}", type=typ, question="Question: " + q,
                                                         answers=ans, forbid=forbid, tags=tags, maxtok=MAXTOK[typ])
    if ctx in NEEDLE_CTXS:
        vault = {d: make_needle(ctx, d) for d in DEPTHS}                    # (label, value, sentence), run_quality format
        for d, (label, value, sent) in vault.items():
            nm = f"d{round(d * 100)}"
            T.append(dict(id=f"needle/{nm}", type="needle", facts=[dict(frac=d, text=sent, tag=nm)], cases=[case(
                "needle", nm, f"What is the secret access code for vault {label}? Reply with the code only.", [value],
                [v for dd, (_, v, _) in vault.items() if dd != d], [nm])]))
    if ctx in RULER_CTXS:
        keys, nums, fr = [key() for _ in range(5)], [num() for _ in range(5)], strat(5, .05, .95)       # multi-key
        facts = [dict(frac=fr[i], text=f"One of the special magic numbers for {keys[i]} is: {nums[i]}.", tag=f"mk{i}") for i in range(5)]
        T.append(dict(id="multikey", type="multikey", facts=facts, cases=[case(
            "multikey", str(j), f"What is the special magic number for {keys[i]} mentioned in the text above? Reply with the number only.",
            [nums[i]], [nums[k] for k in range(5) if k != i], [f"mk{i}"]) for j, i in enumerate((0, 2, 4))]))
        mk, vals, facts = [key() for _ in range(3)], [[num() for _ in range(4)] for _ in range(3)], []        # multi-value
        for i in range(3):
            facts += [dict(frac=f, text=f"One of the special magic numbers for {mk[i]} is: {vals[i][j]}.", tag=f"mv{i}-{j}")
                      for j, f in enumerate(strat(4, .03, .97))]
        T.append(dict(id="multivalue", type="multivalue", facts=facts, cases=[case(
            "multivalue", str(i), f"What are all the special magic numbers for {mk[i]} mentioned in the text above? List every one "
            "of them, separated by commas, and nothing else.", vals[i], [v for k in range(3) if k != i for v in vals[k]],
            [f"mv{i}-{j}" for j in range(4)]) for i in range(3)]))
        chains, facts = [], []                                                                              # variable tracking
        for ci in range(5):                                                       # 0-2 targets (5 hops), 3-4 distractor chains
            names, value = [var() for _ in range(6 if ci < 3 else 4)], num()
            lines = [f"{names[0]} = {value}"] + [f"{names[j]} = {names[j - 1]}" for j in range(1, len(names))]
            facts += [dict(frac=f, text=lines[j], tag=f"vt{ci}-{j}") for j, f in enumerate(strat(len(names), .03, .97))]
            chains.append((names, value))
        T.append(dict(id="vartrack", type="vartrack", facts=facts, cases=[case(
            "vartrack", str(i), f"Find all variables that are assigned the value {chains[i][1]} in the text above, directly or "
            "through other variables (a line `VAR_X = VAR_Y` copies the value of VAR_Y). List all their names, separated by spaces, "
            "and nothing else.", chains[i][0], [n for k in range(5) if k != i for n in chains[k][0]],
            [f"vt{i}-{j}" for j in range(len(chains[i][0]))]) for i in range(3)]))
    return T


def compact(s):
    return re.sub(r"[^0-9A-Z]", "", strip_think(s).upper())


def score_case(case, text):
    """-> (ok, score, found, extras): ok = every expected item present and no `forbid` item present."""
    if case["type"] == "vartrack":     # whole 5-letter words: the answer may drop the VAR_ prefix
        got = set(re.findall(r"(?<![A-Z])[A-Z]{5}(?![A-Z])", strip_think(text).upper()))
        found = [a for a in case["answers"] if a[4:] in got]
        extras = [f for f in case["forbid"] if f[4:] in got]
    else:                              # separators/case ignored: ABCD-2345-EFGH == abcd2345efgh, 1,234,567 == 1234567
        c = compact(text)
        found = [a for a in case["answers"] if compact(a) in c]
        extras = [f for f in case["forbid"] if compact(f) in c]
    return len(found) == len(case["answers"]) and not extras, len(found) / len(case["answers"]), found, extras


# ----------------------------------------------------------------------------------------------- documents
def build_doc(ctx, tsets, tok, hay_ids):
    """Token-exact prefix (chat head + preface + haystack with the facts inserted at line boundaries + '\\n\\n') shared by
    all cases of the document; per case only `question + chat tail` follows.  Prompt <= ctx - RESERVE tokens after re-tokenising."""
    ph = "<<HAY>>"
    head, tail = tok.apply_chat_template([{"role": "user", "content": ph}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False).split(ph)
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    nl = functools.lru_cache(None)(lambda i: "\n" in tok.decode([i]))
    suffix = {c["id"]: c["question"] + tail for t in tsets for c in t["cases"]}
    st = max(len(enc(s)) for s in suffix.values())
    facts = sorted((f for t in tsets for f in t["facts"]), key=lambda f: f["frac"])
    fids, base, end = [enc("\n\n" + f["text"] + "\n\n") for f in facts], enc(head) + enc(INSTR), enc("\n\n")
    n_hay = ctx - RESERVE - st - len(base) - len(end) - sum(map(len, fids))
    for _ in range(6):
        n_hay = min(n_hay, len(hay_ids))
        while n_hay > 0 and not nl(hay_ids[n_hay - 1]):          # end on a line boundary (no cut multi-byte character)
            n_hay -= 1
        ids, prev, last, at = list(base), 0, -GAP, {}
        for f, fi in zip(facts, fids):
            p = min(max(int(f["frac"] * n_hay), last + GAP), n_hay - 1)
            p = next((k + 1 for k in range(p, min(p + 300, n_hay)) if nl(hay_ids[k])), p)   # insert after a line break
            p = max(p, prev)
            ids += hay_ids[prev:p]; at[f["tag"]] = len(ids); ids += fi; prev = last = p
        ids += hay_ids[prev:n_hay] + end
        text = tok.decode(ids)
        retok = len(enc(text))
        over = retok + st - (ctx - RESERVE)
        if over <= 0:
            break
        n_hay -= over
    else:
        raise RuntimeError(f"cannot fit ctx {ctx}")
    assert "�" not in text
    return dict(text=text, tokens=retok, local=len(ids), st=st, hay=n_hay, suffix=suffix, enc=enc,
                depth={t: round(o / len(ids), 4) for t, o in at.items()}, nfacts=len(facts))


def plan(ctxs, layout, hay_text):
    """-> [(doc_id, ctx, tasksets)] in context order."""
    docs = []
    for ctx in sorted(ctxs):
        ts = gen_tasksets(ctx, hay_text)
        if layout == "shared":
            docs.append((str(ctx), ctx, ts))
        else:
            docs += [(f"{ctx}-{t['id'].replace('/', '')}", ctx, [t]) for t in ts]
    return docs


def selected(cid, only):
    return not only or any(cid == o or cid.startswith(o + "/") for o in only)


def est_seconds(ctx, tokens, ncases, cache):
    cold = tokens / PREFILL_TPS.get(ctx, 600) + 3
    return cold + (ncases - 1) * WARM_S if cache else ncases * cold


# ----------------------------------------------------------------------------------------------- compare
def compare(ref_path, cand_path):
    R, C = (json.load(open(p)) for p in (ref_path, cand_path))
    for k in ("layout", "haystack_sha256"):
        if R.get(k) != C.get(k):
            print(f"NOT COMPARABLE: {k} differs ({R.get(k)} vs {C.get(k)})"); return 2
    rc, cc = R["cases"], C["cases"]
    bad = [i for i in rc if i in cc and rc[i].get("prompt_sha256") != cc[i].get("prompt_sha256")]
    if bad:
        print("NOT COMPARABLE: different prompts for", bad[:6]); return 2
    ids = sorted(rc, key=lambda i: (int(i.split("/")[0]), TYPES.index(i.split("/")[1]), i))
    missing = [i for i in ids if i not in cc]
    reg = [i for i in ids if i in cc and rc[i]["ok"] and not cc[i]["ok"]]
    gain = [i for i in ids if i in cc and not rc[i]["ok"] and cc[i]["ok"]]
    soft = [i for i in ids if i in cc and cc[i]["score"] < rc[i]["score"] and i not in reg]
    base_fail = [i for i in ids if not rc[i]["ok"]]
    print(f"ref {R.get('label')} ({ref_path})  vs  cand {C.get('label')} ({cand_path})")
    print(f"{'cell':34} ref  cand   note")
    for i in ids:
        r, c = rc[i], cc.get(i)
        note = "REGRESSION" if i in reg else "gain" if i in gain else "baseline fails (not gating)" if not r["ok"] else \
            "MISSING in cand" if c is None else "score drop %.2f->%.2f" % (r["score"], c["score"]) if i in soft else ""
        print(f"{i:34} {'ok' if r['ok'] else 'X ':4} {'--' if c is None else 'ok' if c['ok'] else 'X ':5}  {note}")
    n_ok = lambda d, ids_: sum(1 for i in ids_ if i in d and d[i]["ok"])
    print(f"ref {n_ok(rc, ids)}/{len(ids)}  cand {n_ok(cc, ids)}/{len(ids)}  regressions {len(reg)}  gains {len(gain)}  "
          f"soft score drops {len(soft)}  missing {len(missing)}  baseline-fail {len(base_fail)}")
    if reg:
        print("FAIL: regressions", reg); return 1
    if missing:
        print("INCOMPLETE: cells of the reference not run by the candidate (zero regressions cannot be claimed)", missing[:8]); return 2
    print("PASS: zero regressions"); return 0


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model"); ap.add_argument("--out")
    ap.add_argument("--label", default="", help="free tag of the run, e.g. bf16 / kv8")
    ap.add_argument("--layout", default="shared", choices=["shared", "separate"])
    ap.add_argument("--only", nargs="+", default=None, help="CTX[/TYPE[/ID]] ...")
    ap.add_argument("--ctxs", type=int, nargs="+", default=None, help="override the contexts (tests only; all types at all of them)")
    ap.add_argument("--timeout", type=int, default=1500, help="per-request timeout in s")
    ap.add_argument("--budget", type=float, default=None, help="seconds: do not start a case estimated to end after this")
    ap.add_argument("--cache", type=int, default=1, choices=[0, 1], help="server prefix cache on (1) / off (0): estimates and --budget only")
    ap.add_argument("--redo", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prefill-to", type=int, default=None, help="only send the document prefix cut at N tokens (max_tokens 1) to fill "
                    "the server prefix cache, no cases: staged cold prefill when one request would exceed 600 s (256K, EXP-106)")
    ap.add_argument("--allow-new-haystack", action="store_true", help="site-packages changed (pip): accept another haystack (all runs to compare must redo it)")
    ap.add_argument("--compare", nargs=2, metavar=("REF", "CAND"))
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    import prompts  # bench/prompts.py: project tokenizer (model dir, read-only)
    tok = prompts.tokenizer()
    hay_text, hay_sha = load_haystack()
    if hay_sha != HAYSTACK_SHA256 and not a.allow_new_haystack:
        print(f"haystack sha256 {hay_sha} != pinned {HAYSTACK_SHA256}: site-packages changed", file=sys.stderr); return 2
    hay_ids = tok.encode(hay_text, add_special_tokens=False)
    if a.ctxs:                                  # ponytail: tests only, mutates the module defaults so every type runs at these contexts
        NEEDLE_CTXS[:] = RULER_CTXS[:] = a.ctxs
    ctxs = sorted(set(NEEDLE_CTXS + RULER_CTXS))
    log(f"haystack {len(hay_text)} chars, {len(hay_ids)} tokens, sha256 {hay_sha[:16]}")
    docs = [(d, ctx, ts) for d, ctx, ts in plan(ctxs, a.layout, hay_text) if any(selected(c["id"], a.only) for t in ts for c in t["cases"])]
    if not docs:
        print("no case selected"); return 2
    rec, out, started = {}, None, time.time()
    if not a.dry_run:
        if not (a.model and a.out):
            ap.error("--model and --out are required (unless --dry-run/--compare)")
        out = a.out; os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        c = Client(a.url, a.model, a.timeout)
        try:
            c._req("/v1/models", timeout=15)
        except urllib.error.URLError as e:
            print(f"server not reachable at {a.url}: {e}", file=sys.stderr); return 2
        except Exception:
            pass
        res = json.load(open(out)) if os.path.exists(out) else None
        if res and (res.get("layout"), res.get("haystack_sha256")) != (a.layout, hay_sha):
            print(f"{out} was made with another layout/haystack: use a new --out", file=sys.stderr); return 2
        res = res or dict(schema=1, created=time.strftime("%Y-%m-%dT%H:%M:%S"), label=a.label, url=a.url, model=a.model, layout=a.layout,
                          seed=SEED, sampling=GREEDY, haystack_sha256=hay_sha, haystack_tokens=len(hay_ids), packages=PACKAGES,
                          harness=dict(script="quality/longctx.py", sha256=sha256_file(os.path.abspath(__file__))),
                          server=c.info(), cases={})
        rec = res["cases"]
    total_est = 0
    for doc_id, ctx, ts in docs:
        cases = [x for t in ts for x in t["cases"] if selected(x["id"], a.only)]
        todo = [x for x in cases if a.dry_run or a.redo or x["id"] not in rec or rec[x["id"]].get("status")]
        if not todo:
            continue
        d = build_doc(ctx, ts, tok, hay_ids)
        est = est_seconds(ctx, d["tokens"], len(todo), a.cache); total_est += est
        if a.dry_run:
            print(f"\n[doc {doc_id}] ctx {ctx}: prefix {d['tokens']} tok (local {d['local']}), max suffix {d['st']}, max prompt "
                  f"{d['tokens'] + d['st']} <= {ctx - RESERVE}; haystack {d['hay']}/{len(hay_ids)} tok, {d['nfacts']} facts; "
                  f"{len(todo)} case(s), est {est:.0f} s (cache={a.cache})")
        if a.prefill_to and not a.dry_run:    # staged cold prefill (EXP-106): one prefix step per invocation, cases skipped
            ids = d["enc"](d["text"])[:a.prefill_to]
            text = tok.decode(ids); text = text[:text.rfind("\n") + 1]           # end on a line break: no cut character
            t0 = time.time(); c.complete(text, 1, timeout=a.timeout)
            log(f"prefill-to {a.prefill_to}: doc {doc_id} prefix {len(ids)} tok served in {time.time() - t0:.0f} s"); continue
        pre_ids = d["enc"](d["text"]) if a.dry_run else None
        warm = False
        for x in todo:
            prompt = d["text"] + d["suffix"][x["id"]]
            sha = hashlib.sha256(prompt.encode()).hexdigest()
            dep = [d["depth"][t] for t in x["tags"]]
            if a.dry_run:
                bnd = d["enc"](prompt)[:len(pre_ids)] == pre_ids                   # prefix tokens unchanged by the question?
                print(f"  {x['id']:30} {d['tokens'] + len(d['enc'](d['suffix'][x['id']])):>7} tok  fact@{min(dep):.3f}"
                      f"{'..%.3f' % max(dep) if len(dep) > 1 else ''}  ans {len(x['answers'])}  forbid {len(x['forbid'])}  "
                      f"prefix-stable={bnd}  sha {sha[:10]}")
                continue
            e = WARM_S if (a.cache and warm) else est_seconds(ctx, d["tokens"], 1, True)   # cold unless the prefix is already served
            if a.budget and time.time() - started + e > a.budget:
                log(f"budget {a.budget:.0f} s: next case {x['id']} (est {e:.0f} s) does not fit; re-run to continue"); return 3
            if rec.get(x["id"]) and rec[x["id"]].get("prompt_sha256") not in (None, sha):
                print(f"{x['id']}: prompt differs from the stored one (generator changed): use a new --out", file=sys.stderr); return 2
            r = dict(id=x["id"], ctx=ctx, type=x["type"], doc=doc_id, depth=dep, answers=x["answers"], prompt_sha256=sha,
                     prompt_tokens_est=d["tokens"] + len(d["enc"](d["suffix"][x["id"]])))
            try:
                g = c.complete(prompt, x["maxtok"], timeout=a.timeout)
                ok, sc, found, extras = score_case(x, g["text"])
                r.update(ok=ok, score=sc, found=found, extras=extras, answer=g["text"][:300], finish=g["finish"], seconds=g["seconds"],
                         usage=g["usage"], prompt_tokens_server=g["usage"].get("prompt_tokens"))
                if not warm and g["seconds"] < 0.25 * d["tokens"] / PREFILL_TPS.get(ctx, 600):
                    r["suspect_cache_hit"] = True      # first request of the document answered far faster than a cold prefill
                    log(f"WARNING {x['id']}: {g['seconds']} s << cold prefill: prefix served from a cache (same-format blocks only!)")
            except Exception as ex:
                r.update(ok=False, score=0.0, status="request_error", msg=str(ex)[:300])
            warm = True
            log(f"{x['id']}: {'OK' if r['ok'] else 'FAIL'} score={r['score']:.2f} ({r.get('seconds')} s) {r.get('answer', r.get('msg', ''))[:70]!r}")
            rec[x["id"]] = r
            with open(out + ".tmp", "w") as f:
                json.dump(res, f, indent=1)
            os.replace(out + ".tmp", out)                                        # atomic: a killed run keeps the finished cases
    if a.dry_run:
        print(f"\ntotal estimate {total_est:.0f} s (cache={a.cache}; prefill speeds from notebook EXP-009, warm case {WARM_S} s: estimates, not measures)")
        return 0
    by = {}
    for i, r in sorted(rec.items()):
        if selected(i, a.only):
            by.setdefault(i.rsplit("/", 1)[0], []).append(r["ok"])
    for k, v in by.items():
        print(f"{k:24} {sum(v)}/{len(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
