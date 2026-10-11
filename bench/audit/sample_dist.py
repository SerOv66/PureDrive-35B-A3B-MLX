"""AUDIT EXP-105: N sampled completions (T 0.6, top-p 0.95, top-k 20 = verifier 'default') of one fixed high-entropy raw
prompt, seed = i, saved as JSON with the generated token ids (re-tokenized text) for a two-sample chi-square
MTP-on (PREDRAFT) vs MTP-off. Usage: python sample_dist.py <url> <n> <max_tokens> <out.json>"""
import json, sys, time, urllib.request

url, n, mt, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
PROMPT = ("Below are fifty unrelated English nouns picked uniformly at random from a dictionary, separated by commas.\n\n"
          "Nouns: lantern, glacier, receipt, falcon, pudding,")
res = []; t0 = time.time()
for i in range(n):
    body = dict(model="coder-q6", prompt=PROMPT, max_tokens=mt, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=i, stream=False)
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        j = json.loads(r.read())
    res.append(dict(seed=i, text=j["choices"][0]["text"], n=j.get("usage", {}).get("completion_tokens")))
json.dump(dict(prompt=PROMPT, max_tokens=mt, items=res, seconds=time.time() - t0), open(out, "w"))
print(f"{n} samples in {time.time() - t0:.1f} s -> {out}")
