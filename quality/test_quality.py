"""Unit tests for the quality suite. Uses only a local mock OpenAI-compatible server (no real model/server).
Run: runtimes/venv-mlx/bin/python -m unittest quality/test_quality.py -v   (from the workspace root)
"""
import hashlib, json, os, re, socket, subprocess, sys, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "bench"))
import run_quality as q  # noqa: E402
import compare as cmp  # noqa: E402

HE = q.load_plus("HumanEvalPlus.jsonl.gz")
MB = q.load_plus("MbppPlus.jsonl.gz")


class Mock:
    """Tiny OpenAI-compatible server in a thread. chat(body)->message dict, complete(body)->text."""

    def __init__(self, chat=None, complete=None):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _send(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            def do_GET(self):
                self._send({"data": [{"id": "mock"}]} if self.path == "/v1/models" else {"error": "nf"}, 200 if self.path == "/v1/models" else 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if self.path == "/v1/chat/completions":
                    msg = chat(body)
                    msg.setdefault("role", "assistant")
                    fin = "tool_calls" if msg.get("tool_calls") else "stop"
                    self._send({"choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
                elif self.path == "/v1/completions":
                    self._send({"choices": [{"index": 0, "text": complete(body), "finish_reason": "length"}],
                                "usage": {"prompt_tokens": len(body["prompt"]) // 4, "completion_tokens": 5}})
                else:
                    self._send({"error": "nf"}, 404)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


def tool_msg(calls, raw_args=None):
    return {"content": None, "tool_calls": [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": n, "arguments": raw_args if raw_args is not None else json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}


def has_tokenizer():
    try:
        import prompts
        prompts.tokenizer(); return True
    except Exception:
        return False


class TestParsing(unittest.TestCase):
    def test_extract_code_picks_block_with_entry_and_handles_truncation(self):
        txt = "Sure!\n```bash\nls\n```\n```python\nimport math\ndef f(x):\n    return x\n```\nUsage:\n```python\nprint(f(1))\n```"
        self.assertIn("def f", q.extract_code(txt, "f"))
        self.assertNotIn("print(f(1))", q.extract_code(txt, "f"))
        self.assertIn("def g", q.extract_code("```python\ndef g():\n    return 1", "g"))  # unterminated fence
        self.assertEqual(q.extract_code("def h(): pass", "h"), "def h(): pass")  # no fences
        self.assertIn("def f", q.extract_code("<think>x</think>```python\ndef f(): pass\n```", "f"))

    def test_build_program_humaneval_body_only_and_full_def(self):
        t = HE["HumanEval/0"]
        full = "```python\nfrom typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    return False\n```"
        self.assertTrue(q.build_program("humaneval", t, full).startswith(t["prompt"]))
        body = q.build_program("humaneval", t, "```python\n    return False\n```")
        compile(body, "x", "exec")  # prompt stub + indented body is a valid program
        self.assertTrue(body.endswith("    return False\n"))

    def test_parse_answer(self):
        P = q.parse_answer
        self.assertEqual(P("so 3+4=7\n#### 7")[0], 7)
        self.assertEqual(P("#### $1,234.50")[0], q.Decimal("1234.50"))
        self.assertEqual(P("#### 5 dollars\nbla\n#### 18")[0], 18)  # last marker wins
        self.assertEqual(P("The result is \\boxed{42}.")[1], "boxed")
        self.assertEqual(P("Therefore the answer is 12 apples.")[0], 12)
        self.assertEqual(P("we get 3 then 9")[1], "last_number")
        self.assertEqual(P("#### -4")[0], -4)
        self.assertEqual(P("no digits here"), (None, None))

    def test_subsets_are_deterministic_and_sized(self):
        he = q.subset_ids(HE, q.HE_STEP, "quick"); mb = q.subset_ids(MB, q.MBPP_STEP, "quick")
        self.assertEqual((len(he), len(mb)), (28, 29))
        self.assertEqual(he[:3], ["HumanEval/0", "HumanEval/6", "HumanEval/12"])
        self.assertEqual(len(q.subset_ids(HE, 6, "full")), 164); self.assertEqual(len(MB), 377)
        g = q.gsm_indices(1319, "quick"); self.assertEqual((len(g), len(set(g)), g[0]), (30, 30, 0))
        self.assertEqual(len(q.gsm_indices(1319, "full")), 200)

    def test_first_div(self):
        self.assertIsNone(q.first_div("abc", "abc")); self.assertEqual(q.first_div("abcd", "abXd"), 2)
        self.assertEqual(q.first_div("abc", "abcd"), 3)

    def test_needle_values_are_unique_and_deterministic(self):
        a, b = q.make_needle(32768, 0.5), q.make_needle(65536, 0.5)
        self.assertEqual(a, q.make_needle(32768, 0.5)); self.assertNotEqual(a[1], b[1])

    def test_tool_scoring(self):
        case = next(c for c in q.TOOL_CASES if c["id"] == "event")
        calls = lambda **a: [dict(name="create_event", arguments=a, valid_json=True)]
        self.assertTrue(q.score_tool_case(case, calls(title="Design review", date="2026-11-03", duration_minutes=45))[0])
        self.assertFalse(q.score_tool_case(case, calls(title="Design review", date="2026-11-03", duration_minutes="45"))[0])
        self.assertFalse(q.score_tool_case(case, calls(title="x", date="2026-11-03", duration_minutes=45))[0])
        self.assertFalse(q.score_tool_case(case, [])[0])
        self.assertFalse(q.score_tool_case(case, [dict(name="create_event", arguments=None, valid_json=False)])[0])
        for c in q.TOOL_CASES:  # every case is satisfiable by its own ideal calls
            ok, why = q.score_tool_case(c, [dict(name=n, arguments=a, valid_json=True) for n, a in q.ideal_calls(c)])
            self.assertTrue(ok, (c["id"], why))
        self.assertGreaterEqual(len(q.TOOL_CASES), 10)


class TestSandbox(unittest.TestCase):
    def run_code(self, ds, task, code, **kw):
        return q.check_solution(ds, task, code, **kw)

    def test_canonical_passes_including_special_oracles(self):
        for ds, db, tid in [("humaneval", HE, "HumanEval/0"), ("humaneval", HE, "HumanEval/32"), ("mbpp", MB, "Mbpp/2"),
                            ("mbpp", MB, "Mbpp/793"), ("mbpp", MB, "Mbpp/124")]:
            t = db[tid]
            code = (t["prompt"] if ds == "humaneval" else "") + t["canonical_solution"]
            r = self.run_code(ds, t, code)
            self.assertTrue(r["passed"], (tid, r))

    def test_wrong_solutions_fail(self):
        t = HE["HumanEval/0"]
        r = self.run_code("humaneval", t, "def has_close_elements(numbers, threshold):\n    return False\n")
        self.assertFalse(r["passed"]); self.assertEqual(r["status"], "ok")
        r = self.run_code("humaneval", t, "def has_close_elements(numbers, threshold):\n    raise ValueError('x')\n")
        self.assertFalse(r["passed"])
        self.assertEqual(self.run_code("humaneval", t, "def other(): pass\n")["status"], "load_error")
        self.assertEqual(self.run_code("humaneval", t, "def has_close_elements(:\n")["status"], "load_error")
        # passes base tests but not plus tests: base_pass True, plus_pass False
        m = MB["Mbpp/2"]
        r = self.run_code("mbpp", m, "def similar_elements(a, b):\n    return tuple(set(a) & set(b)) if a and b else (1,)\n")
        self.assertFalse(r["passed"]); self.assertTrue(r["base_pass"]); self.assertFalse(r["plus_pass"])

    def test_infinite_loop_times_out_fast(self):
        t = HE["HumanEval/0"]
        r = self.run_code("humaneval", t, "def has_close_elements(n, t):\n    while True: pass\n", cand_timeout=2)
        self.assertFalse(r["passed"]); self.assertEqual(r["status"], "timeout"); self.assertLess(r["total_seconds"], 12)
        r = self.run_code("humaneval", t, "x = 10 ** (10 ** 10)\ndef has_close_elements(n, t): return 1\n", cand_timeout=2)
        self.assertFalse(r["passed"]); self.assertLess(r["total_seconds"], 20)  # C-level hang killed by CPU rlimit/watchdog

    def test_memory_hog_is_killed(self):
        t = HE["HumanEval/0"]
        r = self.run_code("humaneval", t, "BIG = b'a' * (900 << 20)\nimport time; time.sleep(5)\ndef has_close_elements(n, t): return 1\n",
                          mem_mb=400)
        self.assertFalse(r["passed"]); self.assertEqual(r["status"], "memory"); self.assertLess(r["total_seconds"], 5)

    def test_no_network_and_no_writes_outside_tempdir(self):
        t = HE["HumanEval/0"]
        canon = t["canonical_solution"]
        net = ("import socket\ntry:\n    socket.create_connection(('93.184.216.34', 80), timeout=2); LEAK = True\n"
               "except BaseException:\n    LEAK = False\n" + t["prompt"] + canon.replace("    sorted_numbers", "    if LEAK: return None\n    sorted_numbers", 1))
        r = self.run_code("humaneval", t, net)
        self.assertTrue(r["passed"], r)  # connection refused/blocked => LEAK False => correct answer
        probe_dir = tempfile.mkdtemp(); probe = os.path.join(probe_dir, "probe.txt")
        wr = (f"try:\n    open({probe!r}, 'w').write('x')\nexcept BaseException:\n    pass\n" + t["prompt"] + canon)
        with tempfile.TemporaryDirectory() as td:
            if q._sandbox_prefix(td):  # sandbox-exec available: writes outside the temp dir must be denied
                self.assertTrue(self.run_code("humaneval", t, wr)["passed"])
                self.assertFalse(os.path.exists(probe))
            else:
                self.skipTest("sandbox-exec not available")
        os.rmdir(probe_dir) if not os.path.exists(probe) else (os.remove(probe), os.rmdir(probe_dir))

    def test_noisy_candidate_and_cleanup(self):
        before = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("qsuite_")}
        t = HE["HumanEval/0"]
        code = ("import sys\nprint('noise'); sys.stderr.write('err')\n" + t["prompt"] + t["canonical_solution"])
        self.assertTrue(self.run_code("humaneval", t, code)["passed"])
        self.assertFalse(self.run_code("humaneval", t, "import os\nos._exit(3)\n")["passed"])  # crash without result
        after = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("qsuite_")}
        self.assertEqual(before, after)


class TestPartsWithMock(unittest.TestCase):
    def test_tools(self):
        def chat(body):
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual((body["temperature"], body["top_k"], body["seed"]), (0, 1, q.SEED))
            self.assertTrue(body["tools"])
            user = body["messages"][-1]["content"]
            case = next(c for c in q.TOOL_CASES if c["user"] == user)
            calls = q.ideal_calls(case)
            if case["id"] == "weather": return {"content": "It is sunny."}  # no tool call
            if case["id"] == "calc": return tool_msg(calls, raw_args="{not json")  # invalid JSON
            if case["id"] == "timer": return tool_msg([("set_timer", {"seconds": 5})])  # wrong value
            if case["id"] == "stock": return tool_msg([("get_weather", {"city": "NVDA"})])  # wrong function
            if case["id"] == "no_args": return tool_msg([("list_tasks", {})], raw_args="{}")
            return tool_msg(calls)
        m = Mock(chat=chat)
        try:
            res = q.part_tools(q.Client(m.url, "mock"), "quick")
        finally:
            m.close()
        n = len(q.TOOL_CASES)
        self.assertEqual((res["n"], res["passed"]), (n, n - 4))
        failed = {i["id"] for i in res["items"] if not i["passed"]}
        self.assertEqual(failed, {"weather", "calc", "timer", "stock"})
        self.assertAlmostEqual(res["score"], (n - 4) / n)

    def test_reasoning(self):
        rows = q.load_gsm8k(); by_q = {r["question"]: r["answer"].split("####")[-1].strip().replace(",", "") for r in rows}
        seen = []

        def chat(body):
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            qtext = body["messages"][0]["content"].split("Problem: ", 1)[1]
            gold = int(by_q[qtext]); seen.append(gold)
            ans = gold if len(seen) % 3 else gold + 1  # every 3rd wrong
            return {"content": f"Let's compute.\n#### {ans}"}
        m = Mock(chat=chat)
        try:
            res = q.part_reasoning(q.Client(m.url, "mock"), "quick")
        finally:
            m.close()
        self.assertEqual(res["n"], 30); self.assertEqual(res["passed"], 20)

    def test_coding_end_to_end(self):
        sol = {}
        for ds, db, step in (("humaneval", HE, q.HE_STEP), ("mbpp", MB, q.MBPP_STEP)):
            for tid in q.subset_ids(db, step, "quick"):
                sol[(ds, tid)] = db[tid]
        bad = {"HumanEval/6", "HumanEval/12", q.subset_ids(MB, q.MBPP_STEP, "quick")[1]}
        body_only = "HumanEval/0"

        def chat(body):
            content = body["messages"][0]["content"]
            for (ds, tid), t in sol.items():
                key = t["prompt"] if ds == "humaneval" else t["prompt"].strip().removeprefix('"""').removesuffix('"""').strip()
                if key in content:
                    if tid in bad:
                        return {"content": f"```python\ndef {t['entry_point']}(*a, **k):\n    raise ValueError\n```"}
                    if tid == body_only:
                        return {"content": "```python\n" + t["canonical_solution"] + "\n```"}  # body only
                    code = (t["prompt"] if ds == "humaneval" else "") + t["canonical_solution"]
                    return {"content": "Here you go:\n```python\n" + code + "\n```\nDone."}
            raise AssertionError("unknown prompt")
        m = Mock(chat=chat)
        try:
            res = q.part_coding(q.Client(m.url, "mock"), "quick")
        finally:
            m.close()
        self.assertEqual(res["n"], 57)
        failed = {i["task_id"] for i in res["items"] if not i["passed"]}
        self.assertEqual(failed, bad & {t for (_, t) in sol})
        self.assertEqual(res["passed"], 57 - len(failed))
        self.assertTrue(all(i["status"] in ("ok", "load_error") for i in res["items"]))

    @unittest.skipUnless(has_tokenizer(), "tokenizer not available")
    def test_needle(self):
        import prompts
        miss = q.make_needle(4096, 0.9)[0]

        def complete(body):
            p = body["prompt"]
            label = re.search(r"code for vault (\S+)\? Reply", p).group(1)
            if label == miss: return "I could not find it."
            return re.search(rf"code for vault {re.escape(label)} is (\S+?)\.\n", p).group(1)
        m = Mock(complete=complete)
        try:
            res = q.part_needle(q.Client(m.url, "mock"), "full", ctxs=[2048, 4096], depths=[0.1, 0.5, 0.9])
        finally:
            m.close()
        self.assertEqual((res["n"], res["passed"]), (6, 5))
        self.assertEqual([i["prompt_tokens_local"] for i in res["items"]], [2048] * 3 + [4096] * 3)
        self.assertFalse(next(i for i in res["items"] if i["ctx"] == 4096 and i["depth"] == 0.9)["passed"])
        # prompt is raw text with the chat template, thinking off, and the needle really inside
        path, body = m.requests[0]
        self.assertEqual(path, "/v1/completions")
        self.assertTrue(body["prompt"].rstrip().endswith("<think>\n\n</think>"))
        self.assertEqual(body["top_k"], 1)
        retok = len(prompts.tokenizer().encode(body["prompt"], add_special_tokens=False))
        self.assertLessEqual(abs(retok - 2048), 4)  # text round-trip drift is 1-2 tokens, never beyond the context
        self.assertEqual(res["items"][0]["prompt_tokens_retok"], retok)

    @unittest.skipUnless(has_tokenizer(), "tokenizer not available")
    def test_identity_ref_compare_and_gate(self):
        import prompts
        variant = {"on": False}

        def complete(body):
            h = hashlib.sha1(body["prompt"].encode()).hexdigest()
            txt = (h + " lorem ipsum ") * 20
            if variant["on"] and "interval" in body["prompt"]:  # one prompt diverges in the candidate
                txt = txt[:37] + "X" + txt[38:]
            return txt
        m = Mock(complete=complete)
        try:
            ref = {"parts": {"identity": q.part_identity(q.Client(m.url, "mock"), "quick")}, "suite": "quick", "model": "ref"}
            self.assertEqual(len(ref["parts"]["identity"]["items"]), 6)
            self.assertIsNone(ref["parts"]["identity"]["score"])  # no --ref => nothing compared
            same = q.part_identity(q.Client(m.url, "mock"), "quick", ref=ref)
            self.assertEqual((same["passed"], same["compared"]), (6, 6))
            variant["on"] = True
            div = q.part_identity(q.Client(m.url, "mock"), "quick", ref=ref)
        finally:
            m.close()
        self.assertEqual(div["passed"], 5)
        bad = [i for i in div["items"] if i["vs_ref"] != "IDENTICAL"]
        self.assertEqual(len(bad), 1); self.assertTrue(bad[0]["vs_ref"].startswith("DIVERGE at char 37"))
        cand = {"parts": {"identity": div}, "suite": "quick", "model": "cand"}
        rules = cmp.load_thresholds(os.path.join(HERE, "THRESHOLDS.md"))["profiles"]
        rows, ok = cmp.compare(ref, cand, rules["lossless"])
        self.assertFalse(ok)
        self.assertEqual(dict((r[0], r[1]) for r in rows)["identity"], "FAIL")
        rows, ok = cmp.compare(ref, {"parts": {"identity": same}, "suite": "quick"}, rules["lossless"])
        self.assertTrue(ok)

    def test_cli_writes_full_json_and_merges_parts(self):
        def chat(body):
            case = next(c for c in q.TOOL_CASES if c["user"] == body["messages"][-1]["content"])
            return tool_msg(q.ideal_calls(case))
        m = Mock(chat=chat)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "r.json")
            try:
                r = subprocess.run([sys.executable, os.path.join(HERE, "run_quality.py"), "--url", m.url, "--model", "mock",
                                    "--part", "tools", "--suite", "quick", "--out", out], capture_output=True, text=True, timeout=120)
            finally:
                m.close()
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            d = json.load(open(out))
        me = os.path.join(HERE, "run_quality.py")
        self.assertEqual(d["harness"]["sha256"], hashlib.sha256(open(me, "rb").read()).hexdigest())
        self.assertEqual(set(d["datasets"]), set(q.DATA_FILES)); self.assertEqual(d["parts"]["tools"]["score"], 1.0)
        self.assertIn("tools", d["timings"]); self.assertEqual(d["suite"], "quick"); self.assertEqual(d["model"], "mock")
        self.assertEqual(d["sampling"]["temperature"], 0); self.assertEqual(d["sampling"]["top_k"], 1)

    def test_unreachable_server_exits_2(self):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, os.path.join(HERE, "run_quality.py"), "--url", f"http://127.0.0.1:{port}",
                                "--model", "x", "--part", "tools", "--out", os.path.join(td, "o.json")],
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)


def doc(suite="quick", **parts):
    return {"suite": suite, "model": "m", "parts": parts}


def part(passed, n, **kw):
    return dict(passed=passed, n=n, items=[dict(kw.get("mk", lambda i: {"id": i})(i), passed=i < passed) for i in range(n)])


class TestCompare(unittest.TestCase):
    rules = cmp.load_thresholds(os.path.join(HERE, "THRESHOLDS.md"))["profiles"]

    def run_cmp(self, profile, ref, cand):
        return dict((r[0], r[1]) for r in cmp.compare(ref, cand, self.rules[profile])[0]), cmp.compare(ref, cand, self.rules[profile])[1]

    def test_thresholds_file_parses(self):
        self.assertEqual(set(self.rules), {"lossless", "numeric"})
        self.assertEqual(self.rules["numeric"]["coding"]["max_drop"], 2); self.assertEqual(self.rules["numeric"]["tools"]["max_drop"], 1)
        self.assertEqual(self.rules["lossless"]["identity"]["min_identical"], 6)

    def test_numeric_profile_boundaries(self):
        mk_c = lambda i: {"task_id": f"t{i}"}; mk_g = lambda i: {"index": i}
        ref = doc(coding=part(50, 57, mk=mk_c), reasoning=part(25, 30, mk=mk_g), tools=part(14, 14))
        v, ok = self.run_cmp("numeric", ref, doc(coding=part(48, 57, mk=mk_c), reasoning=part(23, 30, mk=mk_g), tools=part(13, 14)))
        self.assertEqual((v["coding"], v["reasoning"], v["tools"], ok), ("PASS", "PASS", "PASS", True))
        v, ok = self.run_cmp("numeric", ref, doc(coding=part(47, 57, mk=mk_c), reasoning=part(25, 30, mk=mk_g), tools=part(14, 14)))
        self.assertEqual((v["coding"], ok), ("FAIL", False))
        v, ok = self.run_cmp("numeric", ref, doc(coding=part(50, 57, mk=mk_c), reasoning=part(22, 30, mk=mk_g), tools=part(14, 14)))
        self.assertEqual(v["reasoning"], "FAIL")
        v, ok = self.run_cmp("numeric", ref, doc(coding=part(50, 57, mk=mk_c), reasoning=part(25, 30, mk=mk_g), tools=part(12, 14)))
        self.assertEqual(v["tools"], "FAIL")
        self.assertEqual(v["identity"], "SKIP")

    def test_lossless_requires_equal_scores(self):
        mk = lambda i: {"id": i}
        ref = doc(tools=part(14, 14))
        self.assertEqual(self.run_cmp("lossless", ref, doc(tools=part(13, 14)))[0]["tools"], "FAIL")
        self.assertEqual(self.run_cmp("lossless", ref, doc(tools=part(14, 14)))[0]["tools"], "PASS")

    def test_needle_loss_fails_both_profiles(self):
        mk = lambda i: {"ctx": 1000 * (i + 1), "depth": 0.5}
        ref = doc(needle=part(3, 3, mk=mk))
        for prof in ("lossless", "numeric"):
            self.assertEqual(self.run_cmp(prof, ref, doc(needle=part(2, 3, mk=mk)))[0]["needle"], "FAIL")
            self.assertEqual(self.run_cmp(prof, ref, doc(needle=part(3, 3, mk=mk)))[0]["needle"], "PASS")
        weak = doc(needle=part(1, 3, mk=mk))  # ref misses cells: cand may match or exceed
        self.assertEqual(self.run_cmp("numeric", weak, doc(needle=part(2, 3, mk=mk)))[0]["needle"], "PASS")

    def test_not_comparable(self):
        with self.assertRaises(cmp.NotComparable):
            cmp.compare(doc("quick", tools=part(1, 1)), doc("full", tools=part(1, 1)), self.rules["numeric"])
        with self.assertRaises(cmp.NotComparable):
            cmp.compare(doc(tools=part(1, 2)), doc(tools=part(1, 3)), self.rules["numeric"])

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a.json"), os.path.join(td, "b.json")
            json.dump(doc(tools=part(14, 14)), open(a, "w")); json.dump(doc(tools=part(10, 14)), open(b, "w"))
            run = lambda x, y, *extra: subprocess.run([sys.executable, os.path.join(HERE, "compare.py"), x, y, *extra],
                                                      capture_output=True, text=True)
            self.assertEqual(run(a, a).returncode, 0)
            r = run(a, b, "--profile", "numeric"); self.assertEqual(r.returncode, 1); self.assertIn("FAIL", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
