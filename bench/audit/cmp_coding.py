"""Compare greedy coding outputs of run_quality.py JSONs pairwise (response text + token count per task_id)."""
import json, sys


def items(p):
    return {i["task_id"]: i for i in json.load(open(p))["parts"]["coding"]["items"]}


for pair in sys.argv[1:]:
    a, b = pair.split(":")
    A, B = items(a), items(b)
    same = [k for k in A if k in B and A[k]["response"] == B[k]["response"] and A[k]["completion_tokens"] == B[k]["completion_tokens"]]
    passed = (sum(bool(i["passed"]) for i in A.values()), sum(bool(i["passed"]) for i in B.values()))
    diff = [k for k in A if k not in same]
    print(f"{a} vs {b}: {len(same)}/{len(A)} IDENTICAL (n_b={len(B)}), passed {passed[0]}/{passed[1]}" + (f"  DIFF: {diff[:6]}" if diff else ""))
