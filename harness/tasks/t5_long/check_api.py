"""Check API.md: for every module in lib/ (via ast; if __all__ exists, only those names) the '## module' section must mention >=90% of the public top-level names."""
import ast, pathlib, re, sys
d = pathlib.Path(sys.argv[1]); api = (d / "API.md").read_text() if (d / "API.md").exists() else sys.exit("API.md missing")
secs = {m.group(1).strip(): m.group(2) for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", api, re.M | re.S)}
bad = 0
for f in sorted((d / "lib").glob("*.py")):
    tree = ast.parse(f.read_text())
    allv = next((set(ast.literal_eval(n.value)) for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in n.targets)), None)
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_") and (allv is None or n.name in allv)]
    sec = next((v for k, v in secs.items() if k.strip("`") .split(".")[0] == f.stem), "")
    hit = sum(1 for n in names if re.search(r"\b" + re.escape(n) + r"\b", sec))
    ok = names == [] or hit / len(names) >= 0.9
    print(f"{f.stem:12} {hit}/{len(names)} {'ok' if ok else 'TOO FEW'}"); bad += not ok
sys.exit(bad)
