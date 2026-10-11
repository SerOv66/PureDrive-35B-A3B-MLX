"""256K harness (256K addendum): prompts up to 262,144 tokens without touching the frozen bench files.
Corpus = frozen bench/corpus/corpus.txt (236,629 tokens, CPython 3.12.13 stdlib asyncio/concurrent/http/json/logging)
+ bench/corpus/corpus_ext_256k.txt = CPython 3.12.13 stdlib multiprocessing/ (all .py, alphabetical), same file format
(header line + file + "\\n", rule checked byte-exact against corpus.txt). Template, tasks and tokenizer are those of
bench/prompts.py; the frozen corpus tokens come first, so for ctx <= 236K the prompts are identical to the frozen ones
and the 131K prompt is a shared prefix of the 256K one (prefix cache).
Usage: runtimes/venv-mlx/bin/python bench/prompts_256k.py build    # writes the extension once (sha256 pinned below)
       runtimes/venv-mlx/bin/python bench/prompts_256k.py check    # exact lengths, prefix identity with bench/prompts.py
       runtimes/venv-mlx/bin/python bench/prompts_256k.py bench <bench_http.py args, e.g. --ctx 262144 ...>
         (frozen client bench/bench_http.py unchanged, only its prompt source swapped: same sampling, timing, RESULTS rows)"""
import functools, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts

EXT = os.path.join(HERE, "corpus", "corpus_ext_256k.txt")
EXT_SHA256 = "ea6aca1452be39bfd2ecbc61d48308e3d79aace6f6336c0b67f586f9ae5ed290"
STDLIB = os.environ.get("CPYTHON_STDLIB", "")  # lib/python3.12 of CPython 3.12.13; only needed for `build`
CTXS = [196608, 262144]


def build():
    assert STDLIB, "set CPYTHON_STDLIB to the lib/python3.12 directory of CPython 3.12.13"
    root = os.path.join(STDLIB, "multiprocessing")
    files = sorted(os.path.relpath(os.path.join(d, f), STDLIB) for d, _, fs in os.walk(root) for f in fs if f.endswith(".py"))
    text = "".join(f"# ===== file: {f} =====\n{open(os.path.join(STDLIB, f)).read()}\n" for f in files)
    open(EXT, "w").write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"{len(files)} files, {len(text)} chars, sha256 {digest}")
    assert digest == EXT_SHA256, "not the pinned CPython 3.12.13 multiprocessing sources"


@functools.lru_cache(None)
def corpus_ids():
    text = open(EXT).read()
    assert EXT_SHA256 is None or hashlib.sha256(text.encode()).hexdigest() == EXT_SHA256, "corpus_ext_256k.txt changed"
    return prompts.corpus_ids() + tuple(prompts.tokenizer().encode(text, add_special_tokens=False))


def prompt_ids(ctx, task_idx=0):
    head, tail = prompts.head_tail(task_idx)
    n = ctx - len(head) - len(tail)
    c = corpus_ids()
    assert 0 < n <= len(c), (ctx, n, len(c))
    return list(head) + list(c[:n]) + list(tail)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "build":
        build()
    elif cmd == "check":
        print("corpus tokens: frozen", len(prompts.corpus_ids()), "extended", len(corpus_ids()))
        for ctx in prompts.CTXS:
            for t in range(3):
                assert prompt_ids(ctx, t) == prompts.prompt_ids(ctx, t), (ctx, t)
        for ctx in CTXS:
            for t in range(3):
                ids = prompt_ids(ctx, t)
                assert len(ids) == ctx
        k = prompts.shared_prefix_len(131072, 0)
        assert prompt_ids(262144, 0)[:k] == prompts.prompt_ids(131072, 0)[:k]
        print(f"OK: frozen ctx identical to bench/prompts.py, {CTXS} exact, first {k} tokens shared with the 131K prompt")
    elif cmd == "bench":
        prompts.prompt_ids = prompt_ids
        import bench_http
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        bench_http.main()
