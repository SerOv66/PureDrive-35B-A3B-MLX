"""Deterministic, token-exact benchmark prompts : real code corpus + real coding task, chat template of the model.
prompt_ids(ctx, task_idx) -> list[int] of exactly ctx tokens = template head + corpus[:n] + "\n\n"+task + template tail.
Same token ids are fed to every runtime (MLX in-process, llama-server via token arrays)."""
import json, os, sys, functools

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR") or sys.exit("set MODEL_DIR to the model checkpoint directory (see docs/REPRODUCE.md)")
CTXS = [2048, 4096, 16384, 32768, 40960, 65536, 131072]
PH = "<<CORPUS>>"


@functools.lru_cache(None)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@functools.lru_cache(None)
def tasks():
    return json.load(open(os.path.join(HERE, "corpus", "tasks.json")))


@functools.lru_cache(None)
def corpus_ids():
    return tuple(tokenizer().encode(open(os.path.join(HERE, "corpus", "corpus.txt")).read(), add_special_tokens=False))


@functools.lru_cache(None)
def head_tail(task_idx):
    tok = tokenizer()
    msg = [{"role": "user", "content": PH + "\n\n" + tasks()[task_idx % len(tasks())]}]
    s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    head, tail = s.split(PH)
    return tuple(tok.encode(head, add_special_tokens=False)), tuple(tok.encode(tail, add_special_tokens=False))


def prompt_ids(ctx, task_idx=0):
    head, tail = head_tail(task_idx)
    n = ctx - len(head) - len(tail)
    c = corpus_ids()
    assert 0 < n <= len(c), (ctx, n, len(c))
    return list(head) + list(c[:n]) + list(tail)


def shared_prefix_len(ctx, task_idx=0):
    """Tokens shared by all tasks at this ctx (head + corpus slice): reusable prefix cache."""
    head, tail = head_tail(task_idx)
    return ctx - len(tail)


if __name__ == "__main__":
    print("corpus tokens:", len(corpus_ids()))
    for t in range(len(tasks())):
        h, tl = head_tail(t)
        print(f"task {t}: head {len(h)} tail {len(tl)} tokens")
    for ctx in CTXS:
        ids = prompt_ids(ctx, 0)
        assert len(ids) == ctx
    print("tail text task0:", repr(tokenizer().decode(list(head_tail(0)[1]))[-120:]))
    print("OK all ctx exact")
