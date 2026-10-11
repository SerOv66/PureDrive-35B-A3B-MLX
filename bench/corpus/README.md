# Benchmark corpus (frozen)

* `corpus.txt` (not shipped): CPython 3.12.13 standard-library sources (`asyncio/`, `concurrent/futures/`, `http/`, `json/`, `logging/`), alphabetical by package and path, header line `# ===== file: <path> =====` before each file; 50 files, 1,026,676 bytes, 236,629 tokens; sha256 `bbb8cfe27b733248866ce3df129f9d06dcb5c1f6331204d0c2688ee0faeab6a5`. License: PSF. Rebuild byte-exactly with `CPYTHON_STDLIB=<.../lib/python3.12> python bench/corpus/build_corpus.py`.
* `corpus_ext_256k.txt` (not shipped): the same format for `multiprocessing/`, used only by the 256K harness; sha256 `ea6aca1452be39bfd2ecbc61d48308e3d79aace6f6336c0b67f586f9ae5ed290`; rebuilt by `python bench/prompts_256k.py build`.
* `tasks.json`: the three real coding instructions appended after the corpus (run i uses task i mod 3).
* A prompt = the model's chat template (user turn: corpus truncated to the exact token count + `\n\n` + task) with generation prompt; length exact in tokens (`bench/prompts.py`). The same token ids are fed to every runtime.
