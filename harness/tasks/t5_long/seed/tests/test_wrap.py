import importlib.util, pathlib
_p = pathlib.Path(__file__).resolve().parents[1] / "lib" / "textwrap.py"
_s = importlib.util.spec_from_file_location("textwrap_lib", _p); tw = importlib.util.module_from_spec(_s); _s.loader.exec_module(tw)


def test_simple_wrap():
    assert tw.wrap("the quick brown fox jumps over the lazy dog", 15) == ["the quick brown", "fox jumps over", "the lazy dog"]


def test_long_word_break():
    assert tw.wrap("a" * 25, 10) == ["a" * 10, "a" * 10, "a" * 5]


def test_break_on_hyphens():
    assert tw.wrap("abcd-efghijklmn", 10) == ["abcd-", "efghijklmn"]
    assert tw.wrap("well-known state-of-the-art", 12) == ["well-known", "state-of-", "the-art"]


def test_shorten_and_dedent():
    assert tw.shorten("Hello  world, this is a test", width=17) == "Hello [...]"
    assert tw.dedent("   a\n    b\n   c") == "a\n b\nc"


def test_fill_indent():
    assert tw.fill("x " * 8, width=9, initial_indent="> ", subsequent_indent="  ") == "> x x x x\n  x x x x"
