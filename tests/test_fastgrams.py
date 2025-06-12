import fastgrams


# ---------------------------------------------------------------------------
# n-gram tokenisation --------------------------------------------------------
# ---------------------------------------------------------------------------

def test_ngram_tokenize_basic():
    english_sentence = "Hello,   World!  "
    toks = fastgrams.ngram_tokenize(english_sentence, n=1)
    # ICU NFKC_CaseFold will lower-case the text.
    assert toks == ["hello", "world"]


def test_ngram_tokenize_whitespace():
    whitespace_sentence = "\t Quick\n brown  \r fox  "
    toks = fastgrams.ngram_tokenize(whitespace_sentence, n=1)
    assert toks == ["quick", "brown", "fox"]


def test_ngram_tokenize_cjk():
    # "我爱机器学习" means "I love machine learning" in Chinese
    cjk_sentence = "我爱机器学习"
    toks = fastgrams.ngram_tokenize(cjk_sentence, n=1)
    # Each Han character should be treated as an individual token.
    assert toks == list(cjk_sentence)


def test_ngram_tokenize_bigrams():
    toks = fastgrams.ngram_tokenize("Hello world again", n=2)
    # Boundary character "#" joins adjacent tokens.
    assert toks == ["hello#world", "world#again"]


def test_ngram_tokenize_trigrams():
    toks = fastgrams.ngram_tokenize("one two three four", n=3)
    assert toks == ["one#two#three", "two#three#four"]


def test_ngram_tokenize_n_greater_than_tokens():
    assert fastgrams.ngram_tokenize("one two", n=3) == []
    assert fastgrams.ngram_tokenize("one two", n=2) == ["one#two"]


def test_ngram_tokenize_empty_string():
    assert fastgrams.ngram_tokenize("", n=1) == []


def test_ngram_tokenize_only_whitespace_and_punct():
    assert fastgrams.ngram_tokenize("!@#$%,.  \t\n", n=1) == []


def test_ngram_tokenize_with_punctuation():
    # Punctuation should be removed.
    toks = fastgrams.ngram_tokenize("a-b/c'd", n=1)
    assert toks == ["a", "b", "c", "d"]


def test_ngram_tokenize_mixed_scripts():
    # CJK characters are treated as tokens, Latin text is case-folded.
    toks = fastgrams.ngram_tokenize("I爱machine learning", n=1)
    assert toks == ["i", "爱", "machine", "learning"]


def test_ngram_tokenize_unicode_normalization():
    # Full-width Latin characters.
    toks = fastgrams.ngram_tokenize("Ｈｅｌｌｏ", n=1)
    assert toks == ["hello"]
    # Ligatures
    toks = fastgrams.ngram_tokenize("\uFB03", n=1)  # ffi ligature
    assert toks == ["ffi"]


# ---------------------------------------------------------------------------
# Char-trigram tokenisation --------------------------------------------------
# ---------------------------------------------------------------------------

def _ct(s):
    """Helper that returns char-trigram tokens converted to UTF-8 strings."""
    return fastgrams.char_trigram_tokenize(s)


def test_char_trigram_basic():
    # "cat" – sentinel # at both ends → [#ca, cat, at#]
    assert _ct("cat") == ["#ca", "cat", "at#"]


def test_char_trigram_short_strings():
    assert _ct("a") == ["#a#"]
    assert _ct("ab") == ["#ab", "ab#"]


def test_char_trigram_empty_string():
    assert _ct("") == []


def test_char_trigram_whitespace_bridging():
    """Ensure that trigram generation correctly handles word boundaries."""
    tokens = _ct("hello world")
    expected = [
        # hello
        "#he",
        "hel",
        "ell",
        "llo",
        "lo#",
        # bridging
        "o#w",
        # world
        "#wo",
        "wor",
        "orl",
        "rld",
        "ld#",
    ]
    assert tokens == expected


def test_char_trigram_multiple_whitespace():
    # Multiple whitespace characters should be treated as a single boundary.
    # The C++ implementation iterates over all characters including whitespace,
    # which creates bridging trigrams. Consecutive whitespace chars do not
    # create multiple boundaries because of the `cq.back() != kBoundary` check.
    expected = ["#a#", "a#b", "#b#"]
    assert _ct("a b") == expected
    assert _ct("a  b") == expected
    assert _ct("a \t b") == expected


def test_char_trigram_with_punctuation():
    # Punctuation is treated as a regular character in trigram tokenization.
    assert _ct("a,b") == ["#a,", "a,b", ",b#"]


def test_char_trigram_cjk_mixed():
    # CJK characters are not treated specially in char-trigram generation.
    assert _ct("a我b") == ["#a我", "a我b", "我b#"]


def test_char_trigram_cjk():
    # "我爱机器学习" means "I love machine learning" in Chinese
    cjk_sentence = "我爱机器学习"
    # A sequence of CJK characters should be wrapped with sentinels as well.
    # e.g. "我" → [#我爱, 我爱机...] but to keep test simple we just check count.
    tris = _ct(cjk_sentence)
    expected = ['#我爱', '我爱机', '爱机器', '机器学', '器学习', '学习#']
    assert tris == expected


# ---------------------------------------------------------------------------
# Counting helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def test_ngram_counts():
    counts_py = fastgrams.ngram_counts(["hello world", "hello there"], n=1)
    counts = {k: int(v) for k, v in counts_py.items()}
    expected = {"hello": 2, "world": 1, "there": 1}
    assert counts == expected


def test_ngram_counts_empty():
    assert fastgrams.ngram_counts([]) == {}
    assert fastgrams.ngram_counts(["", "  ", ".,!"]) == {}


def test_char_trigram_counts():
    counts_py = fastgrams.char_trigram_counts(["ab", "ab"])
    counts = {k: int(v) for k, v in counts_py.items()}
    expected_tokens = ["#ab", "ab#"]  # For a 2-letter word there are 2 trigrams
    expected = {t: 2 for t in expected_tokens}
    assert counts == expected


def test_char_trigram_counts_empty():
    assert fastgrams.char_trigram_counts([]) == {}
    assert fastgrams.char_trigram_counts(["", " "]) == {}


# ---------------------------------------------------------------------------
# Vocabulary-aware tokenisers ----------------------------------------------
# ---------------------------------------------------------------------------

def test_vocab_ngram_tokenizer():
    vocab = {"hello": 1, "world": 2}
    tok = fastgrams.VocabNgramTokenizer(vocab, n=1)

    arrays = tok.tokenize(["hello world", "unknown"], default=-1)
    # First sentence → [1, 2]
    assert list(arrays[0]) == [1, 2]
    # Second sentence contains unknown token, should map to default.
    assert list(arrays[1]) == [-1]


def test_vocab_ngram_tokenizer_skip_unknown():
    vocab = {"hello": 1, "world": 2}
    tok = fastgrams.VocabNgramTokenizer(vocab, n=1)
    # With no default, unknown tokens should be skipped.
    arrays = tok.tokenize(["hello world", "unknown"], default=None)
    assert list(arrays[0]) == [1, 2]
    assert list(arrays[1]) == []


def test_vocab_ngram_tokenizer_empty_input():
    tok = fastgrams.VocabNgramTokenizer({"a": 1}, n=1)
    assert tok.tokenize([]) == []
    # An empty string produces an empty array of tokens.
    arrs = tok.tokenize([""])
    assert len(arrs) == 1
    assert len(arrs[0]) == 0


def test_vocab_char_trigram_tokenizer():
    # Build vocab from tokens of "cat"
    tris = fastgrams.char_trigram_tokenize("cat")
    vocab = {t: i for i, t in enumerate(tris)}

    tok = fastgrams.VocabCharTrigramTokenizer(vocab)
    arrays = tok.tokenize(["cat", "dog"], default=-1)

    assert list(arrays[0]) == list(range(len(tris)))
    # "dog" has trigrams not in vocab → all default.
    assert all(v == -1 for v in arrays[1])


def test_vocab_char_trigram_tokenizer_skip_unknown():
    # Build vocab from tokens of "cat"
    tris = fastgrams.char_trigram_tokenize("cat")
    vocab = {t: i for i, t in enumerate(tris)}

    tok = fastgrams.VocabCharTrigramTokenizer(vocab)
    arrays = tok.tokenize(["cat", "dog"], default=None)

    assert list(arrays[0]) == list(range(len(tris)))
    # "dog" has trigrams not in vocab -> all skipped.
    assert list(arrays[1]) == []


def test_vocab_char_trigram_tokenizer_empty_input():
    tok = fastgrams.VocabCharTrigramTokenizer({"#a#": 1})
    assert tok.tokenize([]) == []
    arrs = tok.tokenize([""])
    assert len(arrs) == 1
    assert len(arrs[0]) == 0