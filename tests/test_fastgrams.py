import fastgrams as fg


# ---------------------------------------------------------------------------
# n-gram tokenisation --------------------------------------------------------
# ---------------------------------------------------------------------------


def test_vocab_ngram_tokenizer():
# Simple sanity-check for the VocabNgramTokenizer
    strings = [
        "dog cat mouse fish",  # 4 tokens -> 3 bigrams
        "goat herder"           # 2 tokens -> 1 bigram
    ]

    # Build vocabularies from the sample data
    unigram_counts, bigram_counts = fg.ngram_counts(strings, include_bigrams=True)
    unigram_vocab = {tok: idx for idx, tok in enumerate(unigram_counts)}
    bigram_vocab = {tok: idx for idx, tok in enumerate(bigram_counts)}

    tokenizer = fg.VocabNgramTokenizer(unigram_vocab, bigram_vocab)
    uni_ids, bi_ids = tokenizer.tokenize(strings, include_bigrams=True)

    print("Input strings:", strings)
    print("Unigram IDs:", uni_ids)
    print("Bigram IDs:", bi_ids)

def test_char_trigram_counts_basic():
    """Ensure char_trigram_counts correctly tallies trigram occurrences."""
    counts = fg.char_trigram_counts(["ab", "ab"])
    expected_tokens = ["#ab", "ab#"]  # A 2-char word yields 2 trigrams
    expected = {t: 2 for t in expected_tokens}

    assert counts == expected


def test_vocab_char_trigram_tokenizer_active():
    """Sanity-check VocabCharTrigramTokenizer mapping behaviour."""
    # Build a small vocabulary from the trigrams of "cat"
    tris = fg.char_trigram_tokenize(["cat"])[0]
    vocab = {tok: idx for idx, tok in enumerate(tris)}

    tokenizer = fg.VocabCharTrigramTokenizer(vocab)

    # Two inputs: one fully covered by vocab, one containing only unknown tokens
    arrays = tokenizer.tokenize(["cat", "dog"])

    # First string → sequential IDs 0..len(tris)-1
    assert list(arrays[0]) == list(range(len(tris)))

    # Second string → all OOV, hence default (-1)
    assert all(v == -1 for v in arrays[1])

def test_ngram_tokenize_basic():
    english_sentence = "Hello,   World!  "
    toks = fg.ngram_tokenize([english_sentence])[0]
    # ICU NFKC_CaseFold will lower-case the text.
    assert toks == ["hello", "world"]


def test_ngram_tokenize_whitespace():
    whitespace_sentence = "\t Quick\n brown  \r fox  "
    toks = fg.ngram_tokenize([whitespace_sentence])[0]
    assert toks == ["quick", "brown", "fox"]


def test_ngram_tokenize_cjk():
    # "我爱机器学习" means "I love machine learning" in Chinese
    cjk_sentence = "我爱机器学习"
    toks = fg.ngram_tokenize([cjk_sentence])[0]
    # Each Han character should be treated as an individual token.
    assert toks == list(cjk_sentence)


def test_ngram_tokenize_bigrams():
    unigrams, bigrams = fg.ngram_tokenize(["Hello world again"], include_bigrams=True)
    # Boundary character "#" joins adjacent tokens.
    assert bigrams[0] == ["hello#world", "world#again"]


def test_ngram_tokenize_n_greater_than_tokens():
    # For bigrams, if not enough tokens, should return empty list
    unigrams, bigrams = fg.ngram_tokenize(["one"], include_bigrams=True)
    assert bigrams[0] == []
    # For two tokens, should return one bigram
    unigrams, bigrams = fg.ngram_tokenize(["one two"], include_bigrams=True)
    assert bigrams[0] == ["one#two"]


def test_ngram_tokenize_empty_string():
    toks = fg.ngram_tokenize([""])[0]
    assert toks == []


def test_ngram_tokenize_only_whitespace_and_punct():
    toks = fg.ngram_tokenize(["!@#%,.  \t\n"])[0]
    assert toks == []


def test_ngram_tokenize_with_punctuation():
    # Punctuation should be removed.
    toks = fg.ngram_tokenize(["a-b/c'd"])[0]
    assert toks == ["a", "b", "c", "d"]


def test_ngram_tokenize_mixed_scripts():
    # CJK characters are treated as tokens, Latin text is case-folded.
    toks = fg.ngram_tokenize(["I爱machine learning"])[0]
    assert toks == ["i", "爱", "machine", "learning"]


def test_ngram_tokenize_unicode_normalization():
    # Full-width Latin characters.
    toks = fg.ngram_tokenize(["Ｈｅｌｌｏ"])[0]
    assert toks == ["hello"]
    # Ligatures
    toks = fg.ngram_tokenize(["\uFB03"])[0]  # ffi ligature
    assert toks == ["ffi"]


# ---------------------------------------------------------------------------
# Char-trigram tokenisation --------------------------------------------------
# ---------------------------------------------------------------------------

def _ct_single_str(s: str):
    return fg.char_trigram_tokenize([s])[0]


def test_char_trigram_basic():
    # "cat" – sentinel # at both ends → [#ca, cat, at#]
    assert _ct_single_str("cat") == ["#ca", "cat", "at#"]


def test_char_trigram_short_strings():
    assert _ct_single_str("a") == ["#a#"]
    assert _ct_single_str("ab") == ["#ab", "ab#"]


def test_char_trigram_empty_string():
    assert _ct_single_str("") == []


def test_char_trigram_whitespace_bridging():
    tokens = _ct_single_str("hello world")
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
    assert _ct_single_str("a b") == expected
    assert _ct_single_str("a  b") == expected
    assert _ct_single_str("a \t b") == expected


def test_char_trigram_with_punctuation():
    # Punctuation is treated as a regular character in trigram tokenization.
    assert _ct_single_str("a,b") == ["#a,", "a,b", ",b#"]


def test_char_trigram_cjk_mixed():
    # CJK characters are not treated specially in char-trigram generation.
    assert _ct_single_str("a我b") == ["#a我", "a我b", "我b#"]


def test_char_trigram_cjk():
    # "我爱机器学习" means "I love machine learning" in Chinese
    cjk_sentence = "我爱机器学习"
    # A sequence of CJK characters should be wrapped with sentinels as well.
    # e.g. "我" → [#我爱, 我爱机...] but to keep test simple we just check count.
    tris = _ct_single_str(cjk_sentence)
    expected = ['#我爱', '我爱机', '爱机器', '机器学', '器学习', '学习#']
    assert tris == expected


# ---------------------------------------------------------------------------
# Counting helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def test_ngram_counts():
    counts = fg.ngram_counts(["hello world", "hello there"])
    expected = {"hello": 2, "world": 1, "there": 1}
    assert counts == expected


def test_ngram_counts_empty():
    assert fg.ngram_counts([]) == {}
    assert fg.ngram_counts(["", "  ", ".,!"]) == {}


def test_char_trigram_counts():
    counts = fg.char_trigram_counts(["ab", "ab"])
    expected_tokens = ["#ab", "ab#"]  # For a 2-letter word there are 2 trigrams
    expected = {t: 2 for t in expected_tokens}
    assert counts == expected


def test_char_trigram_counts_empty():
    assert fg.char_trigram_counts([]) == {}
    assert fg.char_trigram_counts(["", "     "]) == {}


# ---------------------------------------------------------------------------
# Vocabulary-aware tokenisers ----------------------------------------------
# ---------------------------------------------------------------------------

def test_vocab_ngram_tokenizer():
    vocab = {"hello": 1, "world": 2}
    tok = fg.VocabNgramTokenizer(vocab)

    arrays = tok.tokenize(["hello world", "unknown"])
    # First sentence → [1, 2]
    assert list(arrays[0]) == [1, 2]
    # Second sentence contains unknown token, should map to default.
    assert list(arrays[1]) == [-1]


def test_vocab_ngram_tokenizer_skip_unknown():
    vocab = {"hello": 1, "world": 2}
    tok = fg.VocabNgramTokenizer(vocab)
    # With no default, unknown tokens should be filled with default
    arrays = tok.tokenize(["hello world", "unknown"])
    assert list(arrays[0]) == [1, 2]
    assert list(arrays[1]) == [-1]


def test_vocab_ngram_tokenizer_empty_input():
    tok = fg.VocabNgramTokenizer({"a": 1})
    assert tok.tokenize([]) == []
    # An empty string produces an empty array of tokens.
    arrs = tok.tokenize([""])
    assert len(arrs) == 1
    assert len(arrs[0]) == 0


def test_vocab_char_trigram_tokenizer():
    # Build vocab from tokens of "cat"
    tris = _ct_single_str("cat")
    vocab = {t: i for i, t in enumerate(tris)}

    tok = fg.VocabCharTrigramTokenizer(vocab)
    arrays = tok.tokenize(["cat", "dog"])

    assert list(arrays[0]) == list(range(len(tris)))
    # "dog" has trigrams not in vocab → all default.
    assert all(v == -1 for v in arrays[1])


def test_vocab_char_trigram_tokenizer_empty_input():
    tok = fg.VocabCharTrigramTokenizer({"#a#": 1})
    assert tok.tokenize([]) == []
    arrs = tok.tokenize(["     "])
    assert len(arrs) == 1
    assert len(arrs[0]) == 0
