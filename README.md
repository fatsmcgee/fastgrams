# Sequen Extension Libraries

* This  contains Python extension packages (those that require compiling native code), which
currently consists of just `fastgrams`.

## Building/Installing This Package

* Activate your Python environment of choice
* `uv pip install -e .` (I believe this would work with vanilla pip too, but didn't test)



## Fastgrams

Fastgrams is a hyper fast ngram tokenizer, which has virtually identical logic to Pinterest's [OmniSearchSage](https://github.com/pinterest/atg-research/tree/main/omnisearchsage)

### Text cleaning and normalization 
  - All input text is [Unicode NFKC normalized](https://unicode.org/reports/tr15/#NFKC_Compatibility_Decomposition), ensuring consistent character representation.
  - All text is lowercased before tokenization.
  - Han (CJK) characters are treated as distinct tokens, separated from surrounding text.
  - Runs of punctuation are collapsed and treated as token boundaries.
  - Whitespace is trimmed and normalized.
  - These steps ensure robust, language-agnostic tokenization and consistent handling of multilingual input.



### Usage examples

```python
import fastgrams as fg

# --------------------------------------------------------------
# 1) Tokenise strings into unigrams / bigrams
# --------------------------------------------------------------
strings = ["hello world"]

# Unigrams only
print(fg.ngram_tokenize(strings))
# -> [['hello', 'world']]

# Unigrams + bigrams
print(fg.ngram_tokenize(strings, include_bigrams=True))
# -> ([['hello', 'world']], [['hello#world']])

# --------------------------------------------------------------
# 2) Frequency counts
# --------------------------------------------------------------
print(fg.ngram_counts(["hello world hello"]))
# -> {'hello': 2, 'world': 1}

print(fg.ngram_counts(["hello world hello"], include_bigrams=True))
# -> ({'hello': 2, 'world': 1}, {'hello#world': 1, 'world#hello': 1})

# --------------------------------------------------------------
# 3) Vocabulary-aware n-gram tokeniser
# --------------------------------------------------------------
uni_vocab = {'hello': 1, 'world': 2}
bi_vocab  = {'hello#world': 5}

tok = fg.VocabNgramTokenizer(uni_vocab, bi_vocab)
print(tok.tokenize(["hello world unknown"]))
# -> [array([ 1,  2, -1])]

print(tok.tokenize(["hello world"], include_bigrams=True))
# -> ([array([1, 2])], [array([5])])

# --------------------------------------------------------------
# 4) Character trigram helpers
# --------------------------------------------------------------
print(fg.char_trigram_tokenize(["hello"]))
# -> [['#he', 'hel', 'ell', 'llo', 'lo#']]

print(fg.char_trigram_counts(["hello"]))
# -> {'#he': 1, 'hel': 1, 'ell': 1, 'llo': 1, 'lo#': 1}

# --------------------------------------------------------------
# 5) Vocabulary-aware character trigram tokeniser
# --------------------------------------------------------------
tg_vocab = {'#he': 1, 'hel': 2, 'ell': 3}
ctok = fg.VocabCharTrigramTokenizer(tg_vocab)
print(ctok.tokenize(["hello"]))
# -> [array([ 1,  2,  3, -1, -1])]
```

