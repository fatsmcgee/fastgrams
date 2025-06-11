import numpy as np
from ._fastgrams import (
    ngram_tokenize,
    char_trigram_tokenize,
    ngram_counts,
    char_trigram_counts,
    VocabNgramTokenizer,
    VocabCharTrigramTokenizer
)



_MASK_21 = (1 << 21) - 1


def packed_trigram_to_string(packed: int) -> str:
    """Inverse of :func:`string_to_packed_trigrams` for a single packed trigram (mostly for debugging)."""

    code_int = int(packed)
    a = code_int & _MASK_21
    b = (code_int >> 21) & _MASK_21
    c = (code_int >> 42) & _MASK_21
    arr = np.array([a, b, c], dtype=np.uint32)
    return arr.tobytes().decode("utf-32-le")


# Re-export helper names so they are accessible as fastgrams.string_to_packed_trigrams
__all__ = [
    "ngram_tokenize",
    "char_trigram_tokenize",
    "ngram_counts",
    "char_trigram_counts",
    "VocabNgramTokenizer",
    "VocabCharTrigramTokenizer",
    "packed_trigram_to_string",
]