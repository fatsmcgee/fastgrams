import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as patypes
from typing import Iterable, Any
from ._fastgrams import (
    ArrowStringArrayMapper,
    Uint64Mapper,
)

# Captures single Han character
_RE_HAN_SINGLE = r"(\p{Han})"
# Captures 1 or more unicode puncutation characters
_RE_PUNCT = r"\p{P}+"
_WHITESPACE_RE = r"\s+"

def extract_bigrams_from_unigrams_pa(unigrams: pa.ListArray) -> pa.ListArray:
    """
    Given a ListArray of unigrams (essentially List[List[str]]), create a ListArray of bigrams

    Parameters
    ----------
    tokens : pa.ListArray of strings
        A PyArrow ListArray where each element is a list of tokens (strings).
        Example: [['a', 'b', 'c'], ['d', 'e']]

    Returns
    -------
    pa.ListArray of String
        A PyArrow ListArray where each element is a list of bigrams.
        Example: [['a#b', 'b#c'], ['d#e']]
    """

    # Get flattened 1-d tokens list and 1-d row indices
    flat_tokens = pc.list_flatten(unigrams)
    parent_indices = pc.list_parent_indices(unigrams)
    assert len(flat_tokens) == len(parent_indices), "Flat tokens and parent indices must have same length"

    # Create adjacent token pairs
    n_tokens = len(flat_tokens)
    left = flat_tokens.slice(0, n_tokens - 1)
    right = flat_tokens.slice(1)
    indices_left = parent_indices.slice(0, n_tokens - 1)
    indices_right = parent_indices.slice(1)
    assert len(left) == len(right), "Left and right slices for bigrams must be same length"

    # Filter adjacent token pairs to those that share the same original rows
    same_parent_mask = pc.equal(indices_left, indices_right)
    left_filtered = pc.filter(left, same_parent_mask)
    right_filtered = pc.filter(right, same_parent_mask)

    # Join adjacent tokens into bigram strings
    bigrams_flat = pc.binary_join_element_wise(left_filtered, right_filtered, "#")

    # Reconstruct the list array structure
    # First, calculate the number of bigrams for each original list,
    # which is (n_unigrams - 1) unless there is 1 or fewer unigram
    lengths = pc.list_value_length(unigrams)
    bigram_counts = pc.subtract(lengths, 1)
    bigram_counts = pc.if_else(pc.less(bigram_counts, 0), 0, bigram_counts)

    # Then, calculate the offsets for the new ListArray from these counts.
    # The offsets are the cumulative sum of counts, prepended with a 0.
    cumulative_counts = pc.cumulative_sum(bigram_counts)
    zero = pa.array([0], type=cumulative_counts.type)
    offsets = pa.concat_arrays([zero, cumulative_counts])
    assert offsets[-1].as_py() == len(bigrams_flat), "Final offset must match flat bigram count"
    return pa.ListArray.from_arrays(offsets, bigrams_flat)


def unigram_tokenize_pa(texts: Iterable[str]) -> pa.ListArray:
    """
    Parameters
    ----------
    texts : Iterable[str]
    Returns
    -------
    py arrow ListArray of strings, essentially a List[List[str]]
    """
    # Assert that there are no None values in the input texts
    assert all(t is not None for t in texts), "texts must not contain None entries"

    # Put the raw Python list into an Arrow string array
    arr = pa.array(texts, pa.string())

    # NFKC normalize and lowercase
    arr = pc.utf8_normalize(arr, "NFKC")
    arr = pc.utf8_lower(arr)

    # Token‐boundary pre-processing
    # * surround every Han code-point with spaces
    # * turn all runs of punctuation into a single space
    arr = pc.replace_substring_regex(arr, _RE_HAN_SINGLE, r" \1 ")
    arr = pc.replace_substring_regex(arr, _RE_PUNCT, " ")
    arr = pc.utf8_trim_whitespace(arr)

    # Handle empty strings by setting to None (so they don't produce tokens) and then coalescing at the end
    is_empty_mask = pc.equal(arr, "")
    arr = pc.if_else(is_empty_mask, pa.scalar(None, type=arr.type), arr)

    # Split by whitespace, reset None entries to empty list
    tokens = pc.utf8_split_whitespace(arr)
    tokens = pc.coalesce(tokens, pa.scalar([], type=tokens.type))

    # Assert the function returns a ListArray of strings
    assert isinstance(tokens, pa.ListArray) and patypes.is_string(tokens.type.value_type), (
        "Expected output to be ListArray<string>"
    )
    return tokens


def string_to_packed_trigrams(s: str):
    """
    :param s: string
    :returns: Numpy uint32 array, one element per character in original string, via utf-32 encoding

    Takes advantage of the fact that UTF-32 encoding is always 4 bytes per character,
    but never uses more than the 21 rightmost bits.
    Hence, you can pack 3 characters into 63 bytes with a | (b << 21) | (c << 62)
    """

    if len(s) < 3:
        return np.array([], dtype=np.uint64)

    s_utf32 = s.encode('utf-32-le')
    # Take the raw bytes, get an array of each utf-32 codepoint, then upcast to uint64 to give room for bit shifting
    char_array = np.frombuffer(s_utf32, dtype=np.uint32).astype(np.uint64)
    assert np.all(char_array < (1 << 21)), "Character codes exceed 21 bits"

    packed_trigrams = char_array[:-2] + (char_array[1:-1] << 21) + (char_array[2:] << 42)
    return packed_trigrams

def packed_trigram_to_string(packed_tg: int | np.uint64) -> str:
    """
    Decodes a NumPy array of packed uint64 codes into trigram strings.

    This function is the specific inverse of `string_to_packed_trigrams`, mostly for debugging puproses
    Args:
        trigram_codes: A NumPy array with dtype=uint64, where each element is
                       a packed trigram.

    Returns:
        A list of the decoded three-character strings.
    """
    mask = (1<<21) - 1
    packed_tg = int(packed_tg)
    a = packed_tg & mask
    b = (packed_tg  >> 21) & mask
    c = (packed_tg >> 42) & mask
    arr = np.array([a,b,c], dtype=np.uint32)
    return arr.tobytes().decode('utf-32-le')


def extract_packed_trigrams_pa(texts):
    """
    Generates packed uint64 representations of character trigrams from a string

    Takes advantage of the fact that UTF-32 encoding is always 4 bytes per character,
    but never uses more than the 21 rightmost bits.

    Hence, you can pack 3 characters into 63 bytes with a | (b << 21) | (c << 62)

    While this seems a bit fancy, it is much faster (~5x) than just iterating over length 3 substrings

    Args:
        text: The input string.

    Returns:
        A NumPy array of dtype uint64, where each element is a
        packed representation of a character trigram. Returns an empty
        array if the input text is too short to form a trigram.
    """

    # Normalize unicode, lowercase
    texts = pa.array(texts, pa.string())
    texts = pc.utf8_normalize(texts, "NFKC")
    texts = pc.utf8_lower(texts)


    # Add boundary markers '#' to start / end of string and to multiple consecutive whitespaces
    # For some reason in pyarrow you have to do ^\s* instead of ^ (same for $) to get this to work
    texts = pc.replace_substring_regex(texts, r"^\s*|\s+|\s*$", "#")
    texts = pc.replace_substring_regex(texts, r"#{2,}", "#")
    
    texts = texts.to_numpy(zero_copy_only=False)
    packed_trigrams = [string_to_packed_trigrams(s) for s in texts]
    return pa.array(packed_trigrams, type=pa.list_(pa.uint64()))


def ngram_tokenize(strings: Iterable[str], include_bigrams: bool = False):
    """
    Tokenize input strings into unigrams and, optionally, bigrams.

    Parameters
    ----------
    strings : Iterable[str]
        The input collection of strings to tokenize.
    include_bigrams : bool, default False
        If True, also extract bigrams formed by adjacent unigrams within each
        input string.

    Returns
    -------
    list | tuple(list, list)
        If `include_bigrams` is False, returns the list of unigram tokens for
        each input string
        If `include_bigrams` is True, also return bigram tokens
    """

    # Generate the unigram tokens first
    unigrams_pa = unigram_tokenize_pa(strings)

    # If bigrams are not requested, return unigrams only
    if not include_bigrams:
        return unigrams_pa.to_pylist()

    # Extract bigrams from the previously-computed unigrams
    bigrams_pa = extract_bigrams_from_unigrams_pa(unigrams_pa)
    bigrams_py = bigrams_pa.to_pylist()
    return unigrams_pa.to_pylist(), bigrams_py


def pa_value_counts_to_dict(struct_arr: pa.StructArray) -> dict[Any,int]:
    """Convert a StructArray returned by pc.value_counts to a Python dict mapping value to count."""
    return {
        token: count
        for token, count in zip(
            struct_arr.field("values").to_pylist(),
            struct_arr.field("counts").to_pylist(),
        )
    }

def ngram_counts(strings: Iterable[str], include_bigrams: bool = False):
    """
    Generate frequency counts of unigrams and, optionally, bigrams occurring in
    the provided *strings*.

    Parameters
    ----------
    strings : Iterable[str]
        The input strings to analyse.
    include_bigrams : bool, default False
        If True, also compute counts for bigrams derived from adjacent
        unigrams in each string.

    Returns
    -------
    If not including bigrams, a single Dict[str,int] counting tokens
    If including bigrams, a tuple of (unigram counts, bigram counts)
    """

    # --- Unigram counts ---
    unigrams_pa = unigram_tokenize_pa(strings)
    unigram_counts = pa_value_counts_to_dict(pc.value_counts(unigrams_pa.values))

    if not include_bigrams:
        return unigram_counts

    # --- Bigram counts ---
    bigrams_pa = extract_bigrams_from_unigrams_pa(unigrams_pa)
    bigram_counts = pa_value_counts_to_dict(pc.value_counts(bigrams_pa.values))

    return unigram_counts, bigram_counts


class VocabNgramTokenizer:
    """Tokenize text into unigram / bigram ID sequences using fast Arrow-based mappers.

    Parameters
    ----------
    unigram_vocab : dict[str,int]
        Mapping from unigram token string to integer ID.
    bigram_vocab : dict[str,int] | None, default None
        Optional mapping for bigram tokens.
    """

    def __init__(self, unigram_vocab: dict[str, int], bigram_vocab: dict[str, int] | None = None):
        self._uni_mapper = ArrowStringArrayMapper(unigram_vocab)
        self._bi_mapper = ArrowStringArrayMapper(bigram_vocab) if bigram_vocab is not None else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _map_list_array_to_id_lists(list_arr: pa.ListArray, mapper: ArrowStringArrayMapper):
        """Convert a PyArrow ListArray<string> into List[np.ndarray[int64]]."""
        string_arr: pa.StringArray = list_arr.values  # flat string array

        # Ensure no nulls – our upstream tokenisers should not generate any
        assert string_arr.null_count == 0, "StringArray must not contain null values"
        # This format is garanteed by the Arrow format specification
        # E.g see "Variable-size Binary Layout": https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
        null_buf, offsets_buf, data_buf = string_arr.buffers()
        assert null_buf is None, "Unexpected null bitmap present in StringArray"

        arr_len = len(string_arr)
        # Offsets buffer is (arr_len + 1) int32 values
        offset_np = np.frombuffer(offsets_buf, dtype=np.int32, count=arr_len + 1)
        data_bytes = data_buf.to_pybytes()

        # Map to IDs via the fast C++ helper
        ids_flat = mapper.map_ids(data_bytes, offset_np)
        assert len(ids_flat) == arr_len, "Mapped ID array length mismatch"

        # Now split the flat IDs array back into per-row lists using the list offsets
        assert list_arr.null_count ==0, "ListArray must not contain null values"
        list_offsets = list_arr.offsets.to_numpy()

        out: list[np.ndarray] = []
        for i in range(len(list_arr)):
            start = list_offsets[i]
            end = list_offsets[i + 1]
            out.append(ids_flat[start:end])
        return out

    def tokenize(self, strings: Iterable[str], include_bigrams: bool = False):
        """Tokenize *strings* and map tokens to integer IDs.

        If *include_bigrams* is False, returns List[np.ndarray[int64]] – one array per
        input string, containing unigram IDs.  If True, returns a tuple with the
        unigram ID lists and bigram ID lists respectively.
        """
        # Unigram IDs
        uni_list_arr = unigram_tokenize_pa(strings)
        uni_ids = self._map_list_array_to_id_lists(uni_list_arr, self._uni_mapper)

        if not include_bigrams:
            return uni_ids

        # Need bigrams – ensure we have a mapper
        assert self._bi_mapper is not None, "bigram_vocab was not supplied at construction time"
        bi_list_arr = extract_bigrams_from_unigrams_pa(uni_list_arr)
        bi_ids = self._map_list_array_to_id_lists(bi_list_arr, self._bi_mapper)
        return uni_ids, bi_ids

def char_trigram_tokenize(strings: Iterable[str]):
    packed_tg_list = extract_packed_trigrams_pa(strings)
    return [
        [packed_trigram_to_string(tg) for tg in packed_tg_arr]
        for packed_tg_arr in packed_tg_list.to_numpy(zero_copy_only=False)
    ]

def char_trigram_counts(strings: Iterable[str]) -> dict[str,int]:
    """
    Compute frequency counts of character trigrams occurring in *strings*.

    Parameters
    ----------
    strings : Iterable[str]
        The input strings to analyse.

    Returns
    -------
    dict[str, int]
        Mapping from trigram string → occurrence count.
    """

    # Generate packed trigram codes per input string (ListArray<uint64>)
    packed_tg_list = extract_packed_trigrams_pa(strings)

    # Flatten to one UInt64Array and count occurrences
    packed_tg_flat: pa.UInt64Array = packed_tg_list.values
    counts_struct = pc.value_counts(packed_tg_flat)

    # Convert to Python dict {packed_code: count}
    packed_counts: dict[int, int] = pa_value_counts_to_dict(counts_struct)

    # Convert packed codes → 3-character strings
    trigram_counts: dict[str, int] = {
        packed_trigram_to_string(int(code)): count
        for code, count in packed_counts.items()
    }
    return trigram_counts



class VocabCharTrigramTokenizer:
    """Tokenize text into mapped character trigram ID sequences.
    Parameters
    ----------
    trigram_vocab : dict[str, int]
        Mapping from 3-character trigram string to integer ID.
    """

    # ---------------------------- helpers ----------------------------
    @staticmethod
    def _pack_trigram_string(trigram: str) -> int:
        """Pack a single 3-character string into its uint64 representation."""
        packed_arr = string_to_packed_trigrams(trigram)
        assert len(packed_arr) == 1, "Expected a single trigram string of length 3"
        return int(packed_arr[0])

    @staticmethod
    def _map_list_array_to_id_lists(list_arr: pa.ListArray, mapper: Uint64Mapper):
        """Convert a PyArrow ListArray<uint64> into List[np.ndarray[int64]]."""
        uint64_arr: pa.UInt64Array = list_arr.values  # flat uint64 values

        # Ensure there are no nulls in the flat values
        assert uint64_arr.null_count == 0, "UInt64Array must not contain null values"

        # Convert flat array to NumPy for mapping
        flat_np = uint64_arr.to_numpy()
        ids_flat = mapper.map_ids(flat_np)
        assert len(ids_flat) == len(uint64_arr), "Mapped ID array length mismatch"

        # Re-chunk into per-row lists using ListArray offsets
        assert list_arr.null_count == 0, "ListArray must not contain null values"
        offsets = list_arr.offsets.to_numpy()

        out: list[np.ndarray] = []
        for i in range(len(list_arr)):
            start = offsets[i]
            end = offsets[i + 1]
            out.append(ids_flat[start:end])
        return out

    def __init__(self, trigram_vocab: dict[str, int]):
        # Convert UTF-8 trigram keys to packed uint64 codes
        packed_dict: dict[int, int] = {
            self._pack_trigram_string(k): v for k, v in trigram_vocab.items()
        }

        self._mapper = Uint64Mapper(packed_dict)

    def tokenize(self, strings: Iterable[str]):
        """Tokenize *strings* and map trigrams to integer IDs.

        Returns
        -------
        List[np.ndarray[int64]] – one array per input string, containing
        trigram IDs.  Unknown trigrams are encoded as -1.
        """

        list_arr = extract_packed_trigrams_pa(strings)
        return self._map_list_array_to_id_lists(list_arr, self._mapper)
