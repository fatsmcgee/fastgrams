// fastgrams – condensed n-gram / char-trigram code path taken from
// the Pinterest OmniSearchSage implementation, python-centric (no Torch)
// with some modifications/optimizations (especially for char-trigram tokenization).
//
// Public Python API (via pybind11):
//   ngram_tokenize(s: str, n: int = 1)          -> List[str]
//   char_trigram_tokenize(s: str)               -> List[str]
//   ngram_counts(strings: Iterable[str], n=1)   -> Dict[str, int]
//   char_trigram_counts(strings: Iterable[str]) -> Dict[str, int]
//
// Build with scikit-build-core + pybind11.  C++-side deps: ICU, absl.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <absl/container/flat_hash_map.h>

#include <deque>
#include <string>
#include <vector>
#include <cstdint>
#include <cassert>
#include <string_view>

namespace py = pybind11;
// Unicode code points are at most 21 bits (max value 0x10FFFF), so no character can be larger than 1<<21.
// and 3 characters can be packed into a 63-bit integer.
// This is a much more compact way to store a trigram than an std::string.
typedef std::uint64_t packed_trigram;

// Forward declaration so structs above can call it
inline packed_trigram pack_trigram(char32_t a, char32_t b, char32_t c);

/* ------------------------------------------------------------------------ */
/*  Small helpers                                                           */
/* ------------------------------------------------------------------------ */
namespace {

constexpr char32_t kBoundary = U'#';

inline std::string to_utf8(const icu::UnicodeString& us) {
    std::string out;
    us.toUTF8String(out);
    return out;
}

/* ---------- Unicode tests (mirrors tokenization_bert_utils.h) ------------ */
inline bool is_whitespace(char32_t cp) {
    return cp == U'\t' || cp == U'\n' || cp == U'\r' || cp == U' ' ||
           u_charType(cp) == U_SPACE_SEPARATOR;
}
inline bool is_punctuation(char32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        return true;
    return u_ispunct(cp);
}
inline bool is_cjk(char32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF)  ||  // CJK Unified Ideographs
           (cp >= 0x3400 && cp <= 0x4DBF)  ||  // CJK Ext-A
           (cp >= 0x20000 && cp <= 0x2A6DF)||
           (cp >= 0x2A700 && cp <= 0x2B73F)||
           (cp >= 0x2B740 && cp <= 0x2B81F)||
           (cp >= 0x2B820 && cp <= 0x2CEAF)||
           (cp >= 0xF900 && cp <= 0xFAFF)  ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

/* ---------- ICU normaliser (NFKC_CaseFold) ------------------------------ */
const icu::Normalizer2* get_nfkc_cf() {
    static UErrorCode status = U_ZERO_ERROR;
    static const icu::Normalizer2* nfkc_cf =
        icu::Normalizer2::getNFKCCasefoldInstance(status);
    if (U_FAILURE(status))
        throw std::runtime_error(u_errorName(status));
    return nfkc_cf;
}

icu::UnicodeString normalize(const icu::UnicodeString& src) {
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString dest;
    get_nfkc_cf()->normalize(src, dest, status);
    if (U_FAILURE(status))
        throw std::runtime_error(u_errorName(status));
    return dest;
}
/* ------------------------------------------------------------------------ */
/*  Sliding buffers used by original implementation                          */
/* ------------------------------------------------------------------------ */

class CharQueue {
public:
    explicit CharQueue(std::size_t n) : max_(n) {}
    void push(char32_t c) {
        if (buf_.size() == max_) buf_.pop_front();
        buf_.push_back(c);
    }
    std::size_t size() const { return buf_.size(); }
    // Returns the packed uint64 version of the current 3-character buffer.
    // Only valid when the queue size equals its capacity (n <= 3 in our use-case).
    packed_trigram to_packed() const {
        assert(max_ == 3 && "to_packed only supports n == 3");
        assert(buf_.size() == max_ && "to_packed called with incomplete buffer");
        return pack_trigram(buf_[0], buf_[1], buf_[2]);
    }
    icu::UnicodeString str() const {
        icu::UnicodeString out(max_, 0, 0);
        for (auto c : buf_) out.append(static_cast<int32_t>(c));
        return out;
    }
    char32_t back() const { return buf_.back(); }
private:
    std::deque<char32_t> buf_;
    std::size_t          max_;
};

class WordQueue {
public:
    explicit WordQueue(std::size_t n) : max_(n) {}
    void push(icu::UnicodeString&& token) {
        if (buf_.size() == max_) buf_.pop_front();
        buf_.push_back(std::move(token));
    }
    std::size_t size() const { return buf_.size(); }
    icu::UnicodeString str() const {
        icu::UnicodeString out;
        for (std::size_t i = 0; i < buf_.size(); ++i) {
            if (i) out.append(static_cast<int32_t>(kBoundary));
            out.append(buf_[i]);
        }
        return out;
    }
private:
    std::deque<icu::UnicodeString> buf_;
    std::size_t                    max_;
};

/* ------------------------------------------------------------------------ */
/*  Core algorithms (borrowed verbatim, but isolated)                        */
/* ------------------------------------------------------------------------ */


bool is_ws_or_punct(char32_t c) { return is_whitespace(c) || is_punctuation(c); }

void ngramify(const icu::UnicodeString& text,
              std::size_t n,
              std::vector<icu::UnicodeString>& out) {

    if (static_cast<std::size_t>(text.length()) < n) return;

    WordQueue wq(n);

    int32_t len = text.length();
    int32_t w_begin = 0, w_end = 0;

    while (w_end < len) {
        while (w_begin < len && is_ws_or_punct(text.char32At(w_begin)))
            ++w_begin;
        w_end = w_begin;
        while (w_end < len && !is_ws_or_punct(text.char32At(w_end)))
            ++w_end;

        if (w_end > w_begin) {
            icu::UnicodeString token;
            text.extract(w_begin, w_end - w_begin, token);
            if (n == 1) {
                out.emplace_back(std::move(token));
            } else {
                wq.push(std::move(token));
                if (wq.size() == n) out.emplace_back(wq.str());
            }
        }
        w_begin = w_end;
    }
}


icu::UnicodeString tokenize_chinese_chars(const icu::UnicodeString& text) {
    // Add spaces around all Han characters so that in subsequent whitespace tokenization
    // they will be treated as individual words
    icu::UnicodeString out(text.length(), 0, 0);
    for (int32_t i = 0; i < text.length(); ) {
        UChar32 cp = text.char32At(i);
        i += U16_LENGTH(cp);
        if (is_cjk(cp)) {
            out.append(' ').append(cp).append(' ');
        } else {
            out.append(cp);
        }
    }
    return out;
}

// Helper functions to convert between packed trigrams and UTF-8 strings
constexpr std::uint64_t kMask21 = (1ULL << 21) - 1ULL;

inline std::string unpack_trigram(packed_trigram p) {
    char32_t a = static_cast<char32_t>(p & kMask21);
    char32_t b = static_cast<char32_t>((p >> 21) & kMask21);
    char32_t c = static_cast<char32_t>((p >> 42) & kMask21);
    icu::UnicodeString us;
    us.append(static_cast<int32_t>(a))
      .append(static_cast<int32_t>(b))
      .append(static_cast<int32_t>(c));
    return to_utf8(us);
}

inline packed_trigram pack_trigram(char32_t a, char32_t b, char32_t c) {
    return (static_cast<std::uint64_t>(a) & kMask21) |
           ((static_cast<std::uint64_t>(b) & kMask21) << 21) |
           ((static_cast<std::uint64_t>(c) & kMask21) << 42);
}

inline packed_trigram pack_trigram(const icu::UnicodeString& us) {
    assert(us.countChar32() == 3 && "pack_trigram expects exactly 3 code points");
    int32_t offset = 0;
    char32_t a = us.char32At(offset);
    offset += U16_LENGTH(a);
    char32_t b = us.char32At(offset);
    offset += U16_LENGTH(b);
    char32_t c = us.char32At(offset);
    return pack_trigram(a, b, c);
}

}  // namespace


inline packed_trigram pack_trigram(char32_t a, char32_t b, char32_t c) {
    constexpr std::uint64_t kMask21_global = (1ULL << 21) - 1ULL;
    return (static_cast<std::uint64_t>(a) & kMask21_global) |
           ((static_cast<std::uint64_t>(b) & kMask21_global) << 21) |
           ((static_cast<std::uint64_t>(c) & kMask21_global) << 42);
}

/* ------------------------------------------------------------------------ */
/*  Public (non-class) API – mirrored signatures                             */
/* ------------------------------------------------------------------------ */
namespace fastgrams {

std::vector<std::string> ngram_tokenize(const std::string& utf8,
                                        int n = 1) {
    icu::UnicodeString us = icu::UnicodeString::fromUTF8(utf8);
    icu::UnicodeString norm = normalize(us);
    icu::UnicodeString spaced = tokenize_chinese_chars(norm);

    std::vector<icu::UnicodeString> tmp;
    ngramify(spaced, static_cast<std::size_t>(n), tmp);

    std::vector<std::string> out;
    out.reserve(tmp.size());
    for (auto& u : tmp) out.push_back(to_utf8(u));
    return out;
}

std::vector<packed_trigram> char_trigram_tokenize_packed(const std::string& utf8) {
    icu::UnicodeString norm = normalize(
        icu::UnicodeString::fromUTF8(utf8));

    std::vector<packed_trigram> out;
    if (norm.length() + 2 < 3)      // fewer than 3 code-points incl. sentinels
        return out;

    CharQueue cq(3);                // 3-char sliding window
    cq.push(kBoundary);             // leading '#'

    int32_t i = 0, len = norm.length();
    while (true) {
        bool exhausted = (i == len);     // true ⇒ flush trailing '#'
        char32_t cp = exhausted ? kBoundary : norm.char32At(i);

        if (is_whitespace(cp) || exhausted) {
            if (cq.back() != kBoundary) {
                cq.push(kBoundary);
                if (cq.size() == 3) {
                    out.push_back(cq.to_packed());
                }
            }
        } else {
            cq.push(cp);
            if (cq.size() == 3) {
                out.push_back(cq.to_packed());
            }
        }

        if (exhausted) break;
        ++i;
    }

    return out;
}

std::vector<std::string> char_trigram_tokenize(const std::string& utf8) {
    std::vector<packed_trigram> packed = char_trigram_tokenize_packed(utf8);
    std::vector<std::string> out;
    out.reserve(packed.size());
    for (auto p : packed) {
        out.push_back(unpack_trigram(p));
    }
    return out;
}

/* ------------------------ counting helpers ------------------------------ */
py::dict ngram_counts(py::iterable strings, int n = 1) {
    absl::flat_hash_map<std::string, std::int64_t> freq;
    for (auto obj : strings) {
        for (const auto& tok : ngram_tokenize(py::cast<std::string>(obj), n))
            ++freq[tok];
    }
    py::dict out;
    for (auto& kv : freq) out[py::str(kv.first)] = kv.second;
    return out;
}

py::dict char_trigram_counts(py::iterable strings, int /*unused*/ = 1) {
    absl::flat_hash_map<packed_trigram, std::int64_t> freq;
    for (auto obj : strings) {
        for (const auto& tok : char_trigram_tokenize_packed(py::cast<std::string>(obj)))
            ++freq[tok];
    }
    py::dict out;
    for (auto& kv : freq) out[py::str(unpack_trigram(kv.first))] = kv.second;
    return out;
}

/* -------------------------------------------------------------------- */
/*  Vocabulary-aware tokenizer classes                                   */
/* -------------------------------------------------------------------- */

class VocabNgramTokenizer {
public:
    VocabNgramTokenizer(py::dict vocab, int n = 1) : n_(n) {
        for (auto kv : vocab) {
            vocab_[py::cast<std::string>(kv.first)] =
                static_cast<std::int64_t>(py::cast<std::int64_t>(kv.second));
        }
    }

    py::list tokenize(py::iterable strings, py::object default_obj = py::none()) const {
        const bool has_default = !default_obj.is_none();
        std::int64_t default_val = 0;
        if (has_default) {
            default_val = static_cast<std::int64_t>(py::cast<std::int64_t>(default_obj));
        }

        py::list out;
        for (auto py_s : strings) {
            std::vector<std::string> toks = ngram_tokenize(py::cast<std::string>(py_s), n_);

            std::vector<std::int64_t> idxs;
            idxs.reserve(toks.size());

            for (const auto& tok : toks) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end()) {
                    idxs.push_back(it->second);
                } else if (has_default) {
                    idxs.push_back(default_val);
                }
            }

            // Build numpy array
            py::array_t<std::int64_t> arr(idxs.size());
            auto buf = arr.mutable_unchecked<1>();
            for (std::size_t i = 0; i < idxs.size(); ++i)
                buf(i) = idxs[i];
            out.append(std::move(arr));
        }
        return out;
    }

private:
    absl::flat_hash_map<std::string, std::int64_t> vocab_;
    int n_;
};

class VocabCharTrigramTokenizer {
public:
    explicit VocabCharTrigramTokenizer(py::dict vocab) {
        for (auto kv : vocab) {
            std::string key_utf8 = py::cast<std::string>(kv.first);
            icu::UnicodeString us = icu::UnicodeString::fromUTF8(key_utf8);
            packed_trigram key = pack_trigram(us);
            vocab_[key] = static_cast<std::int64_t>(py::cast<std::int64_t>(kv.second));
        }
    }

    py::list tokenize(py::iterable strings, py::object default_obj = py::none()) const {
        const bool has_default = !default_obj.is_none();
        std::int64_t default_val = 0;
        if (has_default) {
            default_val = static_cast<std::int64_t>(py::cast<std::int64_t>(default_obj));
        }

        py::list out;
        for (auto py_s : strings) {
            std::vector<packed_trigram> toks = char_trigram_tokenize_packed(py::cast<std::string>(py_s));

            std::vector<std::int64_t> idxs;
            idxs.reserve(toks.size());

            for (const auto& tok : toks) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end()) {
                    idxs.push_back(it->second);
                } else if (has_default) {
                    idxs.push_back(default_val);
                }
            }

            py::array_t<std::int64_t> arr(idxs.size());
            auto buf = arr.mutable_unchecked<1>();
            for (std::size_t i = 0; i < idxs.size(); ++i)
                buf(i) = idxs[i];
            out.append(std::move(arr));
        }
        return out;
    }

private:
    absl::flat_hash_map<packed_trigram, std::int64_t> vocab_;
};

class ArrowStringArrayMapper {
public:
    explicit ArrowStringArrayMapper(py::dict vocab) {
        for (auto kv : vocab) {
            vocab_[py::cast<std::string>(kv.first)] =
                static_cast<std::int64_t>(py::cast<std::int64_t>(kv.second));
        }
    }

    /*
    This takes the buffer representation of an Arrow StringArray, which has:
    - data: raw utf-8 bytes
    - offsets: int32 array such that data[offsets[i]:offsets[i+1]] has the utf-8 data for the i-th string
    */
    py::array_t<std::int64_t> map_ids(
        py::bytes utf8_data,
        py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> offsets) const {
        // copy the byte buffer
        std::string data = utf8_data;
        const char* base = data.data();
        std::size_t data_size = data.size();

        auto off = offsets.unchecked<1>();
        if (off.shape(0) < 1)
            throw std::runtime_error("offsets must contain at least one element");

        std::size_t n_strings = static_cast<std::size_t>(off.shape(0) - 1);
        py::array_t<std::int64_t> out(n_strings);
        auto out_buf = out.mutable_unchecked<1>();

        for (std::size_t i = 0; i < n_strings; ++i) {
            std::int32_t start = off(i);
            std::int32_t end   = off(i + 1);
            if (start < 0 || end < start || static_cast<std::size_t>(end) > data_size)
                throw std::runtime_error("invalid offsets for provided utf8_data");

            std::string_view sv(base + start, static_cast<std::size_t>(end - start));
            auto it = vocab_.find(sv);
            out_buf(i) = (it == vocab_.end()) ? -1 : it->second;
        }
        return out;
    }

private:
    absl::flat_hash_map<std::string, std::int64_t> vocab_;
};

class Uint64Mapper {
public:
    explicit Uint64Mapper(py::dict vocab) {
        for (auto kv : vocab) {
            std::uint64_t key = static_cast<std::uint64_t>(py::cast<std::uint64_t>(kv.first));
            std::int64_t  val = static_cast<std::int64_t>(py::cast<std::int64_t>(kv.second));
            vocab_[key] = val;
        }
    }

    // Map a uint64 NumPy array to int64 IDs using the internal hash map. Unknowns → -1.
    py::array_t<std::int64_t> map_ids(
        py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> ids) const {
        auto in = ids.unchecked<1>();
        py::array_t<std::int64_t> out(in.shape(0));
        auto out_buf = out.mutable_unchecked<1>();
        for (ssize_t i = 0; i < in.shape(0); ++i) {
            auto it = vocab_.find(in(i));
            out_buf(i) = (it == vocab_.end()) ? -1 : it->second;
        }
        return out;
    }

private:
    absl::flat_hash_map<std::uint64_t, std::int64_t> vocab_;
};

} // namespace fastgrams

//PyBind glue
PYBIND11_MODULE(_fastgrams, m) {
    m.doc() = "Fast n-gram & char-trigram tokeniser (ICU + absl).";

    m.def("ngram_tokenize",        &fastgrams::ngram_tokenize,
          py::arg("s"), py::arg("n") = 1);
    m.def("char_trigram_tokenize", &fastgrams::char_trigram_tokenize,
          py::arg("s"));
    m.def("ngram_counts",          &fastgrams::ngram_counts,
          py::arg("strings"), py::arg("n") = 1);
    m.def("char_trigram_counts",   &fastgrams::char_trigram_counts,
          py::arg("strings"), py::arg("n") = 1);

    // Expose vocabulary-aware tokenizers
    py::class_<fastgrams::VocabNgramTokenizer>(m, "VocabNgramTokenizer")
        .def(py::init<py::dict, int>(), py::arg("vocab"), py::arg("n") = 1)
        .def("tokenize", &fastgrams::VocabNgramTokenizer::tokenize,
             py::arg("strings"), py::arg("default") = py::none());

    py::class_<fastgrams::VocabCharTrigramTokenizer>(m, "VocabCharTrigramTokenizer")
        .def(py::init<py::dict>(), py::arg("vocab"))
        .def("tokenize", &fastgrams::VocabCharTrigramTokenizer::tokenize,
             py::arg("strings"), py::arg("default") = py::none());

    py::class_<fastgrams::ArrowStringArrayMapper>(m, "ArrowStringArrayMapper")
        .def(py::init<py::dict>(), py::arg("vocab"))
        .def("map_ids", &fastgrams::ArrowStringArrayMapper::map_ids,
             py::arg("utf8_data"), py::arg("offsets"));

    py::class_<fastgrams::Uint64Mapper>(m, "Uint64Mapper")
        .def(py::init<py::dict>(), py::arg("vocab"))
        .def("map_ids", &fastgrams::Uint64Mapper::map_ids,
             py::arg("ids"));
}
