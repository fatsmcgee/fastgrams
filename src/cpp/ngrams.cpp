// fastgrams – condensed n-gram / char-trigram code path taken from
// the Pinterest OSS implementation, stripped of Torch & logging.
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

namespace py = pybind11;

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

void char_ngramify(const icu::UnicodeString& text,
                   std::size_t n,
                   std::vector<icu::UnicodeString>& out) {

    if (static_cast<std::size_t>(text.length()) + 2 < n) return;

    CharQueue cq(n);
    cq.push(kBoundary);                       // leading '#'

    int32_t i = 0;
    const int32_t len = text.length();
    while (i <= len) {
        if (i == len) {                       // flush trailing boundary
            if (cq.back() != kBoundary) {
                cq.push(kBoundary);
                if (cq.size() == n) out.emplace_back(cq.str());
            }
            ++i;
            continue;
        }

        char32_t cp = text.char32At(i);
        if (is_whitespace(cp)) {
            if (cq.back() != kBoundary) {
                cq.push(kBoundary);
                if (cq.size() == n) out.emplace_back(cq.str());
            }
        } else {
            cq.push(cp);
            if (cq.size() == n) out.emplace_back(cq.str());
        }
        ++i;
    }
}

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

}  // namespace

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

std::vector<std::string> char_trigram_tokenize(const std::string& utf8) {
    icu::UnicodeString us = icu::UnicodeString::fromUTF8(utf8);
    icu::UnicodeString norm = normalize(us);

    std::vector<icu::UnicodeString> tmp;
    char_ngramify(norm, /*n=*/3, tmp);

    std::vector<std::string> out;
    out.reserve(tmp.size());
    for (auto& u : tmp) out.push_back(to_utf8(u));
    return out;
}

/* ------------------------ counting helpers ------------------------------ */
template <typename TokenFunc>
py::dict generic_counts(py::iterable strings,
                        int n,
                        TokenFunc&& fn) {
    absl::flat_hash_map<std::string, std::int64_t> freq;
    for (auto obj : strings) {
        for (const auto& tok : fn(py::cast<std::string>(obj), n))
            ++freq[tok];
    }
    py::dict out;
    for (auto& kv : freq) out[py::str(kv.first)] = kv.second;
    return out;
}

py::dict ngram_counts(py::iterable strings, int n = 1) {
    return generic_counts(strings, n, [](const std::string& s, int nn) {
        return ngram_tokenize(s, nn);
    });
}

py::dict char_trigram_counts(py::iterable strings, int /*unused*/ = 1) {
    return generic_counts(strings, 0, [](const std::string& s, int) {
        return char_trigram_tokenize(s);
    });
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
            std::vector<std::string> toks = char_trigram_tokenize(py::cast<std::string>(py_s));

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
    absl::flat_hash_map<std::string, std::int64_t> vocab_;
};

}

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
}
