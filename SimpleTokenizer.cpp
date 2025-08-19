#include "SimpleTokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>

// --- Helper Functions ---
std::vector<std::string> bytes_to_unicode() {
    std::vector<std::string> result;
    std::ifstream ifs("../models/vocab.txt");
    if (!ifs.is_open()) {
        throw std::runtime_error("Byte vocabulary file not found.");
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            result.push_back(line);
        }
    }
    ifs.close();
    return result;
}

std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) return pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.insert({word[i], word[i+1]});
    }
    return pairs;
}

std::string basic_clean(std::string text) {
    // Basic HTML unescape
    text = std::regex_replace(text, std::regex("&lt;"), "<");
    text = std::regex_replace(text, std::regex("&gt;"), ">");
    text = std::regex_replace(text, std::regex("&quot;"), "\"");
    text = std::regex_replace(text, std::regex("&#39;"), "'");
    text = std::regex_replace(text, std::regex("&amp;"), "&");

    // Remove leading/trailing whitespace
    text = std::regex_replace(text, std::regex("^\\s+|\\s+$"), "");
    return text;
}

std::string whitespace_clean(std::string text) {
    text = std::regex_replace(text, std::regex("\\s+"), " ");
    text = std::regex_replace(text, std::regex("^\\s+|\\s+$"), "");
    return text;
}

std::string clean_lower(std::string x) {
    std::string cleaned = whitespace_clean(basic_clean(x));
    std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return cleaned;
}


// --- SimpleTokenizer Class Implementation ---

SimpleTokenizer::SimpleTokenizer(const std::string& bpe_path, int context_length, const std::vector<std::string>& additional_special_tokens) {
    this->context_length_ = context_length;
    this->clean_fn_ = clean_lower;

    // 1. Initialize byte encoder/decoder
    this->byte_encoder_ = bytes_to_unicode();

    // 2. Load BPE merges from file
    std::ifstream merges_file(bpe_path);
    if (!merges_file.is_open()) {
        throw std::runtime_error("BPE vocabulary file not found at " + bpe_path);
    }
    std::vector<std::pair<std::string, std::string>> merges;
    std::string line;
    std::getline(merges_file, line); // Skip header line
    while (std::getline(merges_file, line)) {
        std::stringstream ss(line);
        std::string first, second;
        ss >> first >> second;
        if (!first.empty() && !second.empty()) {
            merges.push_back({first, second});
        }
    }

    // 3. Build vocabulary
    std::vector<std::string> vocab;
    for (const auto& pair : this->byte_encoder_) {
        vocab.push_back(pair);
    }
    for (const auto& pair : this->byte_encoder_) {
        vocab.push_back(pair + "</w>");
    }
    for (const auto& merge : merges) {
        vocab.push_back(merge.first + merge.second);
    }

    std::vector<std::string> special_tokens = {"<start_of_text>", "<end_of_text>"};
    special_tokens.insert(special_tokens.end(), additional_special_tokens.begin(), additional_special_tokens.end());
    vocab.insert(vocab.end(), special_tokens.begin(), special_tokens.end());

    // 4. Build encoder and decoder maps
    for (size_t i = 0; i < vocab.size(); ++i) {
        this->encoder_[vocab[i]] = i;
    }

    // 5. Build BPE ranks map
    for (size_t i = 0; i < merges.size(); ++i) {
        this->bpe_ranks_[merges[i]] = i;
    }

    this->vocab_size_ = vocab.size();

    // 6. Setup special tokens and regex pattern
    this->all_special_ids_.reserve(special_tokens.size());
    std::string special_pattern_part;
    for(const auto& token : special_tokens) {
        this->all_special_ids_.push_back(this->encoder_.at(token));
        if (!special_pattern_part.empty()) special_pattern_part += "|";
        special_pattern_part += token;
    }
    this->sot_token_id_ = this->encoder_.at("<start_of_text>");
    this->eot_token_id_ = this->encoder_.at("<end_of_text>");

    // NOTE: The regex uses ASCII approximations for Unicode properties \p{L} and \p{N}
    std::string pattern_str = special_pattern_part + R"(|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w\d]+)";
    this->pat_ = std::regex(pattern_str, std::regex::icase);
}


std::string SimpleTokenizer::bpe(const std::string& token) const {
    if (cache_.count(token)) {
        return cache_.at(token);
    }

    std::vector<std::string> word;
    for (char c : token) {
        word.push_back(std::string(1, c));
    }
    if (word.empty()) {
        return token;
    }
    word.back() += "</w>";

    auto pairs = get_pairs(word);

    if (pairs.empty()) {
        std::string result;
        for(const auto& s : word) result += s;
        cache_[token] = result;
        return result;
    }

    while (true) {
        std::pair<std::string, std::string> bigram;
        int min_rank = std::numeric_limits<int>::max();

        for (const auto& p : pairs) {
            if (bpe_ranks_.count(p)) {
                if (bpe_ranks_.at(p) < min_rank) {
                    min_rank = bpe_ranks_.at(p);
                    bigram = p;
                }
            }
        }

        if (min_rank == std::numeric_limits<int>::max()) {
            break;
        }

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            bool found = false;
            size_t j = i;
            while(j < word.size()){
                if(word[j] == bigram.first) break;
                j++;
            }
            if(j > i) {
                new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);
            }
            i = j;

            if (i >= word.size()) {
                break;
            }

            if (word[i] == bigram.first && i < word.size() - 1 && word[i+1] == bigram.second) {
                new_word.push_back(bigram.first + bigram.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = new_word;
        if (word.size() == 1) {
            break;
        }
        pairs = get_pairs(word);
    }

    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        result += word[i] + (i == word.size() - 1 ? "" : " ");
    }
    cache_[token] = result;
    return result;
}

std::vector<int> SimpleTokenizer::encode(std::string text) {
    std::vector<int> bpe_tokens;
    text = this->clean_fn_(text);

    std::smatch m;
    auto it = std::sregex_iterator(text.begin(), text.end(), pat_);
    auto end = std::sregex_iterator();

    for (; it != end; ++it) {
        std::string token_str = it->str();
        std::string encoded_token;
        for (unsigned char b : token_str) {
            encoded_token += this->byte_encoder_.at(b - 33);
        }

        std::string bpe_result = bpe(encoded_token);

        std::stringstream ss(bpe_result);
        std::string sub_token;
        while (ss >> sub_token) {
            if (encoder_.count(sub_token)) {
                bpe_tokens.push_back(encoder_.at(sub_token));
            }
        }
    }
    return bpe_tokens;
}

std::vector<std::vector<int>> SimpleTokenizer::operator()(const std::vector<std::string>& texts, std::optional<int> context_length_opt) {
    int current_context_length = context_length_opt.value_or(this->context_length_);

    std::vector<std::vector<int>> all_tokens;
    for (const auto& text : texts) {
        auto encoded = encode(text);
        std::vector<int> tokens = {sot_token_id_};
        tokens.insert(tokens.end(), encoded.begin(), encoded.end());
        tokens.push_back(eot_token_id_);
        all_tokens.push_back(tokens);
    }

    std::vector<std::vector<int>> result(all_tokens.size(), std::vector<int>(current_context_length, 0));
    for (size_t i = 0; i < all_tokens.size(); ++i) {
        auto& tokens = all_tokens[i];
        if (tokens.size() > current_context_length) {
            tokens.resize(current_context_length);
            tokens.back() = eot_token_id_;
        }
        std::copy(tokens.begin(), tokens.end(), result[i].begin());
    }

    return result;
}

std::vector<std::vector<int>> SimpleTokenizer::operator()(const std::string& text) {
    return this->operator()(std::vector<std::string>{text});
}
