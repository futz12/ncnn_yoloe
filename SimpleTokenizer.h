#ifndef SIMPLE_TOKENIZER_H
#define SIMPLE_TOKENIZER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <regex>
#include <optional>
#include <functional>


class SimpleTokenizer {
public:
    explicit SimpleTokenizer(
        const std::string& bpe_path,
        int context_length = 77,
        const std::vector<std::string>& additional_special_tokens = {}
    );

    std::vector<std::vector<int>> operator()(
        const std::vector<std::string>& texts,
        std::optional<int> context_length = std::nullopt
    );

    std::vector<std::vector<int>> operator()(const std::string& text);

    std::vector<int> encode(std::string text);

    std::string decode(const std::vector<int>& tokens);

private:
    int vocab_size_;
    int context_length_;
    int sot_token_id_;
    int eot_token_id_;

    std::vector<std::string> byte_encoder_;

    std::unordered_map<std::string, int> encoder_;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks_;

    mutable std::unordered_map<std::string, std::string> cache_;

    std::regex pat_;

    std::vector<int> all_special_ids_;
    std::function<std::string(std::string)> clean_fn_;

    std::string bpe(const std::string& token) const;
};

#endif // SIMPLE_TOKENIZER_H
