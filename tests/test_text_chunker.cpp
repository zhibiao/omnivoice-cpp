#include "text_tokenizer/text_chunker.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/text_chunker_cases.json");
    if (!file) {
        std::cerr << "failed to open text chunker golden\n";
        return 1;
    }
    nlohmann::json golden;
    file >> golden;
    for (const auto& test : golden) {
        std::vector<std::string> chunks;
        const omnivoice::Status status =
            omnivoice::chunk_text_punctuation(
                test.at("text").get<std::string>(),
                test.at("chunk_length").get<int32_t>(),
                test.at("minimum_chunk_length").get<int32_t>(),
                chunks);
        const std::vector<std::string> expected =
            test.at("chunks").get<std::vector<std::string>>();
        if (!status || chunks != expected) {
            std::cerr << "text chunk mismatch\ninput: "
                      << test.at("text").get<std::string>()
                      << "\nexpected:";
            for (const std::string& chunk : expected) {
                std::cerr << "\n  [" << chunk << ']';
            }
            std::cerr << "\nactual:";
            for (const std::string& chunk : chunks) {
                std::cerr << "\n  [" << chunk << ']';
            }
            std::cerr << '\n';
            return 1;
        }
    }

    std::vector<std::string> planned;
    omnivoice::Status status = omnivoice::plan_text_chunks(
        "Short text.",
        750,
        25,
        15.0f,
        30.0f,
        planned);
    if (!status ||
        planned != std::vector<std::string>({"Short text."})) {
        std::cerr << "exactly-threshold text must not be chunked\n";
        return 1;
    }
    status = omnivoice::plan_text_chunks(
        "First sentence. Second sentence. Third sentence. Fourth.",
        2000,
        25,
        15.0f,
        30.0f,
        planned);
    if (!status || planned.size() < 2) {
        std::cerr << "long text planning did not split\n";
        return 1;
    }
    return 0;
}
