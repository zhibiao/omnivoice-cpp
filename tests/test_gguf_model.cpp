#include "gguf_model.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool write_prefix(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::streamsize bytes) {
    std::ifstream input(source, std::ios::binary);
    if (!input) {
        return false;
    }
    std::vector<char> buffer(static_cast<size_t>(bytes));
    input.read(buffer.data(), bytes);
    if (!input) {
        return false;
    }
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    output.write(buffer.data(), bytes);
    return static_cast<bool>(output);
}

} // namespace

int main() {
    const std::string data_dir = OMNIVOICE_TEST_DATA_DIR;
    const std::filesystem::path config_path =
        std::filesystem::path(data_dir) / "omnivoice-lm-config-m3.gguf";
    const std::filesystem::path not_gguf_path =
        std::filesystem::path(data_dir) / "text_chunker_cases.json";
    if (!std::filesystem::is_regular_file(config_path) ||
        !std::filesystem::is_regular_file(not_gguf_path)) {
        std::cerr << "GGUF test data is unavailable; skipping\n";
        return 77;
    }

    try {
        // Missing files are rejected.
        std::unique_ptr<omnivoice::GGUFModel> model;
        omnivoice::Status status = omnivoice::GGUFModel::open(
            (std::filesystem::path(data_dir) / "missing.gguf").string(),
            model);
        const bool missing_rejected = !status && model == nullptr;

        // Non-GGUF payloads are rejected.
        status = omnivoice::GGUFModel::open(not_gguf_path.string(), model);
        const bool foreign_rejected = !status && model == nullptr;

        // Truncated GGUF payloads are rejected.
        const std::filesystem::path truncated_path =
            std::filesystem::temp_directory_path() /
            "omnivoice_truncated.gguf";
        std::filesystem::remove(truncated_path);
        const bool prefix_written = write_prefix(
            config_path,
            truncated_path,
            64);
        bool truncated_rejected = false;
        if (prefix_written) {
            status = omnivoice::GGUFModel::open(
                truncated_path.string(),
                model);
            truncated_rejected = !status && model == nullptr;
        }
        std::filesystem::remove(truncated_path);

        // A valid model exposes typed metadata with fallbacks.
        status = omnivoice::GGUFModel::open(config_path.string(), model);
        if (!status || model == nullptr) {
            throw std::runtime_error(
                status ? "GGUF model is null" : status.message());
        }
        const bool schema_present =
            model->has_key("omnivoice.schema_version");
        const bool schema_u32 =
            model->key_type("omnivoice.schema_version") ==
            GGUF_TYPE_UINT32;
        const bool role_present =
            model->has_key("omnivoice.file_role");
        const bool role_string =
            model->key_type("omnivoice.file_role") ==
            GGUF_TYPE_STRING;
        const bool llm_blocks_present =
            model->has_key("omnivoice.llm.block_count");
        const bool missing_key_absent =
            !model->has_key("omnivoice.missing.key");
        const bool missing_type_count =
            model->key_type("omnivoice.missing.key") ==
            GGUF_TYPE_COUNT;
        const bool missing_fallback =
            model->get_i32("omnivoice.missing.key", -7) == -7 &&
            model->get_string("omnivoice.missing.key", "fallback") ==
                "fallback";
        const bool missing_tensor_null =
            model->tensor("omnivoice.missing.tensor") == nullptr;
        const bool tensor_count_non_negative =
            model->tensor_count() >= 0;

        // A config-only GGUF has metadata but no tensors.
        const bool config_only =
            model->tensor_count() == 0;

        std::cout << "gguf_missing_rejected: "
                  << (missing_rejected ? "true" : "false") << '\n'
                  << "gguf_foreign_rejected: "
                  << (foreign_rejected ? "true" : "false") << '\n'
                  << "gguf_truncated_rejected: "
                  << (truncated_rejected ? "true" : "false") << '\n'
                  << "gguf_schema_present: "
                  << (schema_present ? "true" : "false") << '\n'
                  << "gguf_schema_u32: "
                  << (schema_u32 ? "true" : "false") << '\n'
                  << "gguf_role_present: "
                  << (role_present ? "true" : "false") << '\n'
                  << "gguf_role_string: "
                  << (role_string ? "true" : "false") << '\n'
                  << "gguf_llm_blocks_present: "
                  << (llm_blocks_present ? "true" : "false") << '\n'
                  << "gguf_missing_key_absent: "
                  << (missing_key_absent ? "true" : "false") << '\n'
                  << "gguf_missing_type_count: "
                  << (missing_type_count ? "true" : "false") << '\n'
                  << "gguf_missing_fallback: "
                  << (missing_fallback ? "true" : "false") << '\n'
                  << "gguf_missing_tensor_null: "
                  << (missing_tensor_null ? "true" : "false") << '\n'
                  << "gguf_tensor_count: " << model->tensor_count()
                  << '\n'
                  << "gguf_config_only: "
                  << (config_only ? "true" : "false") << '\n';

        model.reset();
        if (!missing_rejected || !foreign_rejected ||
            !truncated_rejected || !schema_present || !schema_u32 ||
            !role_present || !role_string || !llm_blocks_present ||
            !missing_key_absent || !missing_type_count ||
            !missing_fallback || !missing_tensor_null ||
            !tensor_count_non_negative) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GGUF model test failed: " << error.what() << '\n';
        return 1;
    }
}
