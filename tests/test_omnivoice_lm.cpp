#include "config.h"
#include "gguf_model.h"
#include "qwen3_lm/qwen3_model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Fixture {
    int32_t sequence = 0;
    int32_t codebooks = 0;
    int32_t target_start = 0;
    int32_t target_frames = 0;
    int32_t hidden = 0;
    int32_t audio_vocab = 0;
    std::vector<int32_t> input_ids;
    std::vector<float> audio_mask;
    std::vector<float> hidden_value;
    std::vector<float> logits;
};

Fixture read_fixture() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/omnivoice_lm_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open OmniVoice LM fixture");
    }
    char magic[8] = {};
    std::array<int32_t, 7> header{};
    file.read(magic, sizeof(magic));
    file.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(header.size() * sizeof(int32_t)));
    if (!file ||
        std::memcmp(magic, "OVLM0001", 8) != 0 ||
        header[6] != 4) {
        throw std::runtime_error("invalid OmniVoice LM fixture");
    }

    Fixture fixture;
    fixture.sequence = header[0];
    fixture.codebooks = header[1];
    fixture.target_start = header[2];
    fixture.target_frames = header[3];
    fixture.hidden = header[4];
    fixture.audio_vocab = header[5];

    fixture.input_ids.resize(
        static_cast<size_t>(fixture.codebooks) *
        fixture.sequence);
    file.read(
        reinterpret_cast<char*>(fixture.input_ids.data()),
        static_cast<std::streamsize>(
            fixture.input_ids.size() * sizeof(int32_t)));
    fixture.audio_mask.resize(static_cast<size_t>(fixture.sequence));
    file.read(
        reinterpret_cast<char*>(fixture.audio_mask.data()),
        static_cast<std::streamsize>(
            fixture.audio_mask.size() * sizeof(float)));
    fixture.hidden_value.resize(
        static_cast<size_t>(fixture.hidden) *
        fixture.sequence);
    file.read(
        reinterpret_cast<char*>(fixture.hidden_value.data()),
        static_cast<std::streamsize>(
            fixture.hidden_value.size() * sizeof(float)));
    fixture.logits.resize(
        static_cast<size_t>(fixture.codebooks) *
        fixture.target_frames *
        fixture.audio_vocab);
    file.read(
        reinterpret_cast<char*>(fixture.logits.data()),
        static_cast<std::streamsize>(
            fixture.logits.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("truncated OmniVoice LM fixture");
    }
    return fixture;
}

float max_abs_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float result = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        result = std::max(
            result,
            std::abs(actual[index] - expected[index]));
    }
    return result;
}

} // namespace

int main() {
    if (!std::filesystem::is_regular_file(OMNIVOICE_LM_GGUF)) {
        std::cerr << "OmniVoice LM GGUF is not available; skipping\n";
        return 77;
    }

    try {
        std::unique_ptr<omnivoice::GGUFModel> gguf;
        omnivoice::Status status = omnivoice::GGUFModel::open(
            OMNIVOICE_LM_GGUF,
            gguf);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        omnivoice::OmniVoiceLMConfig config;
        status = omnivoice::load_omnivoice_lm_config(*gguf, config);
        if (!status) {
            throw std::runtime_error(status.message());
        }

        ggml_backend_load_all();
        ggml_backend_t backend =
            ggml_backend_init_by_name("CPU", nullptr);
        if (backend == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }

        auto model = std::make_unique<omnivoice::OmniVoiceLM>();
        status = model->load(config, *gguf, backend);
        if (!status) {
            model.reset();
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }

        const Fixture fixture = read_fixture();
        omnivoice::OmniVoiceLmInput input;
        input.sequence_length = fixture.sequence;
        input.codebooks = fixture.codebooks;
        input.target_start = fixture.target_start;
        input.target_frames = fixture.target_frames;
        input.input_ids = fixture.input_ids;
        for (float value : fixture.audio_mask) {
            input.audio_mask.push_back(value != 0.0f ? 1 : 0);
        }

        omnivoice::OmniVoiceLmInput unconditional;
        unconditional.sequence_length = input.target_frames;
        unconditional.codebooks = input.codebooks;
        unconditional.target_start = 0;
        unconditional.target_frames = input.target_frames;
        unconditional.audio_mask.assign(
            static_cast<size_t>(input.target_frames),
            1);
        unconditional.input_ids.resize(
            static_cast<size_t>(input.codebooks) *
            static_cast<size_t>(input.target_frames));
        for (int32_t codebook = 0; codebook < input.codebooks; ++codebook) {
            std::copy(
                input.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(
                        codebook * input.sequence_length +
                        input.target_start),
                input.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(
                        codebook * input.sequence_length +
                        input.target_start +
                        input.target_frames),
                unconditional.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(
                        codebook * input.target_frames));
        }

        status = model->init_diffusion_graph(
            input,
            unconditional,
            true,
            true,
            0.0f);
        if (!status) {
            model.reset();
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }
        omnivoice::OmniVoiceLmPackedOutput packed;
        status = model->run_diffusion_graph(
            input,
            unconditional,
            packed);
        if (!status) {
            model.reset();
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }

        const size_t conditional_hidden_values =
            static_cast<size_t>(config.llm.hidden_size) *
            static_cast<size_t>(input.sequence_length);
        const std::vector<float> conditional_hidden(
            packed.hidden.begin(),
            packed.hidden.begin() +
                static_cast<std::ptrdiff_t>(conditional_hidden_values));
        const float hidden_error = max_abs_error(
            conditional_hidden,
            fixture.hidden_value);
        const float logits_error = max_abs_error(
            packed.conditional_target_logits,
            fixture.logits);
        std::cout << "lm_hidden_max_error: " << hidden_error << '\n'
                  << "lm_logits_max_error: " << logits_error << '\n';
        if (!std::isfinite(hidden_error) ||
            !std::isfinite(logits_error) ||
            hidden_error > 5e-3f ||
            logits_error > 5e-3f) {
            model.reset();
            ggml_backend_free(backend);
            return 1;
        }
        model.reset();
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OmniVoice LM test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
