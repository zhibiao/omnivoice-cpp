#include "config.h"
#include "diffusion_decoder/diffusion_decoder.h"
#include "gguf_model.h"
#include "prompt_builder.h"
#include "qwen3_lm/qwen3_model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Fixture {
    int32_t codebooks = 0;
    int32_t reference_frames = 0;
    int32_t target_frames = 0;
    int32_t sequence_length = 0;
    std::vector<int32_t> reference_codes;
    std::vector<int32_t> target_codes;
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> audio_mask;
};

Fixture read_fixture() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/voice_clone_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open diffusion fixture");
    }
    char magic[8] = {};
    std::array<int32_t, 5> header{};
    file.read(magic, sizeof(magic));
    file.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(
            header.size() * sizeof(int32_t)));
    if (!file ||
        std::memcmp(magic, "HIGCLONE", 8) != 0 ||
        header[0] <= 0 ||
        header[1] <= 0 ||
        header[2] <= 0 ||
        header[3] <= 0) {
        throw std::runtime_error("invalid diffusion fixture");
    }

    Fixture fixture;
    fixture.codebooks = header[0];
    fixture.reference_frames = header[1];
    fixture.target_frames = header[2];
    fixture.sequence_length = header[3];
    fixture.reference_codes.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.reference_frames));
    fixture.target_codes.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.target_frames));
    fixture.input_ids.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.sequence_length));
    fixture.audio_mask.resize(
        static_cast<size_t>(fixture.sequence_length));

    file.read(
        reinterpret_cast<char*>(fixture.reference_codes.data()),
        static_cast<std::streamsize>(
            fixture.reference_codes.size() * sizeof(int32_t)));
    file.read(
        reinterpret_cast<char*>(fixture.target_codes.data()),
        static_cast<std::streamsize>(
            fixture.target_codes.size() * sizeof(int32_t)));
    file.read(
        reinterpret_cast<char*>(fixture.input_ids.data()),
        static_cast<std::streamsize>(
            fixture.input_ids.size() * sizeof(int32_t)));
    file.read(
        reinterpret_cast<char*>(fixture.audio_mask.data()),
        static_cast<std::streamsize>(fixture.audio_mask.size()));
    if (!file) {
        throw std::runtime_error("truncated diffusion fixture");
    }
    return fixture;
}

omnivoice::PromptPlan make_prompt(const Fixture& fixture) {
    omnivoice::PromptPlan prompt;
    prompt.codebooks = fixture.codebooks;
    prompt.layout.style_frames = 0;
    prompt.layout.text_frames = 0;
    prompt.layout.reference_start = 0;
    prompt.layout.target_frames = fixture.target_frames;
    prompt.layout.target_start =
        fixture.sequence_length - fixture.target_frames;
    prompt.layout.sequence_length = fixture.sequence_length;
    prompt.input_ids = fixture.input_ids;
    prompt.audio_mask = fixture.audio_mask;
    return prompt;
}

omnivoice::VoiceCloneConfig decode_options() {
    omnivoice::VoiceCloneConfig options;
    options.num_step = 4;
    options.guidance_scale = 2.0f;
    options.t_shift = 0.1f;
    options.layer_penalty_factor = 5.0f;
    options.position_temperature = 0.0f;
    options.class_temperature = 0.0f;
    return options;
}

} // namespace

int main() {
    if (!std::filesystem::is_regular_file(OMNIVOICE_LM_GGUF)) {
        std::cerr << "OmniVoice LM GGUF is unavailable; skipping\n";
        return 77;
    }

    try {
        const Fixture fixture = read_fixture();
        std::unique_ptr<omnivoice::GGUFModel> gguf;
        omnivoice::Status status =
            omnivoice::GGUFModel::open(OMNIVOICE_LM_GGUF, gguf);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        omnivoice::OmniVoiceLMConfig config;
        status = omnivoice::load_omnivoice_lm_config(*gguf, config);
        if (!status) {
            throw std::runtime_error(status.message());
        }

        ggml_backend_load_all();
        const char* device_name = std::getenv("OMNIVOICE_TEST_DEVICE");
        if (device_name == nullptr || device_name[0] == '\0') {
            device_name = "CPU";
        }
        ggml_backend_dev_t device =
            ggml_backend_dev_by_name(device_name);
        ggml_backend_t backend = device == nullptr
            ? nullptr
            : ggml_backend_dev_init(device, nullptr);
        if (backend == nullptr) {
            std::cerr << "backend device unavailable; skipping\n";
            return 77;
        }

        auto model = std::make_unique<omnivoice::OmniVoiceLM>();
        status = model->load(config, *gguf, backend);
        if (!status) {
            throw std::runtime_error(status.message());
        }

        const omnivoice::PromptPlan prompt = make_prompt(fixture);
        const omnivoice::VoiceCloneConfig options = decode_options();

        std::vector<std::string> logs;
        omnivoice::DiffusionDecodeResult result;
        status = omnivoice::run_diffusion_decoder(
            prompt,
            config,
            options,
            1234,
            *model,
            result,
            [&](const std::string& line) { logs.push_back(line); });
        model->release_diffusion_graph();
        if (!status) {
            throw std::runtime_error(status.message());
        }

        bool diffusion_logged = false;
        for (const std::string& line : logs) {
            if (line.find("diffusion step") != std::string::npos) {
                diffusion_logged = true;
            }
        }

        const int32_t cells =
            fixture.codebooks * fixture.target_frames;
        const int32_t selected_total = std::accumulate(
            result.counts_per_step.begin(),
            result.counts_per_step.end(),
            0);
        const bool counts_valid =
            static_cast<int32_t>(result.counts_per_step.size()) ==
                options.num_step &&
            selected_total == cells &&
            std::all_of(
                result.counts_per_step.begin(),
                result.counts_per_step.end(),
                [](int32_t count) { return count >= 0; });
        const bool codes_valid =
            result.target_codes.codebooks == fixture.codebooks &&
            result.target_codes.frames == fixture.target_frames &&
            result.target_codes.data.size() ==
                fixture.target_codes.size();
        bool golden_match = codes_valid;
        int32_t first_mismatch = -1;
        if (codes_valid) {
            for (size_t index = 0;
                 index < fixture.target_codes.size();
                 ++index) {
                if (result.target_codes.data[index] !=
                    fixture.target_codes[index]) {
                    golden_match = false;
                    first_mismatch = static_cast<int32_t>(index);
                    break;
                }
            }
        }

        omnivoice::DiffusionDecodeResult repeat;
        status = omnivoice::run_diffusion_decoder(
            prompt,
            config,
            options,
            1234,
            *model,
            repeat);
        model->release_diffusion_graph();
        if (!status) {
            throw std::runtime_error(status.message());
        }
        const bool deterministic =
            repeat.target_codes.data == result.target_codes.data;

        omnivoice::DiffusionDecodeResult alternate_seed;
        status = omnivoice::run_diffusion_decoder(
            prompt,
            config,
            options,
            987654321,
            *model,
            alternate_seed);
        model->release_diffusion_graph();
        if (!status) {
            throw std::runtime_error(status.message());
        }
        const bool seed_independent =
            alternate_seed.target_codes.data == result.target_codes.data;

        omnivoice::PromptPlan invalid = prompt;
        invalid.input_ids[
            static_cast<size_t>(prompt.layout.target_start)] = 0;
        omnivoice::DiffusionDecodeResult rejected;
        status = omnivoice::run_diffusion_decoder(
            invalid,
            config,
            options,
            1234,
            *model,
            rejected);
        const bool unmasked_rejected = !status;
        model->release_diffusion_graph();

        std::cout << "diffusion_codes_match: "
                  << (golden_match ? "true" : "false") << '\n'
                  << "diffusion_first_mismatch: "
                  << first_mismatch << '\n'
                  << "diffusion_counts_valid: "
                  << (counts_valid ? "true" : "false") << '\n'
                  << "diffusion_selected_total: "
                  << selected_total << '\n'
                  << "diffusion_deterministic: "
                  << (deterministic ? "true" : "false") << '\n'
                  << "diffusion_seed_independent: "
                  << (seed_independent ? "true" : "false") << '\n'
                  << "diffusion_unmasked_rejected: "
                  << (unmasked_rejected ? "true" : "false") << '\n'
                  << "diffusion_device: " << device_name << '\n';

        model.reset();
        ggml_backend_free(backend);
        if (!diffusion_logged ||
            !counts_valid ||
            !golden_match ||
            !deterministic ||
            !seed_independent ||
            !unmasked_rejected) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "diffusion decoder test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
