#include "config.h"
#include "diffusion_decoder/token_selector.h"
#include "gguf_model.h"
#include "qwen3_lm/qwen3_model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
    int32_t codebooks = 0;
    int32_t reference_frames = 0;
    int32_t target_frames = 0;
    int32_t sequence_length = 0;
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> audio_mask;
    std::vector<float> target_logits;
};

Fixture read_fixture() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/voice_clone_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open merged diffusion fixture");
    }
    char magic[8] = {};
    std::array<int32_t, 5> header{};
    file.read(magic, sizeof(magic));
    file.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(
            header.size() * sizeof(int32_t)));
    if (!file || std::memcmp(magic, "HIGCLONE", 8) != 0) {
        throw std::runtime_error("invalid merged diffusion fixture");
    }

    Fixture fixture;
    fixture.codebooks = header[0];
    fixture.reference_frames = header[1];
    fixture.target_frames = header[2];
    fixture.sequence_length = header[3];
    fixture.input_ids.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.sequence_length));
    fixture.audio_mask.resize(
        static_cast<size_t>(fixture.sequence_length));
    fixture.target_logits.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.target_frames) *
        1025);

    file.seekg(
        static_cast<std::streamoff>(
            (static_cast<size_t>(fixture.codebooks) *
                 fixture.reference_frames +
             static_cast<size_t>(fixture.codebooks) *
                 fixture.target_frames) *
            sizeof(int32_t)),
        std::ios::cur);
    file.read(
        reinterpret_cast<char*>(fixture.input_ids.data()),
        static_cast<std::streamsize>(
            fixture.input_ids.size() * sizeof(int32_t)));
    file.read(
        reinterpret_cast<char*>(fixture.audio_mask.data()),
        static_cast<std::streamsize>(fixture.audio_mask.size()));
    file.seekg(
        static_cast<std::streamoff>(
            static_cast<size_t>(fixture.sequence_length) * 1024 *
            sizeof(float)),
        std::ios::cur);
    file.read(
        reinterpret_cast<char*>(fixture.target_logits.data()),
        static_cast<std::streamsize>(
            fixture.target_logits.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("truncated merged diffusion fixture");
    }
    return fixture;
}

float maximum_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        error = std::max(
            error,
            std::abs(actual[index] - expected[index]));
    }
    return error;
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

        std::unique_ptr<omnivoice::OmniVoiceLM> model =
            std::make_unique<omnivoice::OmniVoiceLM>();
        status = model->load(config, *gguf, backend);
        if (!status) {
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }

        omnivoice::OmniVoiceLmInput conditional;
        conditional.sequence_length = fixture.sequence_length;
        conditional.codebooks = fixture.codebooks;
        conditional.target_start =
            fixture.sequence_length - fixture.target_frames;
        conditional.target_frames = fixture.target_frames;
        conditional.input_ids = fixture.input_ids;
        conditional.audio_mask.assign(
            fixture.audio_mask.begin(),
            fixture.audio_mask.end());

        omnivoice::OmniVoiceLmInput unconditional;
        unconditional.sequence_length = fixture.target_frames;
        unconditional.codebooks = fixture.codebooks;
        unconditional.target_start = 0;
        unconditional.target_frames = fixture.target_frames;
        unconditional.audio_mask.assign(
            static_cast<size_t>(fixture.target_frames),
            1);
        unconditional.input_ids.resize(
            static_cast<size_t>(fixture.codebooks) *
            static_cast<size_t>(fixture.target_frames));
        for (int32_t codebook = 0; codebook < fixture.codebooks; ++codebook) {
            std::copy(
                conditional.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(
                        codebook * fixture.sequence_length +
                        conditional.target_start),
                conditional.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(
                        codebook * fixture.sequence_length +
                        conditional.target_start +
                        fixture.target_frames),
                unconditional.input_ids.begin() +
                    static_cast<std::ptrdiff_t>(
                        codebook * fixture.target_frames));
        }

        status = model->init_diffusion_graph(
            conditional,
            unconditional,
            true,
            false,
            2.0f);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        omnivoice::OmniVoiceLmPackedOutput packed;
        status = model->run_diffusion_graph(
            conditional,
            unconditional,
            packed);
        if (!status) {
            throw std::runtime_error(status.message());
        }

        const float golden_error = maximum_error(
            packed.conditional_target_logits,
            fixture.target_logits);

        std::vector<int32_t> all_masked(
            static_cast<size_t>(fixture.codebooks) *
                static_cast<size_t>(fixture.target_frames),
            config.audio_mask_id);
        omnivoice::TokenSelectionInput selector_input;
        selector_input.codebooks = fixture.codebooks;
        selector_input.frames = fixture.target_frames;
        selector_input.vocab_size = config.audio_vocab_size;
        selector_input.mask_id = config.audio_mask_id;
        selector_input.top_k = 1;
        selector_input.cond_logits =
            &packed.conditional_target_logits;
        selector_input.uncond_logits =
            &packed.unconditional_target_logits;
        selector_input.current_tokens = &all_masked;
        omnivoice::TokenSelectorOptions selector_options;
        selector_options.guidance_scale = 2.0f;
        selector_options.layer_penalty_factor = 0.0f;
        selector_options.position_temperature = 0.0f;
        selector_options.class_temperature = 0.0f;
        omnivoice::TokenSelector selector;
        omnivoice::TokenSelectionOutput reference_selection;
        status = selector.select(
            selector_input,
            selector_options,
            reference_selection);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        const float confidence_error = maximum_error(
            packed.confidence,
            reference_selection.confidence);
        const bool predictions_match =
            packed.predictions == reference_selection.predictions;

        const std::vector<float> first_conditional =
            packed.conditional_target_logits;
        for (int32_t& token : unconditional.input_ids) {
            token = (token + 29) % (config.audio_vocab_size - 1);
        }
        status = model->run_diffusion_graph(
            conditional,
            unconditional,
            packed);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        const float reuse_invariance_error = maximum_error(
            packed.conditional_target_logits,
            first_conditional);
        const std::vector<float> changed_unconditional =
            packed.unconditional_target_logits;
        for (int32_t codebook = 0;
             codebook < conditional.codebooks;
             ++codebook) {
            for (int32_t frame = 0;
                 frame < conditional.target_frames;
                 ++frame) {
                const size_t index =
                    static_cast<size_t>(codebook) *
                        static_cast<size_t>(conditional.sequence_length) +
                    static_cast<size_t>(conditional.target_start) +
                    static_cast<size_t>(frame);
                conditional.input_ids[index] =
                    ((codebook * conditional.target_frames + frame) %
                     (config.audio_vocab_size - 1)) +
                    1;
            }
        }
        status = model->run_diffusion_graph(
            conditional,
            unconditional,
            packed);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        const float reverse_reuse_invariance_error = maximum_error(
            packed.unconditional_target_logits,
            changed_unconditional);

        const bool cuda_device = std::string(device_name) != "CPU";
        const float logits_tolerance = cuda_device ? 8e-3f : 5e-3f;
        model->release_diffusion_graph();
        status = model->init_diffusion_graph(
            conditional,
            unconditional,
            true,
            false,
            0.0f);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        omnivoice::OmniVoiceLmPackedOutput unguided;
        status = model->run_diffusion_graph(
            conditional,
            unconditional,
            unguided);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        selector_input.cond_logits = &unguided.conditional_target_logits;
        selector_input.uncond_logits =
            &unguided.unconditional_target_logits;
        selector_input.top_k = static_cast<int32_t>(all_masked.size());
        selector_options.guidance_scale = 0.0f;
        selector_options.layer_penalty_factor = 5.0f;
        omnivoice::TokenSelectionOutput unguided_reference;
        status = selector.select(
            selector_input,
            selector_options,
            unguided_reference);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        const float unguided_confidence_error = maximum_error(
            unguided.confidence,
            unguided_reference.confidence);
        const bool unguided_predictions_match =
            unguided.predictions == unguided_reference.predictions;

        // Both branches run the same LM math: feeding an identical
        // sequence to each must produce identical target logits.
        omnivoice::OmniVoiceLmInput mirrored = unconditional;
        model->release_diffusion_graph();
        status = model->init_diffusion_graph(
            mirrored,
            unconditional,
            true,
            false,
            0.0f);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        omnivoice::OmniVoiceLmPackedOutput equivalent;
        status = model->run_diffusion_graph(
            mirrored,
            unconditional,
            equivalent);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        model->release_diffusion_graph();
        const float branch_equivalence_error = maximum_error(
            equivalent.conditional_target_logits,
            equivalent.unconditional_target_logits);

        std::cout << "merged_diffusion_golden_max_error: "
                  << golden_error << '\n'
                  << "merged_diffusion_confidence_max_error: "
                  << confidence_error << '\n'
                  << "merged_diffusion_predictions_match: "
                  << (predictions_match ? "true" : "false") << '\n'
                  << "merged_diffusion_reuse_invariance_error: "
                  << reuse_invariance_error << '\n'
                  << "merged_diffusion_reverse_invariance_error: "
                  << reverse_reuse_invariance_error << '\n'
                  << "merged_diffusion_unguided_confidence_error: "
                  << unguided_confidence_error << '\n'
                  << "merged_diffusion_unguided_predictions_match: "
                  << (unguided_predictions_match ? "true" : "false")
                  << '\n'
                  << "merged_diffusion_branch_equivalence_error: "
                  << branch_equivalence_error << '\n';

        model.reset();
        ggml_backend_free(backend);
        if (!std::isfinite(golden_error) ||
            golden_error > logits_tolerance ||
            !std::isfinite(confidence_error) ||
            confidence_error > logits_tolerance ||
            !predictions_match ||
            !std::isfinite(reuse_invariance_error) ||
            reuse_invariance_error > logits_tolerance ||
            !std::isfinite(reverse_reuse_invariance_error) ||
            reverse_reuse_invariance_error > logits_tolerance ||
            !std::isfinite(unguided_confidence_error) ||
            unguided_confidence_error > logits_tolerance ||
            !unguided_predictions_match ||
            !std::isfinite(branch_equivalence_error) ||
            branch_equivalence_error > logits_tolerance) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "merged diffusion test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
