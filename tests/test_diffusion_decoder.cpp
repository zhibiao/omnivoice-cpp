#include "diffusion_decoder/diffusion_decoder.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

class FakeRunner final : public omnivoice::DiffusionLmRunner {
public:
    explicit FakeRunner(int32_t codebooks, int32_t frames, int32_t vocab)
        : codebooks_(codebooks),
          frames_(frames),
          vocab_(vocab) {
    }

    omnivoice::Status init_graph(
        const omnivoice::OmniVoiceLmInput& conditional,
        const omnivoice::OmniVoiceLmInput& unconditional,
        bool read_host_logits,
        float guidance_scale) override {
        ++init_calls;
        conditional_length = conditional.sequence_length;
        unconditional_length = unconditional.sequence_length;
        if (read_host_logits || guidance_scale != 2.0f) {
            return omnivoice::Status::error(
                "FakeRunner received unexpected graph options");
        }
        return omnivoice::Status::ok();
    }

    omnivoice::Status run_diffusion(
        const omnivoice::OmniVoiceLmInput& conditional,
        const omnivoice::OmniVoiceLmInput& unconditional,
        omnivoice::OmniVoiceLmPackedOutput& output) override {
        (void)conditional;
        output = {};
        output.predictions.resize(
            static_cast<size_t>(codebooks_) * frames_);
        output.confidence.resize(output.predictions.size());
        int32_t masked = 0;
        for (size_t cell = 0; cell < output.predictions.size(); ++cell) {
            if (unconditional.input_ids[cell] == 9) {
                ++masked;
            }
            output.predictions[cell] =
                static_cast<int32_t>(cell % 8) + 1;
            output.confidence[cell] =
                1000.0f - static_cast<float>(cell) * 0.01f;
        }
        masked_per_call.push_back(masked);
        return omnivoice::Status::ok();
    }

    std::vector<int32_t> masked_per_call;
    int32_t init_calls = 0;
    int32_t conditional_length = 0;
    int32_t unconditional_length = 0;

private:
    int32_t codebooks_;
    int32_t frames_;
    int32_t vocab_;
};

} // namespace

int main() {
    constexpr int32_t codebooks = 2;
    constexpr int32_t frames = 4;
    constexpr int32_t vocab = 10;
    constexpr int32_t mask = 9;
    constexpr int32_t style = 2;
    constexpr int32_t text = 1;
    constexpr int32_t reference = 2;
    constexpr int32_t target_start = style + text + reference;
    constexpr int32_t sequence_length = target_start + frames;

    omnivoice::OmniVoiceLMConfig config;
    config.audio_vocab_size = vocab;
    config.audio_mask_id = mask;
    config.num_audio_codebooks = codebooks;

    omnivoice::PromptPlan prompt;
    prompt.codebooks = codebooks;
    prompt.layout.style_frames = style;
    prompt.layout.text_frames = text;
    prompt.layout.reference_start = style + text;
    prompt.layout.target_start = target_start;
    prompt.layout.target_frames = frames;
    prompt.layout.sequence_length = sequence_length;
    prompt.input_ids.assign(
        static_cast<size_t>(codebooks) *
            sequence_length,
        mask);
    prompt.audio_mask.assign(sequence_length, 0);
    for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
        const size_t row =
            static_cast<size_t>(codebook) * sequence_length;
        for (int32_t index = 0; index < target_start; ++index) {
            prompt.input_ids[row + static_cast<size_t>(index)] =
                10 + index;
        }
        prompt.input_ids[
            row + static_cast<size_t>(prompt.layout.reference_start)] = 2;
        prompt.input_ids[
            row +
            static_cast<size_t>(prompt.layout.reference_start + 1)] = 3;
    }
    for (int32_t index = prompt.layout.reference_start;
         index < sequence_length;
         ++index) {
        prompt.audio_mask[static_cast<size_t>(index)] = 1;
    }

    omnivoice::VoiceCloneConfig options;
    options.num_step = 3;
    options.t_shift = 1.0f;
    options.guidance_scale = 2.0f;
    options.layer_penalty_factor = 0.0f;
    options.position_temperature = 0.0f;
    options.class_temperature = 0.0f;

    FakeRunner runner(codebooks, frames, vocab);
    omnivoice::DiffusionDecodeResult result;
    std::vector<std::string> logs;
    omnivoice::Status status = omnivoice::run_diffusion_decoder(
        prompt,
        config,
        options,
        1234,
        runner,
        result,
        [&](const std::string& line) {
            logs.push_back(line);
        });
    if (!status) {
        std::cerr << "diffusion decode failed: "
                  << status.message() << '\n';
        return 1;
    }
    if (logs.size() != 5 ||
        logs.front().find("diffusion:") == std::string::npos ||
        logs[1].find("single packed graph") == std::string::npos ||
        logs.back().find("remaining=0") == std::string::npos) {
        std::cerr << "unexpected diffusion progress logs\n";
        return 1;
    }

    if (result.counts_per_step !=
        std::vector<int32_t>({3, 3, 2})) {
        std::cerr << "unexpected per-step counts\n";
        return 1;
    }
    if (runner.init_calls != 1 ||
        runner.conditional_length != sequence_length ||
        runner.unconditional_length != frames) {
        std::cerr << "unexpected diffusion graph initialization\n";
        return 1;
    }
    if (runner.masked_per_call !=
        std::vector<int32_t>({8, 5, 2})) {
        std::cerr << "committed token state was not propagated\n";
        return 1;
    }
    if (result.target_codes.codebooks != codebooks ||
        result.target_codes.frames != frames ||
        result.target_codes.data.size() !=
            static_cast<size_t>(codebooks * frames)) {
        std::cerr << "invalid diffusion code output shape\n";
        return 1;
    }
    for (size_t cell = 0;
         cell < result.target_codes.data.size();
         ++cell) {
        const int32_t expected =
            static_cast<int32_t>(cell % 8) + 1;
        if (result.target_codes.data[cell] != expected) {
            std::cerr << "unexpected generated token\n";
            return 1;
        }
    }

    prompt.input_ids[
        static_cast<size_t>(target_start)] =
        1;
    if (omnivoice::run_diffusion_decoder(
            prompt,
            config,
            options,
            1234,
            runner,
            result)) {
        std::cerr << "non-MASK target region must be rejected\n";
        return 1;
    }
    return 0;
}
