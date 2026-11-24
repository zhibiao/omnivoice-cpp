#include "omnivoice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
    std::vector<float> hidden;
    std::vector<float> target_logits;
    std::vector<float> waveform;
};

Fixture read_fixture() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/voice_clone_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open Voice Clone fixture");
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
        header[3] <= 0 ||
        header[4] <= 0) {
        throw std::runtime_error("invalid Voice Clone fixture");
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
    fixture.hidden.resize(
        static_cast<size_t>(fixture.sequence_length) * 1024);
    fixture.target_logits.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.target_frames) *
        1025);
    fixture.waveform.resize(static_cast<size_t>(header[4]));

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
    file.read(
        reinterpret_cast<char*>(fixture.hidden.data()),
        static_cast<std::streamsize>(
            fixture.hidden.size() * sizeof(float)));
    file.read(
        reinterpret_cast<char*>(fixture.target_logits.data()),
        static_cast<std::streamsize>(
            fixture.target_logits.size() * sizeof(float)));
    file.read(
        reinterpret_cast<char*>(fixture.waveform.data()),
        static_cast<std::streamsize>(
            fixture.waveform.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("truncated Voice Clone fixture");
    }
    return fixture;
}

} // namespace

int main() {
    if (!std::filesystem::is_regular_file(OMNIVOICE_LM_GGUF) ||
        !std::filesystem::is_regular_file(OMNIVOICE_AUDIO_GGUF)) {
        std::cerr << "OmniVoice model GGUF is unavailable; skipping\n";
        return 77;
    }

    try {
        const Fixture fixture = read_fixture();
        const char* device = std::getenv("OMNIVOICE_TEST_DEVICE");
        omnivoice::OmniVoiceModelParams params;
        params.device = device == nullptr ? "CPU" : device;
        omnivoice::OmniVoiceModel model;
        omnivoice::Status status = model.load(
            OMNIVOICE_LM_GGUF,
            OMNIVOICE_AUDIO_GGUF,
            params);
        if (!status) {
            if (device != nullptr) {
                std::cerr << "requested backend unavailable; skipping\n";
                return 77;
            }
            throw std::runtime_error(status.message());
        }

        omnivoice::VoiceClonePrompt prompt;
        prompt.reference_text = "Hello.";
        prompt.language = "en";
        prompt.reference_rms = 0.2f;
        prompt.sample_rate = 24000;
        prompt.denoise = true;
        prompt.reference_codes.codebooks = fixture.codebooks;
        prompt.reference_codes.frames = fixture.reference_frames;
        prompt.reference_codes.data = fixture.reference_codes;

        omnivoice::VoiceCloneRequest request;
        request.target_text = "Hi.";
        request.language = "en";
        request.target_frames = fixture.target_frames;
        request.seed = 1234;
        omnivoice::VoiceCloneConfig options;
        options.num_step = 4;
        options.guidance_scale = 2.0f;
        options.t_shift = 0.1f;
        options.layer_penalty_factor = 5.0f;
        options.position_temperature = 0.0f;
        options.class_temperature = 0.0f;
        options.postprocess_output = false;
        options.pad_duration = 0.0f;
        options.fade_duration = 0.0f;
        request.generation_config = options;
        std::vector<std::string> logs;
        request.log = [&](const std::string& line) {
            logs.push_back(line);
        };

        omnivoice::VoiceCloneOutput output;
        status = model.generate_voice_clone(prompt, request, output);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        bool diffusion_logged = false;
        for (const std::string& line : logs) {
            if (line.find("diffusion step") != std::string::npos) {
                diffusion_logged = true;
            }
        }
        if (!diffusion_logged) {
            std::cerr << "Voice Clone did not emit diffusion logs\n";
            return 1;
        }

        if (output.prompt_plan.input_ids != fixture.input_ids ||
            output.prompt_plan.audio_mask != fixture.audio_mask) {
            std::cerr << "Voice Clone prompt tensor mismatch\n";
            return 1;
        }
        if (output.counts_per_step.size() != 4 ||
            std::accumulate(
                output.counts_per_step.begin(),
                output.counts_per_step.end(),
                0) !=
                fixture.codebooks * fixture.target_frames) {
            std::cerr << "Voice Clone schedule mismatch\n";
            return 1;
        }
        const bool cuda_device =
            device != nullptr &&
            std::string(device) != "CPU";
        const bool codes_match =
            output.target_codes.codebooks == fixture.codebooks &&
            output.target_codes.frames == fixture.target_frames &&
            output.target_codes.data == fixture.target_codes;
        int32_t code_mismatches = 1 << 30;
        if (output.target_codes.codebooks == fixture.codebooks &&
            output.target_codes.frames == fixture.target_frames &&
            output.target_codes.data.size() ==
                fixture.target_codes.size()) {
            code_mismatches = 0;
            for (size_t index = 0;
                 index < fixture.target_codes.size();
                 ++index) {
                if (output.target_codes.data[index] !=
                    fixture.target_codes[index]) {
                    ++code_mismatches;
                }
            }
        }
        const int32_t allowed_mismatches = 0;
        if (!codes_match) {
            std::cerr << "Voice Clone generated codes mismatch\n";
            std::cerr << "mismatch count: "
                      << code_mismatches << '\n';
        }
        if (code_mismatches > allowed_mismatches) {
            return 1;
        }
        if (output.audio.sample_rate != 24000 ||
            output.audio.channels != 1 ||
            output.audio.samples.size() != fixture.waveform.size()) {
            std::cerr << "Voice Clone waveform shape mismatch\n";
            return 1;
        }
        float maximum_error = 0.0f;
        if (codes_match) {
            for (size_t index = 0;
                 index < fixture.waveform.size();
                 ++index) {
                maximum_error = std::max(
                    maximum_error,
                    std::abs(
                        output.audio.samples[index] -
                        fixture.waveform[index]));
            }
        }
        bool finite_audio = true;
        for (float sample : output.audio.samples) {
            finite_audio = finite_audio && std::isfinite(sample);
        }
        std::cout << "voice_clone_codes_match: "
                  << (codes_match ? "true" : "false") << '\n'
                  << "voice_clone_code_mismatches: "
                  << code_mismatches << '\n'
                  << "voice_clone_waveform_max_error: "
                  << maximum_error << '\n';
        if (!finite_audio ||
            (codes_match &&
             (!std::isfinite(maximum_error) ||
              maximum_error > 5e-4f))) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Voice Clone test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
