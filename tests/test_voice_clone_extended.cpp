#include "omnivoice.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct PromptFixture {
    int32_t codebooks = 0;
    int32_t reference_frames = 0;
    int32_t target_frames = 0;
    std::vector<int32_t> reference_codes;
};

PromptFixture read_prompt_fixture() {
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
        header[2] <= 0) {
        throw std::runtime_error("invalid Voice Clone fixture");
    }

    PromptFixture fixture;
    fixture.codebooks = header[0];
    fixture.reference_frames = header[1];
    fixture.target_frames = header[2];
    fixture.reference_codes.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.reference_frames));
    file.read(
        reinterpret_cast<char*>(fixture.reference_codes.data()),
        static_cast<std::streamsize>(
            fixture.reference_codes.size() * sizeof(int32_t)));
    if (!file) {
        throw std::runtime_error("truncated Voice Clone fixture");
    }
    return fixture;
}

void validate_codes(
    const omnivoice::AudioCodes& codes,
    int32_t expected_frames) {
    if (codes.codebooks != 8 ||
        codes.frames != expected_frames ||
        codes.data.size() != codes.size()) {
        throw std::runtime_error("invalid generated code shape");
    }
    for (int32_t code : codes.data) {
        if (code < 0 || code >= 1024) {
            throw std::runtime_error("generated code is out of range");
        }
    }
}

void validate_audio(const omnivoice::AudioBuffer& audio) {
    if (audio.sample_rate != 24000 ||
        audio.channels != 1 ||
        audio.samples.empty()) {
        throw std::runtime_error("invalid generated audio shape");
    }
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error("generated audio is not finite");
        }
    }
}

} // namespace

int main() {
    if (!std::filesystem::is_regular_file(OMNIVOICE_LM_GGUF) ||
        !std::filesystem::is_regular_file(OMNIVOICE_AUDIO_GGUF)) {
        std::cerr << "OmniVoice model GGUF is unavailable; skipping\n";
        return 77;
    }

    try {
        const PromptFixture fixture = read_prompt_fixture();
        omnivoice::OmniVoiceModelParams params;
        params.device = "CUDA0";
        omnivoice::OmniVoiceModel model;
        omnivoice::Status status = model.load(
            OMNIVOICE_LM_GGUF,
            OMNIVOICE_AUDIO_GGUF,
            params);
        if (!status) {
            std::cerr << "CUDA backend unavailable; skipping\n";
            return 77;
        }

        omnivoice::VoiceClonePrompt prompt;
        prompt.reference_text = "Hello.";
        prompt.language = "en";
        prompt.reference_rms = 0.2f;
        prompt.sample_rate = 24000;
        prompt.reference_codes.codebooks = fixture.codebooks;
        prompt.reference_codes.frames = fixture.reference_frames;
        prompt.reference_codes.data = fixture.reference_codes;

        omnivoice::VoiceCloneConfig options;
        options.num_step = 2;
        options.position_temperature = 0.0f;
        options.class_temperature = 0.0f;
        options.postprocess_output = false;
        options.pad_duration = 0.0f;
        options.fade_duration = 0.0f;

        omnivoice::VoiceCloneRequest clone;
        clone.target_text = "你好，OmniVoice！今天用 GGUF 推理。";
        clone.language = "zh";
        clone.target_frames = fixture.target_frames;
        clone.seed = 1234;
        clone.generation_config = options;
        omnivoice::VoiceCloneOutput cloned;
        status = model.generate_voice_clone(prompt, clone, cloned);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        validate_codes(cloned.target_codes, fixture.target_frames);
        validate_audio(cloned.audio);

        omnivoice::VoiceCloneRequest long_clone = clone;
        long_clone.target_text = "Hello. World.";
        long_clone.target_frames = 800;
        std::vector<std::string> logs;
        long_clone.log = [&](const std::string& line) {
            logs.push_back(line);
        };
        omnivoice::VoiceCloneOutput long_output;
        status = model.generate_voice_clone(
            prompt,
            long_clone,
            long_output);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        if (long_output.chunk_codes.size() < 2 ||
            long_output.chunk_prompt_plans.size() !=
                long_output.chunk_codes.size()) {
            throw std::runtime_error(
                "long text did not produce multiple chunks");
        }
        for (const omnivoice::AudioCodes& codes :
             long_output.chunk_codes) {
            validate_codes(codes, codes.frames);
        }
        validate_audio(long_output.audio);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "extended Voice Clone test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
