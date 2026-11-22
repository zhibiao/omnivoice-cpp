#include "omnivoice.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct EncoderFixture {
    std::vector<float> waveform;
    int32_t codebooks = 0;
    int32_t frames = 0;
    std::vector<int32_t> codes;
};

EncoderFixture read_encoder_fixture() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/higgs_encoder_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            "failed to open Higgs encoder fixture");
    }

    char magic[8] = {};
    std::array<int32_t, 4> header{};
    file.read(magic, sizeof(magic));
    file.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(
            header.size() * sizeof(int32_t)));
    if (!file ||
        std::memcmp(magic, "HIGENC01", 8) != 0 ||
        header[0] <= 0 ||
        header[1] <= 0 ||
        header[2] <= 0) {
        throw std::runtime_error(
            "invalid Higgs encoder fixture");
    }

    EncoderFixture fixture;
    fixture.codebooks = header[1];
    fixture.frames = header[2];
    fixture.waveform.resize(static_cast<size_t>(header[0]));
    fixture.codes.resize(
        static_cast<size_t>(fixture.codebooks) *
        static_cast<size_t>(fixture.frames));
    file.read(
        reinterpret_cast<char*>(fixture.waveform.data()),
        static_cast<std::streamsize>(
            fixture.waveform.size() * sizeof(float)));
    file.read(
        reinterpret_cast<char*>(fixture.codes.data()),
        static_cast<std::streamsize>(
            fixture.codes.size() * sizeof(int32_t)));
    if (!file) {
        throw std::runtime_error(
            "truncated Higgs encoder fixture");
    }
    return fixture;
}

int32_t read_target_frames() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/voice_clone_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            "failed to open Voice Clone fixture");
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
        header[2] <= 0) {
        throw std::runtime_error(
            "invalid Voice Clone fixture");
    }
    return header[2];
}

bool same_prompt(
    const omnivoice::VoiceClonePrompt& lhs,
    const omnivoice::VoiceClonePrompt& rhs) {
    return lhs.reference_text == rhs.reference_text &&
        lhs.language == rhs.language &&
        lhs.reference_rms == rhs.reference_rms &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.denoise == rhs.denoise &&
        lhs.reference_codes.codebooks ==
            rhs.reference_codes.codebooks &&
        lhs.reference_codes.frames ==
            rhs.reference_codes.frames &&
        lhs.reference_codes.data ==
            rhs.reference_codes.data;
}

void validate_codes(
    const omnivoice::AudioCodes& codes,
    int32_t expected_frames) {
    if (codes.codebooks != 8 ||
        codes.frames != expected_frames ||
        codes.data.size() != codes.size()) {
        throw std::runtime_error(
            "invalid generated code shape");
    }
    for (int32_t code : codes.data) {
        if (code < 0 || code >= 1024) {
            throw std::runtime_error(
                "generated code is out of range");
        }
    }
}

void validate_audio(const omnivoice::AudioBuffer& audio) {
    if (audio.sample_rate != 24000 ||
        audio.channels != 1 ||
        audio.samples.empty()) {
        throw std::runtime_error(
            "invalid generated audio shape");
    }
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error(
                "generated audio is not finite");
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

    const std::filesystem::path prompt_path =
        std::filesystem::path(OMNIVOICE_TEST_DATA_DIR) /
        "test_voice_clone_e2e.ovprompt";
    try {
        const EncoderFixture fixture = read_encoder_fixture();
        const char* device = std::getenv("OMNIVOICE_TEST_DEVICE");
        omnivoice::GGMLRuntimeParams params;
        params.device = device == nullptr ? "CPU" : device;

        omnivoice::GGMLRuntime runtime;
        omnivoice::Status status = runtime.load(
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

        omnivoice::OmniVoiceModel model;
        status = model.load(runtime);
        if (!status) {
            throw std::runtime_error(status.message());
        }

        omnivoice::AudioBuffer reference_audio;
        reference_audio.samples = fixture.waveform;
        reference_audio.sample_rate = 24000;
        reference_audio.channels = 1;

        omnivoice::VoiceClonePromptRequest prompt_request;
        prompt_request.reference_audio = reference_audio;
        prompt_request.reference_text = "Hello.";
        prompt_request.language = "en";
        prompt_request.denoise = false;
        prompt_request.preprocess_reference = false;
        omnivoice::VoiceClonePrompt encoded_prompt;
        status = model.create_prompt(prompt_request, encoded_prompt);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        if (encoded_prompt.reference_codes.codebooks !=
                fixture.codebooks ||
            encoded_prompt.reference_codes.frames != fixture.frames ||
            encoded_prompt.reference_codes.data.size() !=
                encoded_prompt.reference_codes.size()) {
            throw std::runtime_error(
                "reference prompt code shape is invalid");
        }
        for (int32_t code : encoded_prompt.reference_codes.data) {
            if (code < 0 || code >= 1024) {
                throw std::runtime_error(
                    "reference prompt code is out of range");
            }
        }

        std::filesystem::remove(prompt_path);
        status = omnivoice::save_voice_clone_prompt(
            prompt_path.string(),
            encoded_prompt);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        omnivoice::VoiceClonePrompt loaded_prompt;
        status = omnivoice::load_voice_clone_prompt(
            prompt_path.string(),
            loaded_prompt);
        if (!status || !same_prompt(loaded_prompt, encoded_prompt)) {
            throw std::runtime_error(
                "saved prompt did not round-trip");
        }

        const int32_t target_frames = read_target_frames();
        omnivoice::VoiceCloneConfig options;
        options.num_step = 2;
        options.position_temperature = 0.0f;
        options.class_temperature = 0.0f;
        options.postprocess_output = false;
        options.pad_duration = 0.0f;
        options.fade_duration = 0.0f;

        omnivoice::VoiceCloneRequest request;
        request.target_text = "Hi.";
        request.language = "en";
        request.target_frames = target_frames;
        request.seed = 1234;
        request.generation_config = options;

        omnivoice::VoiceCloneOutput output;
        status = model.generate_voice_clone(loaded_prompt, request, output);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        if (!same_prompt(output.prompt, loaded_prompt)) {
            throw std::runtime_error(
                "generation did not reuse the saved prompt");
        }
        validate_codes(output.target_codes, target_frames);
        validate_audio(output.audio);

        std::filesystem::remove(prompt_path);
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(prompt_path);
        std::cerr << "Voice Clone E2E test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
