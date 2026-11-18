#include "config.h"
#include "gguf_model.h"
#include "higgs_tokenizer/decoder.h"

#include "ggml-backend.h"

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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Fixture {
    int32_t codebooks = 0;
    int32_t frames = 0;
    std::vector<int32_t> codes;
    std::vector<float> waveform;
};

struct ExpectedStage {
    std::string name;
    int32_t channels = 0;
    int32_t frames = 0;
    std::vector<float> values;
};

Fixture read_fixture() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/higgs_decoder_fixture.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open Higgs decoder fixture");
    }
    char magic[8] = {};
    std::array<int32_t, 4> header{};
    file.read(magic, sizeof(magic));
    file.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(header.size() * sizeof(int32_t)));
    if (!file ||
        std::memcmp(magic, "HIGDEC01", 8) != 0 ||
        header[3] != 1) {
        throw std::runtime_error("invalid Higgs decoder fixture");
    }

    Fixture fixture;
    fixture.codebooks = header[0];
    fixture.frames = header[1];
    fixture.codes.resize(
        static_cast<size_t>(fixture.codebooks) * fixture.frames);
    file.read(
        reinterpret_cast<char*>(fixture.codes.data()),
        static_cast<std::streamsize>(
            fixture.codes.size() * sizeof(int32_t)));
    fixture.waveform.resize(header[2]);
    file.read(
        reinterpret_cast<char*>(fixture.waveform.data()),
        static_cast<std::streamsize>(
            fixture.waveform.size() * sizeof(float)));
    if (!file) {
        throw std::runtime_error("truncated Higgs decoder fixture");
    }
    return fixture;
}

std::vector<ExpectedStage> read_stages() {
    std::ifstream file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/higgs_decoder_stages.bin",
        std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open Higgs stage fixture");
    }
    char magic[8] = {};
    uint32_t count = 0;
    file.read(magic, sizeof(magic));
    file.read(
        reinterpret_cast<char*>(&count),
        sizeof(count));
    if (!file || std::memcmp(magic, "HIGSTG01", 8) != 0) {
        throw std::runtime_error("invalid Higgs stage fixture");
    }
    std::vector<ExpectedStage> stages;
    stages.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        char name[16] = {};
        int32_t channels = 0;
        int32_t frames = 0;
        file.read(name, sizeof(name));
        file.read(reinterpret_cast<char*>(&channels), sizeof(channels));
        file.read(reinterpret_cast<char*>(&frames), sizeof(frames));
        if (!file || channels <= 0 || frames <= 0) {
            throw std::runtime_error("invalid Higgs stage header");
        }
        ExpectedStage stage;
        stage.name = name;
        stage.channels = channels;
        stage.frames = frames;
        stage.values.resize(
            static_cast<size_t>(channels) * frames);
        file.read(
            reinterpret_cast<char*>(stage.values.data()),
            static_cast<std::streamsize>(
                stage.values.size() * sizeof(float)));
        if (!file) {
            throw std::runtime_error("truncated Higgs stage fixture");
        }
        stages.push_back(std::move(stage));
    }
    return stages;
}

} // namespace

int main() {
    if (!std::filesystem::is_regular_file(OMNIVOICE_AUDIO_GGUF)) {
        std::cerr << "Higgs tokenizer GGUF is unavailable; skipping\n";
        return 77;
    }

    try {
        std::unique_ptr<omnivoice::GGUFModel> gguf;
        omnivoice::Status status = omnivoice::GGUFModel::open(
            OMNIVOICE_AUDIO_GGUF,
            gguf);
        if (!status) {
            throw std::runtime_error(status.message());
        }
        omnivoice::HiggsTokenizerConfig config;
        status = omnivoice::load_higgs_tokenizer_config(*gguf, config);
        if (!status) {
            throw std::runtime_error(status.message());
        }

        ggml_backend_load_all();
        const char* device_name =
            std::getenv("OMNIVOICE_TEST_DEVICE");
        if (device_name == nullptr || device_name[0] == '\0') {
            device_name = "CPU";
        }
        ggml_backend_t backend =
            ggml_backend_init_by_name(
                device_name,
                nullptr);
        if (backend == nullptr) {
            std::cerr << "requested backend unavailable; skipping\n";
            return 77;
        }

        auto decoder = std::make_unique<omnivoice::HiggsAudioDecoder>();
        status = decoder->load(config, *gguf, backend);
        if (!status) {
            decoder.reset();
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }

        const Fixture fixture = read_fixture();
        const std::vector<ExpectedStage> expected_stages =
            read_stages();
        omnivoice::AudioCodes codes;
        codes.codebooks = fixture.codebooks;
        codes.frames = fixture.frames;
        codes.data = fixture.codes;
        omnivoice::AudioBuffer audio;
        std::vector<omnivoice::HiggsDecoderStage> debug_stages;
        status = decoder->decode(codes, audio, &debug_stages);
        if (!status) {
            decoder.reset();
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }

        if (audio.sample_rate != 24000 ||
            audio.channels != 1 ||
            audio.samples.size() != fixture.waveform.size()) {
            std::cerr << "Higgs decoder output shape mismatch\n";
            decoder.reset();
            ggml_backend_free(backend);
            return 1;
        }
        if (debug_stages.size() != expected_stages.size()) {
            std::cerr << "Higgs decoder stage count mismatch\n";
            decoder.reset();
            ggml_backend_free(backend);
            return 1;
        }
        for (size_t stage_index = 0;
             stage_index < debug_stages.size();
             ++stage_index) {
            const auto& actual_stage = debug_stages[stage_index];
            const auto& expected_stage = expected_stages[stage_index];
            float stage_error = 0.0f;
            if (actual_stage.name == expected_stage.name &&
                actual_stage.channels == expected_stage.channels &&
                actual_stage.frames == expected_stage.frames &&
                actual_stage.values.size() ==
                    expected_stage.values.size()) {
                for (size_t index = 0;
                     index < actual_stage.values.size();
                     ++index) {
                    stage_error = std::max(
                        stage_error,
                        std::abs(
                            actual_stage.values[index] -
                            expected_stage.values[index]));
                }
            } else {
                stage_error = std::numeric_limits<float>::infinity();
            }
            std::cout << "higgs_stage_" << actual_stage.name
                      << "_max_error: " << stage_error << '\n';
            if (actual_stage.name == "quantized" &&
                !actual_stage.values.empty() &&
                !expected_stage.values.empty()) {
                std::cout << "quantized_actual0: "
                          << actual_stage.values[0]
                          << " expected0: "
                          << expected_stage.values[0]
                          << " actual1: " << actual_stage.values[1]
                          << " expected1: "
                          << expected_stage.values[1] << '\n';
                std::cout << "quantized_actual:";
                for (size_t index = 0;
                     index < 5 && index < actual_stage.values.size();
                     ++index) {
                    std::cout << ' ' << actual_stage.values[index];
                }
                std::cout << "\nquantized_expected:";
                for (size_t index = 0;
                     index < 5 && index < expected_stage.values.size();
                     ++index) {
                    std::cout << ' ' << expected_stage.values[index];
                }
                std::cout << '\n';
            }
            if (actual_stage.name == "conv1" &&
                !actual_stage.values.empty() &&
                !expected_stage.values.empty()) {
                std::cout << "conv1_actual:";
                for (size_t index = 0;
                     index < 5 && index < actual_stage.values.size();
                     ++index) {
                    std::cout << ' ' << actual_stage.values[index];
                }
                std::cout << "\nconv1_expected:";
                for (size_t index = 0;
                     index < 5 && index < expected_stage.values.size();
                     ++index) {
                    std::cout << ' ' << expected_stage.values[index];
                }
                std::cout << '\n';
            }
            if (actual_stage.name == "block0" &&
                !actual_stage.values.empty() &&
                !expected_stage.values.empty()) {
                std::cout << "block0_actual:";
                for (size_t index = 0;
                     index < 5 && index < actual_stage.values.size();
                     ++index) {
                    std::cout << ' ' << actual_stage.values[index];
                }
                std::cout << "\nblock0_expected:";
                for (size_t index = 0;
                     index < 5 && index < expected_stage.values.size();
                     ++index) {
                    std::cout << ' ' << expected_stage.values[index];
                }
                std::cout << '\n';
            }
        }

        omnivoice::AudioBuffer repeated_audio;
        std::vector<omnivoice::HiggsDecoderStage>
            repeated_stages;
        status = decoder->decode(
            codes,
            repeated_audio,
            &repeated_stages);
        if (!status ||
            repeated_audio.samples.size() !=
                audio.samples.size() ||
            repeated_stages.size() !=
                debug_stages.size()) {
            std::cerr << "Higgs decoder graph reuse mismatch\n";
            decoder.reset();
            ggml_backend_free(backend);
            return 1;
        }
        for (size_t index = 0;
             index < audio.samples.size();
             ++index) {
            if (repeated_audio.samples[index] !=
                audio.samples[index]) {
                std::cerr << "Higgs decoder graph reuse changed output\n";
                decoder.reset();
                ggml_backend_free(backend);
                return 1;
            }
        }

        float maximum_error = 0.0f;
        double signal_energy = 0.0;
        double noise_energy = 0.0;
        for (size_t index = 0;
             index < audio.samples.size();
             ++index) {
            const double difference = static_cast<double>(
                audio.samples[index] - fixture.waveform[index]);
            maximum_error = std::max(
                maximum_error,
                static_cast<float>(std::abs(difference)));
            signal_energy +=
                static_cast<double>(fixture.waveform[index]) *
                fixture.waveform[index];
            noise_energy += difference * difference;
        }
        const double snr_db = noise_energy > 0.0
            ? 10.0 * std::log10(
                signal_energy / noise_energy)
            : std::numeric_limits<double>::infinity();
        std::cout << "higgs_decoder_max_error: "
                  << maximum_error << '\n'
                  << "higgs_decoder_snr_db: " << snr_db << '\n';
        if (!std::isfinite(maximum_error) ||
            maximum_error > 3e-3f ||
            snr_db < 40.0) {
            decoder.reset();
            ggml_backend_free(backend);
            return 1;
        }

        decoder.reset();
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Higgs decoder test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
