#include "config.h"
#include "gguf_model.h"
#include "higgs_tokenizer/encoder.h"
#include "audio_toolkit/audio_resampler.h"

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
    std::vector<float> waveform;
    int32_t codebooks = 0;
    int32_t frames = 0;
    std::vector<int32_t> codes;
    std::vector<omnivoice::HiggsEncoderStage> stages;
};

Fixture read_fixture() {
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
        header[2] <= 0 ||
        header[3] <= 0) {
        throw std::runtime_error(
            "invalid Higgs encoder fixture");
    }

    Fixture fixture;
    fixture.waveform.resize(static_cast<size_t>(header[0]));
    fixture.codebooks = header[1];
    fixture.frames = header[2];
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
    fixture.stages.reserve(static_cast<size_t>(header[3]));
    for (int32_t index = 0; index < header[3]; ++index) {
        char name[32] = {};
        int32_t channels = 0;
        int32_t frames = 0;
        file.read(name, sizeof(name));
        file.read(
            reinterpret_cast<char*>(&channels),
            sizeof(channels));
        file.read(
            reinterpret_cast<char*>(&frames),
            sizeof(frames));
        if (!file || channels <= 0 || frames <= 0) {
            throw std::runtime_error(
                "invalid Higgs encoder stage");
        }
        omnivoice::HiggsEncoderStage stage;
        stage.name = name;
        stage.channels = channels;
        stage.frames = frames;
        stage.values.resize(
            static_cast<size_t>(channels) *
            static_cast<size_t>(frames));
        file.read(
            reinterpret_cast<char*>(stage.values.data()),
            static_cast<std::streamsize>(
                stage.values.size() * sizeof(float)));
        if (!file) {
            throw std::runtime_error(
                "truncated Higgs encoder stage");
        }
        fixture.stages.push_back(std::move(stage));
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

double signal_to_noise_db(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        return -std::numeric_limits<double>::infinity();
    }
    double signal_energy = 0.0;
    double noise_energy = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double difference = static_cast<double>(
            actual[index] - expected[index]);
        signal_energy +=
            static_cast<double>(expected[index]) * expected[index];
        noise_energy += difference * difference;
    }
    return noise_energy > 0.0
        ? 10.0 * std::log10(signal_energy / noise_energy)
        : std::numeric_limits<double>::infinity();
}

} // namespace

int main() {
    const Fixture fixture = read_fixture();

    const std::vector<float> resampled =
        omnivoice::resample_24k_to_16k(fixture.waveform);
    const auto resampled_stage = std::find_if(
        fixture.stages.begin(),
        fixture.stages.end(),
        [](const auto& stage) {
            return stage.name == "resampled";
        });
    if (resampled_stage == fixture.stages.end()) {
        std::cerr << "missing resampler fixture stage\n";
        return 1;
    }
    const float resampler_error = maximum_error(
        resampled,
        resampled_stage->values);
    std::cout << "resampler_max_error: "
              << resampler_error << '\n';
    if (resampler_error > 2e-6f) {
        return 1;
    }

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
        status = omnivoice::load_higgs_tokenizer_config(
            *gguf,
            config);
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
            ggml_backend_init_by_name(device_name, nullptr);
        if (backend == nullptr) {
            std::cerr << "requested backend unavailable; skipping\n";
            return 77;
        }

        omnivoice::HiggsAudioEncoder encoder;
        status = encoder.load(config, *gguf, backend);
        if (!status) {
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }

        omnivoice::AudioBuffer audio;
        audio.samples = fixture.waveform;
        audio.sample_rate = 24000;
        audio.channels = 1;
        omnivoice::AudioCodes codes;
        std::vector<omnivoice::HiggsEncoderStage> stages;
        status = encoder.encode(audio, codes, &stages);
        if (!status) {
            ggml_backend_free(backend);
            throw std::runtime_error(status.message());
        }

        const bool codes_match =
            codes.codebooks == fixture.codebooks &&
            codes.frames == fixture.frames &&
            codes.data == fixture.codes;
        std::cout << "higgs_encoder_codes_match: "
                  << (codes_match ? "true" : "false") << '\n';
        if (!codes_match &&
            codes.codebooks == fixture.codebooks &&
            codes.frames == fixture.frames) {
            int32_t mismatches = 0;
            for (size_t index = 0;
                 index < codes.data.size();
                 ++index) {
                if (codes.data[index] != fixture.codes[index]) {
                    if (mismatches < 8) {
                        std::cout << "code_mismatch[" << index
                                  << "]: actual=" << codes.data[index]
                                  << " expected="
                                  << fixture.codes[index] << '\n';
                    }
                    ++mismatches;
                }
            }
            std::cout << "higgs_encoder_code_mismatches: "
                      << mismatches << '\n';
        }
        if (stages.size() != fixture.stages.size()) {
            std::cerr << "Higgs encoder stage count mismatch: "
                      << stages.size() << " != "
                      << fixture.stages.size() << '\n';
            ggml_backend_free(backend);
            return 1;
        }

        float worst_stage_error = 0.0f;
        double worst_stage_snr_db =
            std::numeric_limits<double>::infinity();
        for (size_t index = 0; index < stages.size(); ++index) {
            const auto& actual = stages[index];
            const auto& expected = fixture.stages[index];
            const float error =
                actual.name == expected.name &&
                actual.channels == expected.channels &&
                actual.frames == expected.frames
                    ? maximum_error(
                        actual.values,
                        expected.values)
                    : std::numeric_limits<float>::infinity();
            std::cout << "higgs_encoder_stage_"
                      << actual.name
                      << "_max_error: " << error << '\n'
                      << "higgs_encoder_stage_"
                      << actual.name
                      << "_snr_db: "
                      << signal_to_noise_db(
                          actual.values,
                          expected.values)
                      << '\n';
            if (!std::isfinite(error)) {
                std::cout << "stage_shape_actual: "
                          << actual.name << ' '
                          << actual.channels << ' '
                          << actual.frames << ' '
                          << actual.values.size() << '\n'
                          << "stage_shape_expected: "
                          << expected.name << ' '
                          << expected.channels << ' '
                          << expected.frames << ' '
                          << expected.values.size() << '\n';
            }
            if ((actual.name == "feature_conv0" ||
                 actual.name == "pos_group0") &&
                actual.values.size() >= 5 &&
                expected.values.size() >= 5) {
                std::cout << actual.name << "_actual:";
                for (size_t value = 0; value < 5; ++value) {
                    std::cout << ' ' << actual.values[value];
                }
                std::cout << '\n' << actual.name << "_expected:";
                for (size_t value = 0; value < 5; ++value) {
                    std::cout << ' ' << expected.values[value];
                }
                std::cout << '\n';
                if (actual.name == "pos_group0" &&
                    actual.frames > 20 &&
                    actual.channels == expected.channels) {
                    for (int shift = -2; shift <= 2; ++shift) {
                        float shifted_error = 0.0f;
                        for (int32_t channel = 0;
                             channel < actual.channels;
                             ++channel) {
                            for (int32_t frame = 2;
                                 frame + shift >= 2 &&
                                 frame + shift < actual.frames - 2;
                                 ++frame) {
                                const size_t actual_index =
                                    static_cast<size_t>(channel) *
                                        actual.frames +
                                    frame;
                                const size_t expected_index =
                                    static_cast<size_t>(channel) *
                                        actual.frames +
                                    frame + shift;
                                shifted_error = std::max(
                                    shifted_error,
                                    std::abs(
                                        actual.values[actual_index] -
                                        expected.values[expected_index]));
                            }
                        }
                        std::cout << "pos_group0_shift_" << shift
                                  << "_error: " << shifted_error
                                  << '\n';
                    }
                }
            }
            if (actual.name != "resampled") {
                worst_stage_error = std::max(
                    worst_stage_error,
                    error);
                worst_stage_snr_db = std::min(
                    worst_stage_snr_db,
                    signal_to_noise_db(
                        actual.values,
                        expected.values));
            }
        }
        std::cout << "higgs_encoder_worst_stage_error: "
                  << worst_stage_error << '\n'
                  << "higgs_encoder_worst_stage_snr_db: "
                  << worst_stage_snr_db << '\n';
        const bool cuda_device =
            std::string(device_name) != "CPU";
        const bool precision_ok = cuda_device
            ? worst_stage_snr_db >= 50.0
            : worst_stage_error <= 5e-3f;

        ggml_backend_free(backend);
        if (!codes_match ||
            !std::isfinite(worst_stage_error) ||
            !precision_ok) {
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Higgs encoder test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
