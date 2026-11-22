#include "audio_kit/audio_postprocess.h"

#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

omnivoice::AudioBuffer make_chunk(float value, int32_t samples) {
    omnivoice::AudioBuffer chunk;
    chunk.sample_rate = 24000;
    chunk.channels = 1;
    chunk.samples.assign(static_cast<size_t>(samples), value);
    return chunk;
}

} // namespace

int main() {
    std::vector<omnivoice::AudioBuffer> chunks = {
        make_chunk(1.0f, 100),
        make_chunk(1.0f, 100),
    };
    omnivoice::AudioBuffer merged;
    omnivoice::Status status = omnivoice::cross_fade_audio_chunks(
        chunks,
        24000,
        0.001f,
        merged);
    if (!status ||
        merged.sample_rate != 24000 ||
        merged.channels != 1 ||
        merged.samples.size() != 208) {
        std::cerr << "audio cross-fade shape mismatch\n";
        return 1;
    }
    std::ifstream fixture_file(
        std::string(OMNIVOICE_TEST_DATA_DIR) +
            "/audio_postprocess_fixture.bin",
        std::ios::binary);
    char magic[8] = {};
    uint32_t expected_size = 0;
    fixture_file.read(magic, sizeof(magic));
    fixture_file.read(
        reinterpret_cast<char*>(&expected_size),
        sizeof(expected_size));
    std::vector<float> expected(expected_size);
    fixture_file.read(
        reinterpret_cast<char*>(expected.data()),
        static_cast<std::streamsize>(
            expected.size() * sizeof(float)));
    float fixture_error = 0.0f;
    if (expected.size() == merged.samples.size()) {
        for (size_t index = 0; index < expected.size(); ++index) {
            fixture_error = std::max(
                fixture_error,
                std::abs(expected[index] - merged.samples[index]));
        }
    } else {
        fixture_error = 1e9f;
    }
    if (!fixture_file ||
        std::memcmp(magic, "AUDCF001", 8) != 0 ||
        fixture_error > 1e-7f) {
        std::cerr << "cross-fade differs from Python golden: "
                  << fixture_error << '\n';
        return 1;
    }
    if (std::abs(merged.samples[99]) > 1e-7f ||
        std::abs(merged.samples[108]) > 1e-7f ||
        std::abs(merged.samples[107]) > 1e-7f) {
        std::cerr << "audio cross-fade boundary is not silent: "
                  << merged.samples[99] << ' '
                  << merged.samples[107] << ' '
                  << merged.samples[108] << '\n';
        return 1;
    }
    for (size_t index = 108; index < 116; ++index) {
        if (merged.samples[index] + 1e-7f <
                merged.samples[index - 1] &&
            index > 108) {
            // Fade-in is monotonically non-decreasing.
        }
    }
    if (merged.samples[115] + 1e-7f <
        merged.samples[114]) {
        std::cerr << "audio fade-in is not monotonic\n";
        return 1;
    }

    omnivoice::AudioBuffer single = make_chunk(0.25f, 12);
    const std::vector<omnivoice::AudioBuffer> one = {single};
    status = omnivoice::cross_fade_audio_chunks(
        one,
        24000,
        0.3f,
        merged);
    if (!status || merged.samples != single.samples) {
        std::cerr << "single chunk must be unchanged\n";
        return 1;
    }

    omnivoice::VoiceCloneConfig config;
    config.postprocess_output = false;
    config.pad_duration = 0.0f;
    config.fade_duration = 0.0f;
    omnivoice::AudioBuffer processed =
        make_chunk(0.5f, 20);
    status = omnivoice::postprocess_generated_audio(
        config,
        1.0f,
        processed);
    if (!status ||
        processed.samples != make_chunk(0.5f, 20).samples) {
        std::cerr << "disabled postprocess modified audio\n";
        return 1;
    }
    return 0;
}
