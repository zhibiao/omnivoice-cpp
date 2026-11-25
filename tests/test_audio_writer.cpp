#include "status.h"
#include "types.h"
#include "audio_proc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

uint32_t read_u32(const std::array<char, 44>& header, size_t offset) {
    return static_cast<uint32_t>(static_cast<uint8_t>(header[offset])) |
        (static_cast<uint32_t>(static_cast<uint8_t>(header[offset + 1])) << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(header[offset + 2])) << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(header[offset + 3])) << 24);
}

} // namespace

int main() {
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() /
        "omnivoice_audio_writer_test.wav";

    omnivoice::AudioBuffer audio;
    audio.sample_rate = 24000;
    audio.channels = 1;
    audio.samples = {0.0f, 0.25f, -0.25f, 1.0f};

    const omnivoice::Status status =
        omnivoice::write_wav(output.string(), audio);
    if (!status) {
        std::cerr << "write_wav failed: " << status.message() << '\n';
        return 1;
    }

    std::vector<char> file_bytes;
    {
        std::ifstream file(output, std::ios::binary);
        file_bytes.assign(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>());
    }
    std::filesystem::remove(output);

    std::array<char, 44> header{};
    if (file_bytes.size() < header.size()) {
        std::cerr << "WAV output is too short\n";
        return 1;
    }
    std::copy_n(file_bytes.begin(), header.size(), header.begin());

    const std::string riff(header.data(), 4);
    const std::string wave(header.data() + 8, 4);
    const std::string data(header.data() + 36, 4);
    if (riff != "RIFF" || wave != "WAVE" || data != "data") {
        std::cerr << "invalid WAV header\n";
        return 1;
    }
    if (read_u32(header, 24) != 24000) {
        std::cerr << "invalid WAV sample rate\n";
        return 1;
    }
    if (read_u32(header, 40) != audio.samples.size() * 2) {
        std::cerr << "invalid WAV data size\n";
        return 1;
    }

    std::vector<uint8_t> wav_bytes;
    omnivoice::Status encode_status =
        omnivoice::encode_wav_24k_mono(audio, wav_bytes);
    if (!encode_status) {
        std::cerr << "encode_wav_24k_mono failed: "
                  << encode_status.message() << '\n';
        return 1;
    }
    if (wav_bytes.size() != file_bytes.size() ||
        !std::equal(
            wav_bytes.begin(),
            wav_bytes.end(),
            file_bytes.begin(),
            [](uint8_t left, char right) {
                return left == static_cast<uint8_t>(right);
            })) {
        std::cerr << "in-memory WAV differs from write_wav output\n";
        return 1;
    }

    std::vector<uint8_t> pcm_bytes;
    encode_status = omnivoice::encode_pcm_s16le(audio, pcm_bytes);
    if (!encode_status) {
        std::cerr << "encode_pcm_s16le failed: "
                  << encode_status.message() << '\n';
        return 1;
    }
    if (pcm_bytes.size() != audio.samples.size() * 2) {
        std::cerr << "invalid PCM byte count\n";
        return 1;
    }
    if (!std::equal(
            pcm_bytes.begin(),
            pcm_bytes.end(),
            wav_bytes.begin() + 44)) {
        std::cerr << "PCM payload differs from WAV payload\n";
        return 1;
    }

    omnivoice::AudioBuffer empty;
    empty.samples.clear();
    std::vector<uint8_t> rejected;
    if (omnivoice::encode_wav_24k_mono(empty, rejected) ||
        !rejected.empty()) {
        std::cerr << "encode_wav_24k_mono accepted an empty buffer\n";
        return 1;
    }
    if (omnivoice::encode_pcm_s16le(empty, rejected) ||
        !rejected.empty()) {
        std::cerr << "encode_pcm_s16le accepted an empty buffer\n";
        return 1;
    }
    return 0;
}
