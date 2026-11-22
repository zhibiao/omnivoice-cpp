#include "status.h"
#include "types.h"
#include "audio_kit/audio_io.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

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
        "omnivoice_m0_wav_writer_test.wav";

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

    std::ifstream file(output, std::ios::binary);
    std::array<char, 44> header{};
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    file.close();
    std::filesystem::remove(output);

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
    return 0;
}
