#pragma once

#include "status.h"
#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omnivoice {

// Writes a validated 24 kHz mono buffer as 16-bit PCM WAV.
Status save_wav_24k_mono(
    const std::string& path,
    const AudioBuffer& audio);

// Writes 16-bit PCM; sample rate and channel count come from the buffer.
Status write_wav(
    const std::string& path,
    const AudioBuffer& audio);

// Encodes a validated 24 kHz mono buffer as an in-memory 16-bit PCM WAV file.
Status encode_wav_24k_mono(
    const AudioBuffer& audio,
    std::vector<uint8_t>& bytes);

// Encodes a 24 kHz mono buffer as raw little-endian signed 16-bit PCM.
// Samples outside [-1, 1] are clamped.
Status encode_pcm_s16le(
    const AudioBuffer& audio,
    std::vector<uint8_t>& bytes);

} // namespace omnivoice
