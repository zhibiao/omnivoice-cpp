#pragma once

#include "status.h"
#include "types.h"

#include <cstdint>
#include <string>

namespace omnivoice {

struct WavInfo {
    uint16_t format_tag = 0;
    uint16_t bits_per_sample = 0;
    int32_t sample_rate = 0;
    int32_t channels = 0;
    int64_t frames = 0;
};

// Decodes WAV and MP3 files.
//
// Samples are returned exactly as stored in the file: native sample rate,
// native channel count, frame-major interleaved float32. Nothing is downmixed
// or resampled, so callers that need a specific rate must go through
// resample_audio() (audio_resampler.h).
//
// WAV accepts PCM16, PCM24, PCM32 and IEEE float32 sample encodings, including
// WAVE_FORMAT_EXTENSIBLE headers. MP3 is decoded with miniaudio.
Status load_audio(
    const std::string& path,
    AudioBuffer& audio);

// Strict reader for 24 kHz mono WAV files. PCM16, PCM24, PCM32 and IEEE
// float32 sample encodings are supported.
Status load_wav_24k_mono(
    const std::string& path,
    AudioBuffer& audio,
    WavInfo* info = nullptr);

// Loads any supported file as 24 kHz mono float32.
//
// Multi-channel input is averaged down to mono, then resampled to the model
// rate. Samples are clamped to [-1, 1] because the band-limited resampler can
// overshoot slightly on full-scale material.
Status load_audio_24k_mono(
    const std::string& path,
    AudioBuffer& audio);

} // namespace omnivoice
