#pragma once

#include "status.h"
#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omnivoice {

constexpr int32_t OMNIVOICE_SAMPLE_RATE = 24000;
constexpr int32_t OMNIVOICE_HOP_LENGTH = 960;

struct WavInfo {
    uint16_t format_tag = 0;
    uint16_t bits_per_sample = 0;
    int32_t sample_rate = 0;
    int32_t channels = 0;
    int64_t frames = 0;
};

struct SilenceTrimOptions {
    float threshold_dbfs = -50.0f;
    int32_t min_silence_ms = 200;
    int32_t keep_silence_ms = 100;
};

// M2 only accepts 24 kHz, mono WAV files. PCM16, PCM24, PCM32 and IEEE
// float32 sample encodings are supported.
Status load_wav_24k_mono(
    const std::string& path,
    AudioBuffer& audio,
    WavInfo* info = nullptr);

Status save_wav_24k_mono(
    const std::string& path,
    const AudioBuffer& audio);

// Writes 16-bit PCM; sample rate and channel count come from the buffer.
Status write_wav(
    const std::string& path,
    const AudioBuffer& audio);

Status validate_audio_24k_mono(const AudioBuffer& audio);

float compute_rms(const AudioBuffer& audio);

// Drops the incomplete tail so the sample count is a multiple of hop_length.
Status trim_to_hop_multiple(
    AudioBuffer& audio,
    int32_t hop_length = OMNIVOICE_HOP_LENGTH);

// Pads the tail with silence so the sample count is a multiple of hop_length.
Status pad_to_hop_multiple(
    AudioBuffer& audio,
    int32_t hop_length = OMNIVOICE_HOP_LENGTH);

Status remove_silence(
    AudioBuffer& audio,
    const SilenceTrimOptions& options = {});

} // namespace omnivoice
