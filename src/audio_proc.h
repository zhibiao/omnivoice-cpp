#pragma once

#include "status.h"
#include "types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omnivoice {

struct VoiceCloneConfig;

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

// Strict reader for 24 kHz mono WAV files. PCM16, PCM24, PCM32 and IEEE
// float32 sample encodings are supported.
Status load_wav_24k_mono(
    const std::string& path,
    AudioBuffer& audio,
    WavInfo* info = nullptr);

// Loads reference audio as 24 kHz mono float32.
//
// RIFF/WAVE and MP3 containers are supported. Anything else is rejected.
// WAV files that already satisfy the strict 24 kHz mono contract are read
// by the dedicated parser; other sample rates, channel counts and MP3
// streams are decoded and converted to 24 kHz mono.
Status load_audio_24k_mono(
    const std::string& path,
    AudioBuffer& audio);

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

Status cross_fade_audio_chunks(
    const std::vector<AudioBuffer>& chunks,
    int32_t sample_rate,
    float silence_duration,
    AudioBuffer& output);

Status postprocess_generated_audio(
    const VoiceCloneConfig& config,
    float reference_rms,
    AudioBuffer& audio);

// Matches torchaudio.functional.resample with sinc_interp_hann,
// lowpass_filter_width=6 and rolloff=0.99.
std::vector<float> resample_24k_to_16k(
    const std::vector<float>& input);

} // namespace omnivoice
