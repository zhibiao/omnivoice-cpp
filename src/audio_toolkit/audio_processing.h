#pragma once

#include "status.h"
#include "types.h"

#include <cstdint>
#include <vector>

namespace omnivoice {

struct VoiceCloneConfig;

struct SilenceTrimOptions {
    float threshold_dbfs = -50.0f;
    int32_t min_silence_ms = 200;
    int32_t keep_silence_ms = 100;
};

Status validate_audio_24k_mono(const AudioBuffer& audio);

float compute_rms(const AudioBuffer& audio);

// Averages all channels of a frame-major interleaved buffer into mono. The
// sample rate is preserved.
Status downmix_to_mono(
    const AudioBuffer& input,
    AudioBuffer& output);

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

} // namespace omnivoice
