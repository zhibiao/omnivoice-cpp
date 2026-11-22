#pragma once

#include "status.h"
#include "types.h"

#include <vector>

namespace omnivoice {

struct VoiceCloneConfig;

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
