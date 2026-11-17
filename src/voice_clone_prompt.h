#pragma once

#include "higgs_tokenizer/encoder.h"
#include "status.h"
#include "types.h"

#include <cstdint>
#include <string>

namespace omnivoice {

struct VoiceClonePrompt {
    std::string reference_text;
    std::string language = "None";
    float reference_rms = 0.0f;
    int32_t sample_rate = 24000;
    bool denoise = true;
    AudioCodes reference_codes;
};

// Encodes a validated 24 kHz mono waveform and creates a reusable prompt.
Status create_voice_clone_prompt(
    HiggsAudioEncoder& encoder,
    const AudioBuffer& reference_audio,
    const std::string& reference_text,
    const std::string& language,
    bool denoise,
    bool preprocess_reference,
    VoiceClonePrompt& prompt);

Status validate_voice_clone_prompt(
    const VoiceClonePrompt& prompt);

Status save_voice_clone_prompt(
    const std::string& path,
    const VoiceClonePrompt& prompt);

Status load_voice_clone_prompt(
    const std::string& path,
    VoiceClonePrompt& prompt);

} // namespace omnivoice
