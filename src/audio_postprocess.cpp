#include "audio_postprocess.h"

#include "audio_io.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace omnivoice {

Status cross_fade_audio_chunks(
    const std::vector<AudioBuffer>& chunks,
    int32_t sample_rate,
    float silence_duration,
    AudioBuffer& output) {
    output = {};
    if (chunks.empty() || sample_rate != 24000) {
        return Status::error("invalid audio chunk list");
    }
    for (const AudioBuffer& chunk : chunks) {
        if (chunk.sample_rate != sample_rate ||
            chunk.channels != 1 ||
            chunk.samples.empty()) {
            return Status::error("audio chunks must be non-empty 24 kHz mono");
        }
    }
    if (chunks.size() == 1) {
        output = chunks[0];
        return Status::ok();
    }

    const int32_t total_fade_samples = std::max(
        0,
        static_cast<int32_t>(silence_duration * sample_rate));
    const int32_t fade_samples = total_fade_samples / 3;
    const int32_t silence_samples = fade_samples;
    output = chunks[0];
    for (size_t chunk_index = 1;
         chunk_index < chunks.size();
         ++chunk_index) {
        const int32_t fade_out = std::min(
            fade_samples,
            static_cast<int32_t>(output.samples.size()));
        for (int32_t index = 0; index < fade_out; ++index) {
            const float weight =
                1.0f - static_cast<float>(index) /
                    static_cast<float>(std::max(1, fade_out - 1));
            const size_t destination =
                output.samples.size() -
                static_cast<size_t>(fade_out) +
                static_cast<size_t>(index);
            output.samples[destination] *= weight;
        }

        AudioBuffer faded_in = chunks[chunk_index];
        const int32_t fade_in = std::min(
            fade_samples,
            static_cast<int32_t>(faded_in.samples.size()));
        for (int32_t index = 0; index < fade_in; ++index) {
            const float weight =
                static_cast<float>(index) /
                static_cast<float>(std::max(1, fade_in - 1));
            faded_in.samples[static_cast<size_t>(index)] *= weight;
        }

        output.samples.resize(
            output.samples.size() +
            static_cast<size_t>(silence_samples) +
            faded_in.samples.size(),
            0.0f);
        const size_t destination =
            output.samples.size() -
            faded_in.samples.size();
        std::copy(
            faded_in.samples.begin(),
            faded_in.samples.end(),
            output.samples.begin() +
                static_cast<std::ptrdiff_t>(destination));
    }
    return Status::ok();
}

Status postprocess_generated_audio(
    const VoiceCloneConfig& config,
    float reference_rms,
    AudioBuffer& audio) {
    if (config.postprocess_output) {
        for (float& sample : audio.samples) {
            sample = std::clamp(sample, -1.0f, 1.0f);
        }
        SilenceTrimOptions trim;
        trim.threshold_dbfs = -50.0f;
        trim.min_silence_ms = 500;
        trim.keep_silence_ms = 100;
        Status status = remove_silence(audio, trim);
        if (!status) {
            return status;
        }
    }
    if (reference_rms < 0.1f) {
        const float gain = reference_rms / 0.1f;
        for (float& sample : audio.samples) {
            sample *= gain;
        }
    }

    const int32_t fade_samples = std::max(
        0,
        static_cast<int32_t>(
            config.fade_duration * audio.sample_rate));
    const int32_t fade = std::min(
        fade_samples,
        static_cast<int32_t>(audio.samples.size() / 2));
    for (int32_t index = 0; index < fade; ++index) {
        const float position =
            static_cast<float>(index) /
            static_cast<float>(std::max(1, fade - 1));
        audio.samples[static_cast<size_t>(index)] *= position;
        audio.samples[
            audio.samples.size() - 1 -
            static_cast<size_t>(index)] *= position;
    }

    const int32_t padding = std::max(
        0,
        static_cast<int32_t>(
            config.pad_duration * audio.sample_rate));
    if (padding > 0) {
        std::vector<float> padded(
            audio.samples.size() +
                2 * static_cast<size_t>(padding),
            0.0f);
        std::copy(
            audio.samples.begin(),
            audio.samples.end(),
            padded.begin() + padding);
        audio.samples = std::move(padded);
    }
    return Status::ok();
}

} // namespace omnivoice
