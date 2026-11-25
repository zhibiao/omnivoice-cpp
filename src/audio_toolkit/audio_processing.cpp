#include "audio_toolkit/audio_processing.h"

#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

float window_db(
    const std::vector<float>& samples,
    size_t begin,
    size_t end) {
    double energy = 0.0;
    for (size_t index = begin; index < end; ++index) {
        energy += static_cast<double>(samples[index]) * samples[index];
    }
    const double rms = std::sqrt(energy / static_cast<double>(end - begin));
    return static_cast<float>(20.0 * std::log10(rms + 1e-12));
}

} // namespace

Status validate_audio_24k_mono(const AudioBuffer& audio) {
    if (audio.samples.empty()) {
        return Status::error("audio buffer is empty");
    }
    if (audio.sample_rate != OMNIVOICE_SAMPLE_RATE) {
        return Status::error("audio buffer must be 24000 Hz");
    }
    if (audio.channels != 1) {
        return Status::error("audio buffer must be mono");
    }
    for (float sample : audio.samples) {
        if (!std::isfinite(sample)) {
            return Status::error("audio buffer contains non-finite samples");
        }
        if (sample < -1.0f || sample > 1.0f) {
            return Status::error("audio buffer samples must be in [-1, 1]");
        }
    }
    return Status::ok();
}

float compute_rms(const AudioBuffer& audio) {
    if (audio.samples.empty()) {
        return 0.0f;
    }
    double energy = 0.0;
    for (float sample : audio.samples) {
        energy += static_cast<double>(sample) * sample;
    }
    return static_cast<float>(
        std::sqrt(energy / static_cast<double>(audio.samples.size())));
}

Status downmix_to_mono(
    const AudioBuffer& input,
    AudioBuffer& output) {
    output = {};
    if (input.samples.empty()) {
        return Status::error("audio buffer is empty");
    }
    if (input.channels <= 0) {
        return Status::error("audio channels must be positive");
    }
    if (input.sample_rate <= 0) {
        return Status::error("audio sample rate must be positive");
    }
    const size_t channels = static_cast<size_t>(input.channels);
    if (input.samples.size() % channels != 0) {
        return Status::error("audio samples are not frame aligned");
    }

    output.sample_rate = input.sample_rate;
    output.channels = 1;
    if (input.channels == 1) {
        output.samples = input.samples;
        return Status::ok();
    }

    const size_t frames = input.samples.size() / channels;
    const float scale = 1.0f / static_cast<float>(input.channels);
    output.samples.resize(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        float sum = 0.0f;
        for (size_t channel = 0; channel < channels; ++channel) {
            sum += input.samples[frame * channels + channel];
        }
        output.samples[frame] = sum * scale;
    }
    return Status::ok();
}

Status trim_to_hop_multiple(
    AudioBuffer& audio,
    int32_t hop_length) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    if (hop_length <= 0) {
        return Status::error("hop length must be positive");
    }
    const size_t remainder = audio.samples.size() %
        static_cast<size_t>(hop_length);
    if (remainder != 0) {
        audio.samples.resize(audio.samples.size() - remainder);
    }
    if (audio.samples.empty()) {
        return Status::error("audio has no complete hop frame");
    }
    return Status::ok();
}

Status pad_to_hop_multiple(
    AudioBuffer& audio,
    int32_t hop_length) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    if (hop_length <= 0) {
        return Status::error("hop length must be positive");
    }
    const size_t remainder = audio.samples.size() %
        static_cast<size_t>(hop_length);
    if (remainder != 0) {
        audio.samples.resize(
            audio.samples.size() +
            static_cast<size_t>(hop_length) - remainder,
            0.0f);
    }
    return Status::ok();
}

Status remove_silence(
    AudioBuffer& audio,
    const SilenceTrimOptions& options) {
    Status status = validate_audio_24k_mono(audio);
    if (!status) {
        return status;
    }
    if (options.threshold_dbfs >= 0.0f) {
        return Status::error("silence threshold must be below 0 dBFS");
    }
    if (options.min_silence_ms < 10 ||
        options.keep_silence_ms < 0) {
        return Status::error("invalid silence trim duration");
    }

    const size_t window = static_cast<size_t>(
        audio.sample_rate * 10 / 1000);
    const size_t window_count =
        (audio.samples.size() + window - 1) / window;

    size_t first_signal_window = window_count;
    size_t last_signal_window = 0;
    std::vector<uint8_t> silent(window_count, 0);
    for (size_t index = 0; index < window_count; ++index) {
        const size_t begin = index * window;
        const size_t end = std::min(begin + window, audio.samples.size());
        const float db = window_db(audio.samples, begin, end);
        silent[index] = db <= options.threshold_dbfs ? 1 : 0;
        if (silent[index] == 0) {
            if (first_signal_window == window_count) {
                first_signal_window = index;
            }
            last_signal_window = index;
        }
    }
    if (first_signal_window == window_count) {
        return Status::error("audio contains no non-silent samples");
    }

    const size_t keep = static_cast<size_t>(
        audio.sample_rate) *
        static_cast<size_t>(options.keep_silence_ms) / 1000;
    const size_t signal_begin = first_signal_window * window;
    const size_t signal_end = std::min(
        (last_signal_window + 1) * window,
        audio.samples.size());
    const size_t begin = signal_begin > keep ? signal_begin - keep : 0;
    const size_t end = std::min(
        audio.samples.size(),
        signal_end + keep);

    std::vector<float> trimmed(
        audio.samples.begin() + static_cast<std::ptrdiff_t>(begin),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
    if (trimmed.empty()) {
        return Status::error("silence trimming produced empty audio");
    }

    const size_t min_silence_samples = static_cast<size_t>(
        audio.sample_rate) *
        static_cast<size_t>(options.min_silence_ms) / 1000;
    if (min_silence_samples > 0) {
        std::vector<float> compacted;
        compacted.reserve(trimmed.size());
        size_t position = 0;
        while (position < trimmed.size()) {
            const size_t run_begin = position;
            const bool is_silent =
                window_db(
                    trimmed,
                    position,
                    std::min(position + window, trimmed.size()))
                <= options.threshold_dbfs;
            size_t run_end = std::min(
                position + window,
                trimmed.size());
            while (is_silent && run_end < trimmed.size()) {
                const size_t next_end = std::min(
                    run_end + window,
                    trimmed.size());
                if (window_db(trimmed, run_end, next_end) >
                    options.threshold_dbfs) {
                    break;
                }
                run_end = next_end;
            }

            if (is_silent && run_end - run_begin >= min_silence_samples) {
                const size_t left_keep = std::min(
                    keep,
                    run_end - run_begin);
                const size_t right_keep = std::min(
                    keep,
                    run_end - run_begin - left_keep);
                compacted.insert(
                    compacted.end(),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_begin),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(
                        run_begin + left_keep));
                compacted.insert(
                    compacted.end(),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(
                        run_end - right_keep),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_end));
            } else {
                compacted.insert(
                    compacted.end(),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_begin),
                    trimmed.begin() + static_cast<std::ptrdiff_t>(run_end));
            }
            position = run_end;
        }
        trimmed = std::move(compacted);
    }

    if (trimmed.empty()) {
        return Status::error("silence trimming produced empty audio");
    }
    audio.samples = std::move(trimmed);
    return Status::ok();
}

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
