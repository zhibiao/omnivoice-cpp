#include "omnivoice.h"

#include "audio_kit.h"
#include "duration_estimator.h"
#include "gguf_model.h"
#include "text_tokenizer/text_chunker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace omnivoice {

namespace {

using Clock = std::chrono::steady_clock;

std::string format_number(double value, int precision = 2) {
    char text[64];
    std::snprintf(
        text,
        sizeof(text),
        "%.*f",
        precision,
        value);
    return text;
}

std::string elapsed_milliseconds(Clock::time_point start) {
    return format_number(
        std::chrono::duration<double, std::milli>(
            Clock::now() - start).count());
}

} // namespace

OmniVoiceModel::OmniVoiceModel() = default;

OmniVoiceModel::~OmniVoiceModel() {
    release();
}

void OmniVoiceModel::release() noexcept {
    loaded_ = false;
    decoder_.reset();
    encoder_.reset();
    lm_.reset();
    tokenizer_ = {};
    config_ = {};
    runtime_ = nullptr;
}

Status OmniVoiceModel::load(
    GGMLRuntime& runtime,
    const LogCallback& log) {
    if (loaded_) {
        return Status::error("OmniVoiceModel is already loaded");
    }
    if (!runtime.is_ready()) {
        return Status::error("GGML runtime is not ready");
    }

    const auto load_start = Clock::now();
    log_message(log, "loading text tokenizer");
    TextTokenizer tokenizer;
    Status status = tokenizer.load(runtime.config().text_tokenizer);
    if (!status) {
        return status;
    }
    log_message(
        log,
        "text tokenizer ready in " +
            elapsed_milliseconds(load_start) + " ms");

    log_message(
        log,
        "loading OmniVoice LM (" +
            std::to_string(runtime.lm_gguf().tensor_count()) +
            " tensors)");
    const auto lm_start = Clock::now();
    std::unique_ptr<OmniVoiceLM> lm =
        std::make_unique<OmniVoiceLM>();
    status = lm->load(
        runtime.config().lm,
        runtime.lm_gguf(),
        runtime.backend());
    if (!status) {
        return status;
    }
    log_message(
        log,
        "OmniVoice LM ready in " +
            elapsed_milliseconds(lm_start) + " ms");

    log_message(
        log,
        "loading Higgs audio encoder (" +
            std::to_string(runtime.audio_gguf().tensor_count()) +
            " tensors)");
    const auto encoder_start = Clock::now();
    std::unique_ptr<HiggsAudioEncoder> encoder =
        std::make_unique<HiggsAudioEncoder>();
    status = encoder->load(
        runtime.config().audio_tokenizer,
        runtime.audio_gguf(),
        runtime.backend());
    if (!status) {
        return status;
    }
    log_message(
        log,
        "Higgs audio encoder ready in " +
            elapsed_milliseconds(encoder_start) + " ms");

    log_message(log, "loading Higgs audio decoder");
    const auto decoder_start = Clock::now();
    std::unique_ptr<HiggsAudioDecoder> decoder =
        std::make_unique<HiggsAudioDecoder>();
    status = decoder->load(
        runtime.config().audio_tokenizer,
        runtime.audio_gguf(),
        runtime.backend());
    if (!status) {
        return status;
    }
    log_message(
        log,
        "Higgs audio decoder ready in " +
            elapsed_milliseconds(decoder_start) + " ms");

    runtime_ = &runtime;
    config_ = runtime.config();
    tokenizer_ = std::move(tokenizer);
    lm_ = std::move(lm);
    encoder_ = std::move(encoder);
    decoder_ = std::move(decoder);
    loaded_ = true;
    log_message(
        log,
        "OmniVoice model ready in " +
            elapsed_milliseconds(load_start) + " ms");
    return Status::ok();
}

bool OmniVoiceModel::is_loaded() const noexcept {
    return loaded_;
}

VoiceCloneConfig OmniVoiceModel::generation_defaults() const noexcept {
    return config_.voice_clone;
}

Status OmniVoiceModel::create_prompt(
    const VoiceClonePromptRequest& request,
    VoiceClonePrompt& prompt) const {
    if (!loaded_) {
        return Status::error("OmniVoice model is not loaded");
    }
    log_message(
        request.log,
        "encoding reference prompt: audio=" +
            format_number(request.reference_audio.duration_seconds()) +
            " s, RMS=" +
            format_number(compute_rms(request.reference_audio), 4) +
            ", preprocess=" +
            (request.preprocess_reference ? "on" : "off") +
            ", denoise=" + (request.denoise ? "on" : "off"));
    const auto encode_start = Clock::now();
    Status status = create_voice_clone_prompt(
        *encoder_,
        request.reference_audio,
        request.reference_text,
        request.language,
        request.denoise,
        request.preprocess_reference,
        prompt);
    if (!status) {
        return status;
    }
    log_message(
        request.log,
        "reference prompt ready: " +
            std::to_string(prompt.reference_codes.codebooks) +
            " x " + std::to_string(prompt.reference_codes.frames) +
            " codes in " + elapsed_milliseconds(encode_start) + " ms");
    return Status::ok();
}

Status OmniVoiceModel::decode_codes(
    const AudioCodes& codes,
    AudioBuffer& audio,
    const LogCallback& log) const {
    audio = {};
    if (!loaded_) {
        return Status::error("OmniVoice model is not loaded");
    }
    log_message(
        log,
        "decoding audio: " + std::to_string(codes.codebooks) +
            " x " + std::to_string(codes.frames) + " codes");
    const auto decode_start = Clock::now();
    Status status = decoder_->decode(codes, audio);
    if (status) {
        log_message(
            log,
            "audio decoded: " +
                std::to_string(audio.samples.size()) +
                " samples, " + format_number(audio.duration_seconds()) +
                " s in " + elapsed_milliseconds(decode_start) + " ms");
    }
    return status;
}

Status OmniVoiceModel::generate_voice_clone(
    const VoiceClonePrompt& prompt,
    const VoiceCloneRequest& request,
    VoiceCloneOutput& output) const {
    const auto clone_start = Clock::now();
    output = {};
    if (!loaded_) {
        return Status::error("OmniVoice model is not loaded");
    }
    if (request.target_text.empty()) {
        return Status::error("target text must not be empty");
    }
    if (request.speed <= 0.0f) {
        return Status::error("speech speed must be positive");
    }
    if (!std::isfinite(request.duration_seconds) ||
        request.duration_seconds < 0.0f) {
        return Status::error(
            "speech duration must be finite and non-negative");
    }

    output.prompt = prompt;
    log_message(
        request.log,
        "using voice prompt: " +
            std::to_string(output.prompt.reference_codes.codebooks) +
            " x " +
            std::to_string(output.prompt.reference_codes.frames) +
            " codes");
    Status status = validate_voice_clone_prompt(output.prompt);
    if (!status) {
        output = {};
        return status;
    }

    int32_t target_frames = request.target_frames;
    if (request.duration_seconds > 0.0f) {
        target_frames = std::max(
            1,
            static_cast<int32_t>(
                request.duration_seconds *
                static_cast<float>(config_.audio_tokenizer.frame_rate)));
        log_message(
            request.log,
            "target frames: " + std::to_string(target_frames) +
                " (" + format_number(request.duration_seconds) +
                " s at " +
                std::to_string(config_.audio_tokenizer.frame_rate) +
                " Hz)");
    }
    if (target_frames <= 0) {
        status = estimate_target_frames(
            request.target_text,
            output.prompt.reference_text,
            output.prompt.reference_codes.frames,
            request.speed,
            target_frames);
        if (!status) {
            output = {};
            return status;
        }
        log_message(
            request.log,
            "estimated target frames: " +
                std::to_string(target_frames) + " (" +
                format_number(
                    static_cast<double>(target_frames) /
                    static_cast<double>(config_.audio_tokenizer.frame_rate)) +
                " s)");
    } else if (request.duration_seconds <= 0.0f) {
        log_message(
            request.log,
            "target frames: " + std::to_string(target_frames) + " (" +
                format_number(
                    static_cast<double>(target_frames) /
                    static_cast<double>(config_.audio_tokenizer.frame_rate)) +
                " s)");
    }

    const VoiceCloneConfig options =
        request.generation_config.value_or(config_.voice_clone);
    log_message(
        request.log,
        "generation: steps=" + std::to_string(options.num_step) +
            ", guidance=" + format_number(options.guidance_scale, 3) +
            ", chunk threshold=" +
            format_number(options.audio_chunk_threshold) + " s");
    const auto generate_chunk =
        [&](size_t chunk_index,
            size_t chunk_count,
            const std::string& text,
            int32_t frames,
            uint64_t seed,
            PromptPlan& plan,
            AudioCodes& codes,
            AudioBuffer& audio,
            std::vector<int32_t>* counts) -> Status {
        const std::string chunk_label =
            "chunk " + std::to_string(chunk_index + 1) +
            "/" + std::to_string(chunk_count);
        log_message(
            request.log,
            chunk_label + ": building LM prompt, target frames=" +
                std::to_string(frames));
        const auto prompt_start = Clock::now();
        VoiceClonePromptInput prompt_input;
        prompt_input.reference_text = output.prompt.reference_text;
        prompt_input.target_text = text;
        prompt_input.language = request.language;
        prompt_input.reference_codes = output.prompt.reference_codes;
        prompt_input.target_frames = frames;
        prompt_input.denoise = request.denoise;
        Status local_status = build_voice_clone_prompt(
            tokenizer_,
            config_,
            prompt_input,
            plan);
        if (!local_status) {
            return local_status;
        }
        log_message(
            request.log,
            chunk_label + ": prompt ready, sequence length=" +
                std::to_string(plan.layout.sequence_length) +
                " in " + elapsed_milliseconds(prompt_start) + " ms");

        LogCallback diffusion_log;
        if (request.log != nullptr) {
            diffusion_log =
                [log = request.log, chunk_label](
                    const std::string& line) {
                    log(chunk_label + ": " + line);
                };
        }
        DiffusionDecodeResult decoded;
        local_status = run_diffusion_decoder(
            plan,
            config_.lm,
            options,
            seed,
            *lm_,
            decoded,
            diffusion_log);
        lm_->release_diffusion_graph();
        if (!local_status) {
            return local_status;
        }
        codes = std::move(decoded.target_codes);
        if (counts != nullptr) {
            *counts = std::move(decoded.counts_per_step);
        }
        return decode_codes(codes, audio, diffusion_log);
    };

    std::vector<std::string> chunks;
    status = plan_text_chunks(
        request.target_text,
        target_frames,
        config_.audio_tokenizer.frame_rate,
        options.audio_chunk_duration,
        options.audio_chunk_threshold,
        chunks);
    if (!status) {
        output = {};
        return status;
    }
    log_message(
        request.log,
        "text chunks: " + std::to_string(chunks.size()));

    std::vector<AudioBuffer> chunk_audio;
    chunk_audio.reserve(chunks.size());
    output.chunk_prompt_plans.resize(chunks.size());
    output.chunk_codes.resize(chunks.size());
    for (size_t index = 0; index < chunks.size(); ++index) {
        int32_t frames = target_frames;
        if (chunks.size() > 1) {
            status = estimate_target_frames(
                chunks[index],
                output.prompt.reference_text,
                output.prompt.reference_codes.frames,
                request.speed,
                frames);
            if (!status) {
                output = {};
                return status;
            }
            log_message(
                request.log,
                "chunk " + std::to_string(index + 1) + "/" +
                    std::to_string(chunks.size()) +
                    ": estimated target frames=" +
                    std::to_string(frames));
        }
        AudioBuffer audio;
        std::vector<int32_t> counts;
        status = generate_chunk(
            index,
            chunks.size(),
            chunks[index],
            frames,
            request.seed + index,
            output.chunk_prompt_plans[index],
            output.chunk_codes[index],
            audio,
            index == 0 ? &counts : nullptr);
        if (!status) {
            output = {};
            return status;
        }
        if (index == 0) {
            output.counts_per_step = std::move(counts);
        }
        chunk_audio.push_back(std::move(audio));
    }

    output.prompt_plan = output.chunk_prompt_plans.front();
    output.target_codes = output.chunk_codes.front();
    if (chunks.size() == 1) {
        output.audio = std::move(chunk_audio.front());
    } else {
        log_message(
            request.log,
            "cross-fading " + std::to_string(chunks.size()) +
                " audio chunks");
        const auto cross_fade_start = Clock::now();
        status = cross_fade_audio_chunks(
            chunk_audio,
            24000,
            0.3f,
            output.audio);
        if (!status) {
            output = {};
            return status;
        }
        log_message(
            request.log,
            "audio chunks joined in " +
                elapsed_milliseconds(cross_fade_start) + " ms");
    }
    log_message(request.log, "postprocessing generated audio");
    const auto postprocess_start = Clock::now();
    status = postprocess_generated_audio(
        options,
        output.prompt.reference_rms,
        output.audio);
    if (!status) {
        output = {};
        return status;
    }
    log_message(
        request.log,
        "postprocessing ready in " +
            elapsed_milliseconds(postprocess_start) + " ms");
    log_message(
        request.log,
        "voice clone ready: " +
            std::to_string(output.audio.samples.size()) +
            " samples, " + format_number(output.audio.duration_seconds()) +
            " s in " + elapsed_milliseconds(clone_start) + " ms");
    return Status::ok();
}

} // namespace omnivoice
