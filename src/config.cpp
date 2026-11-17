#include "config.h"

#include "gguf_model.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace omnivoice {

namespace {

constexpr uint32_t SCHEMA_VERSION = 1;

Status required_key(
    const GGUFModel& model,
    const std::string& key,
    gguf_type type) {
    if (!model.has_key(key)) {
        return Status::error("GGUF is missing required key: " + key);
    }
    if (model.key_type(key) != type) {
        return Status::error("GGUF key has unexpected type: " + key);
    }
    return Status::ok();
}

Status required_array(
    const GGUFModel& model,
    const std::string& key,
    gguf_type element_type) {
    Status status = required_key(model, key, GGUF_TYPE_ARRAY);
    if (!status) {
        return status;
    }
    const gguf_context* raw = model.raw_gguf();
    const int64_t id = gguf_find_key(raw, key.c_str());
    if (id < 0 || gguf_get_arr_type(raw, id) != element_type) {
        return Status::error(
            "GGUF array has unexpected element type: " + key);
    }
    return Status::ok();
}

Status read_i32(
    const GGUFModel& model,
    const std::string& key,
    int32_t& value) {
    Status status = required_key(model, key, GGUF_TYPE_INT32);
    if (!status) {
        return status;
    }
    value = model.get_i32(key);
    return Status::ok();
}

Status read_u32(
    const GGUFModel& model,
    const std::string& key,
    int32_t& value) {
    Status status = required_key(model, key, GGUF_TYPE_UINT32);
    if (!status) {
        return status;
    }
    const uint32_t raw = model.get_u32(key);
    if (raw > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        return Status::error("GGUF value is too large: " + key);
    }
    value = static_cast<int32_t>(raw);
    return Status::ok();
}

Status read_f32(
    const GGUFModel& model,
    const std::string& key,
    float& value) {
    Status status = required_key(model, key, GGUF_TYPE_FLOAT32);
    if (!status) {
        return status;
    }
    value = model.get_f32(key);
    if (!std::isfinite(value)) {
        return Status::error("GGUF float is not finite: " + key);
    }
    return Status::ok();
}

Status read_bool(
    const GGUFModel& model,
    const std::string& key,
    bool& value) {
    Status status = required_key(model, key, GGUF_TYPE_BOOL);
    if (!status) {
        return status;
    }
    value = model.get_bool(key);
    return Status::ok();
}

Status read_string(
    const GGUFModel& model,
    const std::string& key,
    std::string& value) {
    Status status = required_key(model, key, GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    value = model.get_string(key);
    return Status::ok();
}

Status read_i32_array(
    const GGUFModel& model,
    const std::string& key,
    std::vector<int32_t>& values) {
    Status status = required_array(model, key, GGUF_TYPE_INT32);
    if (!status) {
        return status;
    }
    values = model.get_i32_array(key);
    return Status::ok();
}

Status require_positive(int32_t value, const std::string& name) {
    return value > 0
        ? Status::ok()
        : Status::error("configuration value must be positive: " + name);
}

Status require_positive(float value, const std::string& name) {
    return std::isfinite(value) && value > 0.0f
        ? Status::ok()
        : Status::error("configuration value must be positive: " + name);
}

Status require_schema_and_role(
    const GGUFModel& model,
    const std::string& expected_role) {
    Status status = required_key(
        model,
        "omnivoice.schema_version",
        GGUF_TYPE_UINT32);
    if (!status) {
        return status;
    }
    if (model.get_u32("omnivoice.schema_version") != SCHEMA_VERSION) {
        return Status::error("unsupported OmniVoice GGUF schema version");
    }
    status = required_key(
        model,
        "omnivoice.file_role",
        GGUF_TYPE_STRING);
    if (!status) {
        return status;
    }
    if (model.get_string("omnivoice.file_role") != expected_role) {
        return Status::error(
            "unexpected GGUF file role; expected " + expected_role);
    }
    return Status::ok();
}

Status optional_i32(
    const GGUFModel& model,
    const std::string& key,
    int32_t& value) {
    if (!model.has_key(key)) {
        return Status::ok();
    }
    return read_i32(model, key, value);
}

Status optional_u32(
    const GGUFModel& model,
    const std::string& key,
    int32_t& value) {
    if (!model.has_key(key)) {
        return Status::ok();
    }
    return read_u32(model, key, value);
}

Status optional_f32(
    const GGUFModel& model,
    const std::string& key,
    float& value) {
    if (!model.has_key(key)) {
        return Status::ok();
    }
    return read_f32(model, key, value);
}

Status optional_bool(
    const GGUFModel& model,
    const std::string& key,
    bool& value) {
    if (!model.has_key(key)) {
        return Status::ok();
    }
    return read_bool(model, key, value);
}

} // namespace

Status load_qwen3_config(
    const GGUFModel& model,
    Qwen3Config& config) {
    config = {};
    const std::pair<const char*, int32_t*> values[] = {
        {"omnivoice.llm.hidden_size", &config.hidden_size},
        {"omnivoice.llm.intermediate_size", &config.intermediate_size},
        {"omnivoice.llm.block_count", &config.num_hidden_layers},
        {"omnivoice.llm.attention_head_count", &config.num_attention_heads},
        {
            "omnivoice.llm.attention_head_count_kv",
            &config.num_key_value_heads
        },
        {"omnivoice.llm.attention_key_length", &config.head_dim},
        {
            "omnivoice.llm.context_length",
            &config.max_position_embeddings
        },
        {"omnivoice.llm.vocab_size", &config.vocab_size},
    };
    for (const auto& [key, destination] : values) {
        Status status = read_u32(model, key, *destination);
        if (!status) {
            return status;
        }
    }

    Status status = read_f32(
        model,
        "omnivoice.llm.rms_norm_epsilon",
        config.rms_norm_epsilon);
    if (!status) {
        return status;
    }
    status = read_f32(
        model,
        "omnivoice.llm.rope_theta",
        config.rope_theta);
    if (!status) {
        return status;
    }
    status = read_string(
        model,
        "omnivoice.llm.hidden_activation",
        config.hidden_activation);
    if (!status) {
        return status;
    }
    status = read_string(
        model,
        "omnivoice.llm.rope_type",
        config.rope_type);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "omnivoice.llm.attention_bias",
        config.attention_bias);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "omnivoice.llm.tie_word_embeddings",
        config.tie_word_embeddings);
    if (!status) {
        return status;
    }
    status = read_bool(
        model,
        "omnivoice.llm.use_sliding_window",
        config.use_sliding_window);
    if (!status) {
        return status;
    }

    const std::pair<int32_t, const char*> positive[] = {
        {config.hidden_size, "hidden_size"},
        {config.intermediate_size, "intermediate_size"},
        {config.num_hidden_layers, "block_count"},
        {config.num_attention_heads, "attention_head_count"},
        {config.num_key_value_heads, "attention_head_count_kv"},
        {config.head_dim, "attention_key_length"},
        {config.max_position_embeddings, "context_length"},
        {config.vocab_size, "vocab_size"},
    };
    for (const auto& [value, name] : positive) {
        status = require_positive(value, name);
        if (!status) {
            return status;
        }
    }
    if (config.num_key_value_heads > config.num_attention_heads ||
        config.num_attention_heads % config.num_key_value_heads != 0) {
        return Status::error("invalid Qwen3 GQA head configuration");
    }
    if (config.rms_norm_epsilon <= 0.0f || config.rope_theta <= 0.0f) {
        return Status::error("invalid Qwen3 norm or RoPE configuration");
    }
    return Status::ok();
}

Status load_omnivoice_lm_config(
    const GGUFModel& model,
    OmniVoiceLMConfig& config) {
    config = {};
    Status status = require_schema_and_role(model, "lm");
    if (!status) {
        return status;
    }
    status = load_qwen3_config(model, config.llm);
    if (!status) {
        return status;
    }

    const std::pair<const char*, int32_t*> values[] = {
        {"omnivoice.audio.vocab_size", &config.audio_vocab_size},
        {"omnivoice.audio.mask_id", &config.audio_mask_id},
        {"omnivoice.audio.codebook_count", &config.num_audio_codebooks},
        {"omnivoice.audio.frame_rate", &config.audio_frame_rate},
    };
    for (const auto& [key, destination] : values) {
        status = read_u32(model, key, *destination);
        if (!status) {
            return status;
        }
    }
    status = read_i32_array(
        model,
        "omnivoice.audio.codebook_weights",
        config.audio_codebook_weights);
    if (!status) {
        return status;
    }

    status = require_positive(config.audio_vocab_size, "audio.vocab_size");
    if (!status) {
        return status;
    }
    status = require_positive(
        config.num_audio_codebooks,
        "audio.codebook_count");
    if (!status) {
        return status;
    }
    status = require_positive(config.audio_frame_rate, "audio.frame_rate");
    if (!status) {
        return status;
    }
    if (config.audio_mask_id < 0 ||
        config.audio_mask_id >= config.audio_vocab_size) {
        return Status::error("audio MASK ID is outside the audio vocabulary");
    }
    if (config.audio_codebook_weights.size() !=
        static_cast<size_t>(config.num_audio_codebooks)) {
        return Status::error("audio codebook weights have the wrong size");
    }
    for (int32_t weight : config.audio_codebook_weights) {
        if (weight <= 0) {
            return Status::error("audio codebook weights must be positive");
        }
    }
    return Status::ok();
}

Status load_higgs_tokenizer_config(
    const GGUFModel& model,
    HiggsTokenizerConfig& config) {
    config = {};
    Status status = require_schema_and_role(model, "audio_tokenizer");
    if (!status) {
        return status;
    }
    status = read_string(
        model,
        "omnivoice.audio_tokenizer.model_type",
        config.model_type);
    if (!status) {
        return status;
    }

    const std::pair<const char*, int32_t*> values[] = {
        {"omnivoice.audio_tokenizer.sample_rate", &config.sample_rate},
        {
            "omnivoice.audio_tokenizer.semantic_sample_rate",
            &config.semantic_sample_rate
        },
        {"omnivoice.audio_tokenizer.hop_length", &config.hop_length},
        {"omnivoice.audio_tokenizer.frame_rate", &config.frame_rate},
        {"omnivoice.audio_tokenizer.codebook_count", &config.codebook_count},
        {"omnivoice.audio_tokenizer.codebook_size", &config.codebook_size},
        {"omnivoice.audio_tokenizer.codebook_dim", &config.codebook_dim},
        {
            "omnivoice.audio_tokenizer.semantic_hidden_size",
            &config.semantic_hidden_size
        },
        {
            "omnivoice.audio_tokenizer.semantic_attention_heads",
            &config.semantic_attention_heads
        },
        {
            "omnivoice.audio_tokenizer.semantic_layer_count",
            &config.semantic_layer_count
        },
        {
            "omnivoice.audio_tokenizer.acoustic_hidden_size",
            &config.acoustic_hidden_size
        },
        {
            "omnivoice.audio_tokenizer.acoustic_encoder_hidden_size",
            &config.acoustic_encoder_hidden_size
        },
        {
            "omnivoice.audio_tokenizer.acoustic_decoder_hidden_size",
            &config.acoustic_decoder_hidden_size
        },
    };
    for (const auto& [key, destination] : values) {
        status = read_u32(model, key, *destination);
        if (!status) {
            return status;
        }
    }

    const std::pair<const char*, std::vector<int32_t>*> arrays[] = {
        {
            "omnivoice.audio_tokenizer.semantic_conv_dim",
            &config.semantic_conv_dim
        },
        {
            "omnivoice.audio_tokenizer.semantic_conv_kernel",
            &config.semantic_conv_kernel
        },
        {
            "omnivoice.audio_tokenizer.semantic_conv_stride",
            &config.semantic_conv_stride
        },
        {
            "omnivoice.audio_tokenizer.downsampling_ratios",
            &config.downsampling_ratios
        },
        {
            "omnivoice.audio_tokenizer.upsampling_ratios",
            &config.upsampling_ratios
        },
    };
    for (const auto& [key, destination] : arrays) {
        status = read_i32_array(model, key, *destination);
        if (!status) {
            return status;
        }
        if (destination->empty()) {
            return Status::error(
                "GGUF array must not be empty: " + std::string(key));
        }
        for (int32_t value : *destination) {
            if (value <= 0) {
                return Status::error(
                    "GGUF array values must be positive: " +
                    std::string(key));
            }
        }
    }

    if (config.sample_rate != 24000 ||
        config.semantic_sample_rate != 16000 ||
        config.hop_length != 960 ||
        config.frame_rate != 25) {
        return Status::error("unsupported Higgs audio tokenizer timing");
    }
    const std::pair<int32_t, const char*> positive[] = {
        {config.codebook_count, "codebook_count"},
        {config.codebook_size, "codebook_size"},
        {config.codebook_dim, "codebook_dim"},
        {config.semantic_hidden_size, "semantic_hidden_size"},
        {
            config.semantic_attention_heads,
            "semantic_attention_heads"
        },
        {config.semantic_layer_count, "semantic_layer_count"},
        {config.acoustic_hidden_size, "acoustic_hidden_size"},
        {
            config.acoustic_encoder_hidden_size,
            "acoustic_encoder_hidden_size"
        },
        {
            config.acoustic_decoder_hidden_size,
            "acoustic_decoder_hidden_size"
        },
    };
    for (const auto& [value, name] : positive) {
        status = require_positive(value, name);
        if (!status) {
            return status;
        }
    }
    if (config.semantic_conv_dim.size() !=
            config.semantic_conv_kernel.size() ||
        config.semantic_conv_dim.size() !=
            config.semantic_conv_stride.size()) {
        return Status::error("Higgs semantic conv arrays have inconsistent sizes");
    }
    return Status::ok();
}

Status load_voice_clone_config(
    const GGUFModel& model,
    VoiceCloneConfig& config) {
    config = {};
    Status status = optional_i32(
        model,
        "omnivoice.generation.num_step",
        config.num_step);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.guidance_scale",
        config.guidance_scale);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.t_shift",
        config.t_shift);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.layer_penalty_factor",
        config.layer_penalty_factor);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.position_temperature",
        config.position_temperature);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.class_temperature",
        config.class_temperature);
    if (!status) {
        return status;
    }
    status = optional_bool(
        model,
        "omnivoice.generation.denoise",
        config.denoise);
    if (!status) {
        return status;
    }
    status = optional_bool(
        model,
        "omnivoice.generation.preprocess_prompt",
        config.preprocess_prompt);
    if (!status) {
        return status;
    }
    status = optional_bool(
        model,
        "omnivoice.generation.postprocess_output",
        config.postprocess_output);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.audio_chunk_duration",
        config.audio_chunk_duration);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.audio_chunk_threshold",
        config.audio_chunk_threshold);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.pad_duration",
        config.pad_duration);
    if (!status) {
        return status;
    }
    status = optional_f32(
        model,
        "omnivoice.generation.fade_duration",
        config.fade_duration);
    if (!status) {
        return status;
    }

    if (config.num_step <= 0 ||
        config.guidance_scale < 0.0f ||
        config.t_shift <= 0.0f ||
        config.layer_penalty_factor < 0.0f ||
        config.position_temperature < 0.0f ||
        config.class_temperature < 0.0f) {
        return Status::error("invalid Voice Clone decoding configuration");
    }
    if (config.audio_chunk_duration <= 0.0f ||
        config.audio_chunk_threshold <= 0.0f ||
        config.pad_duration < 0.0f ||
        config.fade_duration < 0.0f) {
        return Status::error("invalid Voice Clone audio configuration");
    }
    return Status::ok();
}

Status load_omnivoice_config(
    const GGUFModel& lm_model,
    const GGUFModel& audio_model,
    OmniVoiceConfig& config) {
    config = {};
    Status status = load_omnivoice_lm_config(lm_model, config.lm);
    if (!status) {
        return status;
    }
    status = load_text_tokenizer_config_from_gguf(
        lm_model,
        config.text_tokenizer);
    if (!status) {
        return status;
    }
    status = load_higgs_tokenizer_config(
        audio_model,
        config.audio_tokenizer);
    if (!status) {
        return status;
    }
    status = load_voice_clone_config(
        lm_model,
        config.voice_clone);
    if (!status) {
        return status;
    }

    if (config.text_tokenizer.vocab_size != config.lm.llm.vocab_size) {
        return Status::error(
            "text tokenizer vocabulary does not match the LM vocabulary");
    }
    if (config.lm.audio_frame_rate !=
        config.audio_tokenizer.frame_rate) {
        return Status::error(
            "LM and audio tokenizer frame rates do not match");
    }
    if (config.lm.num_audio_codebooks !=
        config.audio_tokenizer.codebook_count) {
        return Status::error(
            "LM and audio tokenizer codebook counts do not match");
    }
    return Status::ok();
}

} // namespace omnivoice
