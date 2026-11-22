#pragma once

#include "status.h"
#include "text_tokenizer/text_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace omnivoice {

class GGUFModel;

struct Qwen3Config {
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t max_position_embeddings = 0;
    int32_t vocab_size = 0;
    float rms_norm_epsilon = 0.0f;
    float rope_theta = 0.0f;
    std::string hidden_activation;
    std::string rope_type;
    bool attention_bias = false;
    bool tie_word_embeddings = false;
    bool use_sliding_window = false;
};

struct OmniVoiceLMConfig {
    Qwen3Config llm;
    int32_t audio_vocab_size = 0;
    int32_t audio_mask_id = 0;
    int32_t num_audio_codebooks = 0;
    int32_t audio_frame_rate = 0;
    std::vector<int32_t> audio_codebook_weights;
};

struct HiggsTokenizerConfig {
    std::string model_type;
    int32_t sample_rate = 0;
    int32_t semantic_sample_rate = 0;
    int32_t hop_length = 0;
    int32_t frame_rate = 0;
    int32_t codebook_count = 0;
    int32_t codebook_size = 0;
    int32_t codebook_dim = 0;
    int32_t semantic_hidden_size = 0;
    int32_t semantic_attention_heads = 0;
    int32_t semantic_layer_count = 0;
    int32_t acoustic_hidden_size = 0;
    int32_t acoustic_encoder_hidden_size = 0;
    int32_t acoustic_decoder_hidden_size = 0;
    std::vector<int32_t> semantic_conv_dim;
    std::vector<int32_t> semantic_conv_kernel;
    std::vector<int32_t> semantic_conv_stride;
    std::vector<int32_t> downsampling_ratios;
    std::vector<int32_t> upsampling_ratios;
};

struct VoiceCloneConfig {
    int32_t num_step = 32;
    float guidance_scale = 2.0f;
    float t_shift = 0.1f;
    float layer_penalty_factor = 5.0f;
    float position_temperature = 5.0f;
    float class_temperature = 0.0f;
    bool denoise = true;
    bool preprocess_prompt = true;
    bool postprocess_output = true;
    float audio_chunk_duration = 15.0f;
    float audio_chunk_threshold = 30.0f;
    float pad_duration = 0.1f;
    float fade_duration = 0.1f;
};

struct OmniVoiceConfig {
    OmniVoiceLMConfig lm;
    TextTokenizerConfig text_tokenizer;
    HiggsTokenizerConfig audio_tokenizer;
    VoiceCloneConfig voice_clone;
};

Status load_qwen3_config(
    const GGUFModel& model,
    Qwen3Config& config);

Status load_omnivoice_lm_config(
    const GGUFModel& model,
    OmniVoiceLMConfig& config);

Status load_higgs_tokenizer_config(
    const GGUFModel& model,
    HiggsTokenizerConfig& config);

Status load_voice_clone_config(
    const GGUFModel& model,
    VoiceCloneConfig& config);

Status load_omnivoice_config(
    const GGUFModel& lm_model,
    const GGUFModel& audio_model,
    OmniVoiceConfig& config);

} // namespace omnivoice
