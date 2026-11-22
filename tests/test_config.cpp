#include "config.h"
#include "gguf_model.h"
#include "text_tokenizer/text_tokenizer.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

namespace {

bool close_to(float actual, float expected, float tolerance = 1e-6f) {
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main() {
    const std::string data_dir = OMNIVOICE_TEST_DATA_DIR;

    std::unique_ptr<omnivoice::GGUFModel> lm_gguf;
    omnivoice::Status status = omnivoice::GGUFModel::open(
        data_dir + "/omnivoice-lm-config-m3.gguf",
        lm_gguf);
    if (!status) {
        std::cerr << "LM config GGUF load failed: "
                  << status.message() << '\n';
        return 1;
    }

    std::unique_ptr<omnivoice::GGUFModel> audio_gguf;
    status = omnivoice::GGUFModel::open(
        data_dir + "/omnivoice-audio-config-m3.gguf",
        audio_gguf);
    if (!status) {
        std::cerr << "audio config GGUF load failed: "
                  << status.message() << '\n';
        return 1;
    }

    omnivoice::OmniVoiceConfig config;
    status = omnivoice::load_omnivoice_config(
        *lm_gguf,
        *audio_gguf,
        config);
    if (!status) {
        std::cerr << "OmniVoice config load failed: "
                  << status.message() << '\n';
        return 1;
    }

    if (config.lm.llm.hidden_size != 1024 ||
        config.lm.llm.intermediate_size != 3072 ||
        config.lm.llm.num_hidden_layers != 28 ||
        config.lm.llm.num_attention_heads != 16 ||
        config.lm.llm.num_key_value_heads != 8 ||
        config.lm.llm.head_dim != 128 ||
        config.lm.llm.vocab_size != 151676 ||
        config.lm.llm.hidden_activation != "silu" ||
        config.lm.llm.rope_type != "default" ||
        config.lm.llm.attention_bias ||
        !config.lm.llm.tie_word_embeddings ||
        config.lm.llm.use_sliding_window) {
        std::cerr << "Qwen3 config mismatch\n";
        return 1;
    }
    if (!close_to(config.lm.llm.rope_theta, 1000000.0f) ||
        !close_to(config.lm.llm.rms_norm_epsilon, 1e-6f)) {
        std::cerr << "Qwen3 float config mismatch\n";
        return 1;
    }

    if (config.lm.audio_vocab_size != 1025 ||
        config.lm.audio_mask_id != 1024 ||
        config.lm.num_audio_codebooks != 8 ||
        config.lm.audio_frame_rate != 25 ||
        config.lm.audio_codebook_weights !=
            std::vector<int32_t>({8, 8, 6, 6, 4, 4, 2, 2})) {
        std::cerr << "OmniVoice audio config mismatch\n";
        return 1;
    }

    if (config.audio_tokenizer.model_type != "higgs_audio_v2_tokenizer" ||
        config.audio_tokenizer.sample_rate != 24000 ||
        config.audio_tokenizer.semantic_sample_rate != 16000 ||
        config.audio_tokenizer.hop_length != 960 ||
        config.audio_tokenizer.frame_rate != 25 ||
        config.audio_tokenizer.codebook_count != 8 ||
        config.audio_tokenizer.codebook_size != 1024 ||
        config.audio_tokenizer.codebook_dim != 64 ||
        config.audio_tokenizer.semantic_hidden_size != 768 ||
        config.audio_tokenizer.semantic_attention_heads != 12 ||
        config.audio_tokenizer.semantic_layer_count != 12 ||
        config.audio_tokenizer.acoustic_hidden_size != 256 ||
        config.audio_tokenizer.acoustic_encoder_hidden_size != 64 ||
        config.audio_tokenizer.acoustic_decoder_hidden_size != 1024) {
        std::cerr << "Higgs tokenizer config mismatch\n";
        return 1;
    }
    if (config.audio_tokenizer.semantic_conv_dim.size() != 7 ||
        config.audio_tokenizer.semantic_conv_kernel.size() != 7 ||
        config.audio_tokenizer.semantic_conv_stride.size() != 7 ||
        config.audio_tokenizer.downsampling_ratios !=
            std::vector<int32_t>({8, 5, 4, 2, 3}) ||
        config.audio_tokenizer.upsampling_ratios !=
            std::vector<int32_t>({8, 5, 4, 2, 3})) {
        std::cerr << "Higgs conv config mismatch\n";
        return 1;
    }

    if (config.text_tokenizer.vocab_size != 151676 ||
        config.text_tokenizer.pad_token_id != 151643 ||
        config.text_tokenizer.eos_token_id != 151645 ||
        !config.text_tokenizer.nfc_normalization) {
        std::cerr << "text tokenizer config mismatch\n";
        return 1;
    }

    if (config.voice_clone.num_step != 32 ||
        !close_to(config.voice_clone.guidance_scale, 2.0f) ||
        !close_to(config.voice_clone.t_shift, 0.1f) ||
        !close_to(config.voice_clone.layer_penalty_factor, 5.0f) ||
        !close_to(config.voice_clone.position_temperature, 5.0f) ||
        !close_to(config.voice_clone.class_temperature, 0.0f) ||
        !config.voice_clone.denoise ||
        !config.voice_clone.preprocess_prompt ||
        !config.voice_clone.postprocess_output) {
        std::cerr << "Voice Clone config mismatch\n";
        return 1;
    }

    omnivoice::TextTokenizer tokenizer;
    status = tokenizer.load(config.text_tokenizer);
    if (!status ||
        tokenizer.special_token_id("<|text_start|>") != 151674 ||
        tokenizer.special_token_id("<|text_end|>") != 151675) {
        std::cerr << "configured text tokenizer is unusable\n";
        return 1;
    }

    if (omnivoice::load_higgs_tokenizer_config(*lm_gguf, config.audio_tokenizer)) {
        std::cerr << "LM config GGUF was incorrectly accepted as audio tokenizer\n";
        return 1;
    }
    if (omnivoice::load_omnivoice_lm_config(*audio_gguf, config.lm)) {
        std::cerr << "audio config GGUF was incorrectly accepted as LM\n";
        return 1;
    }

    return 0;
}
