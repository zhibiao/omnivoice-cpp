#pragma once

#include "config.h"
#include "qwen3_lm/qwen3_block.h"
#include "status.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace omnivoice {

class GGUFModel;

struct OmniVoiceLmInput {
    int32_t sequence_length = 0;
    int32_t codebooks = 0;
    int32_t target_start = 0;
    int32_t target_frames = 0;

    // Codebook-major [codebooks, sequence_length].
    std::vector<int32_t> input_ids;
    std::vector<uint8_t> audio_mask;
};

struct OmniVoiceLmOutput {
    // [hidden_size, sequence_length], sequence-major float layout.
    std::vector<float> hidden;
    // Codebook-major [codebooks, target_frames, audio_vocab_size].
    std::vector<float> target_logits;
    // Frame-major LM output kept on the selected backend for graph selection.
    const ggml_tensor* target_logits_tensor = nullptr;
};

class OmniVoiceLM {
public:
    OmniVoiceLM();
    ~OmniVoiceLM();

    OmniVoiceLM(const OmniVoiceLM&) = delete;
    OmniVoiceLM& operator=(const OmniVoiceLM&) = delete;

    // The backend is borrowed and must outlive this model.
    Status load(
        const OmniVoiceLMConfig& config,
        const GGUFModel& model,
        ggml_backend_t backend);

    Status forward(
        const OmniVoiceLmInput& input,
        OmniVoiceLmOutput& output) const;

    Status init_diffusion_graphs(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        bool read_host_logits);
    Status run_conditional_graph(
        const OmniVoiceLmInput& input,
        OmniVoiceLmOutput& output);
    Status run_unconditional_graph(
        const OmniVoiceLmInput& input,
        OmniVoiceLmOutput& output);
    void release_diffusion_graphs() noexcept;
    bool supports_selector_graph() const noexcept;
    ggml_backend_t backend() const noexcept;

private:
    class LmGraph;

    // OmniVoice LM graph:
    //   lm_text_ids / lm_audio_ids.* / lm_positions
    //     -> lm_hidden / lm_audio_logits
    ggml_cgraph* build_graph(
        ggml_context* context,
        const OmniVoiceLmInput& input,
        bool expose_hidden) const;

    ggml_tensor* weight(const std::string& name) const;
    Status validate_input(const OmniVoiceLmInput& input) const;
    void release() noexcept;

    OmniVoiceLMConfig config_;
    ggml_backend_t backend_ = nullptr;
    ggml_context* weight_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> weights_;
    std::vector<Qwen3BlockWeights> blocks_;
    std::vector<ggml_tensor*> qkv_weights_;
    std::unique_ptr<LmGraph> conditional_graph_;
    std::unique_ptr<LmGraph> unconditional_graph_;
    bool diffusion_read_host_logits_ = true;
    bool loaded_ = false;
};

} // namespace omnivoice
