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

struct OmniVoiceLmPackedOutput {
    // Host logits are only populated for the stochastic sampler path.
    // Codebook-major [codebooks, target_frames, audio_vocab_size].
    std::vector<float> conditional_target_logits;
    std::vector<float> unconditional_target_logits;

    // Codebook-major [codebooks, target_frames].
    std::vector<int32_t> predictions;
    std::vector<float> confidence;

    const ggml_tensor* conditional_target_logits_tensor = nullptr;
    const ggml_tensor* unconditional_target_logits_tensor = nullptr;
    const ggml_tensor* predictions_tensor = nullptr;
    const ggml_tensor* confidence_tensor = nullptr;

    // Only populated when the graph is initialized with expose_hidden.
    // Sequence-major [hidden_size, conditional_length + unconditional_length]:
    // the conditional branch occupies the first conditional_length frames.
    std::vector<float> hidden;
    const ggml_tensor* hidden_tensor = nullptr;
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

    Status init_diffusion_graph(
        const OmniVoiceLmInput& conditional,
        const OmniVoiceLmInput& unconditional,
        bool read_host_logits,
        bool expose_hidden,
        float guidance_scale);
    Status run_diffusion_graph(
        const OmniVoiceLmInput& input,
        const OmniVoiceLmInput& unconditional,
        OmniVoiceLmPackedOutput& output);
    void release_diffusion_graph() noexcept;
    ggml_backend_t backend() const noexcept;

private:
    class DiffusionGraph;

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
    std::unique_ptr<DiffusionGraph> diffusion_graph_;
    bool loaded_ = false;
};

} // namespace omnivoice
