#pragma once

#include "config.h"
#include "status.h"
#include "types.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace omnivoice {

class GGUFModel;

struct HiggsEncoderStage {
    std::string name;
    int32_t channels = 0;
    int32_t frames = 0;
    std::vector<float> values;  // [channels, frames]
};

class HiggsAudioEncoder {
public:
    HiggsAudioEncoder();
    ~HiggsAudioEncoder();

    HiggsAudioEncoder(const HiggsAudioEncoder&) = delete;
    HiggsAudioEncoder& operator=(const HiggsAudioEncoder&) = delete;

    // The backend and model are borrowed and must outlive this encoder.
    Status load(
        const HiggsTokenizerConfig& config,
        const GGUFModel& model,
        ggml_backend_t backend);

    Status encode(
        const AudioBuffer& audio,
        AudioCodes& codes,
        std::vector<HiggsEncoderStage>* debug_stages = nullptr) const;

    bool is_loaded() const noexcept;

private:
    // Higgs encoder graph:
    //   higgs_semantic_input / higgs_acoustic_input /
    //   higgs_attention_mask
    //     -> higgs_encoder_embeddings
    ggml_cgraph* build_graph(
        ggml_context* context,
        int64_t semantic_input_length,
        int64_t acoustic_input_length,
        bool record_debug_stages) const;
    Status run_graph(
        const AudioBuffer& audio,
        const std::vector<float>& resampled,
        int64_t semantic_input_length,
        int64_t acoustic_input_length,
        bool pad_acoustic,
        ggml_cgraph* graph,
        AudioCodes& codes,
        std::vector<HiggsEncoderStage>* debug_stages) const;

    ggml_tensor* weight(const std::string& name) const;
    void release() noexcept;

    HiggsTokenizerConfig config_;
    ggml_backend_t backend_ = nullptr;
    const GGUFModel* model_ = nullptr;
    ggml_context* weight_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> weights_;
    bool loaded_ = false;
};

} // namespace omnivoice
