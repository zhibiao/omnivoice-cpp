#pragma once

#include "config.h"
#include "status.h"
#include "types.h"

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace omnivoice {

class GGUFModel;

struct HiggsDecoderStage {
    std::string name;
    int32_t channels = 0;
    int32_t frames = 0;
    std::vector<float> values;  // [channels, frames]
};

class HiggsAudioDecoder {
public:
    HiggsAudioDecoder();
    ~HiggsAudioDecoder();

    HiggsAudioDecoder(const HiggsAudioDecoder&) = delete;
    HiggsAudioDecoder& operator=(const HiggsAudioDecoder&) = delete;

    // The backend is borrowed and must outlive this decoder.
    Status load(
        const HiggsTokenizerConfig& config,
        const GGUFModel& model,
        ggml_backend_t backend);

    Status decode(
        const AudioCodes& codes,
        AudioBuffer& audio,
        std::vector<HiggsDecoderStage>* debug_stages = nullptr) const;

private:
    class DecodeGraph;

    // Higgs decoder graph:
    //   higgs_codes.*
    //     -> higgs_decoder_output
    ggml_cgraph* build_graph(
        ggml_context* context,
        int32_t frames,
        bool record_debug_stages) const;

    ggml_tensor* weight(const std::string& name) const;
    void release() noexcept;

    HiggsTokenizerConfig config_;
    ggml_backend_t backend_ = nullptr;
    ggml_context* weight_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> weights_;
    mutable std::unique_ptr<DecodeGraph> decode_graph_;
    bool use_matrix_transposed_conv_ = false;
    bool loaded_ = false;
};

} // namespace omnivoice
