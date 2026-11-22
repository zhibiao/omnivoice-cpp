#pragma once

#include "status.h"

#include <string>
#include <vector>

namespace omnivoice {

// Matches omnivoice.utils.text.chunk_text_punctuation.
Status chunk_text_punctuation(
    const std::string& text,
    int32_t chunk_length,
    int32_t minimum_chunk_length,
    std::vector<std::string>& chunks);

int32_t utf8_codepoint_count(const std::string& text);

Status plan_text_chunks(
    const std::string& text,
    int32_t estimated_frames,
    int32_t frame_rate,
    float chunk_duration,
    float chunk_threshold,
    std::vector<std::string>& chunks);

} // namespace omnivoice
