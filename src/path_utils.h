#pragma once

#include <filesystem>
#include <string>

namespace omnivoice::detail {

inline std::filesystem::path path_from_utf8(const std::string& value) {
    return std::filesystem::u8path(value);
}

} // namespace omnivoice::detail
