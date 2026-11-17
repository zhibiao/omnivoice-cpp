#pragma once

#include "status.h"
#include "types.h"

#include <string>

namespace omnivoice {

Status write_wav(
    const std::string& path,
    const AudioBuffer& audio);

} // namespace omnivoice
