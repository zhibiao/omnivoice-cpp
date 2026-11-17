#include "types.h"

#include <cmath>
#include <iostream>

int main() {
    omnivoice::AudioBuffer audio;
    audio.sample_rate = 24000;
    audio.channels = 1;
    audio.samples.resize(24000);

    if (std::abs(audio.duration_seconds() - 1.0) > 1e-9) {
        std::cerr << "unexpected AudioBuffer duration\n";
        return 1;
    }

    omnivoice::AudioCodes codes;
    codes.frames = 25;
    codes.codebooks = 8;
    codes.data.resize(codes.size());

    if (codes.data.size() != 200) {
        std::cerr << "unexpected AudioCodes size\n";
        return 1;
    }
    return 0;
}
