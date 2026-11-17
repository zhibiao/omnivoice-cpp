#include "voice_clone_prompt.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

omnivoice::VoiceClonePrompt make_prompt() {
    omnivoice::VoiceClonePrompt prompt;
    prompt.reference_text = "Reference text with 中文.";
    prompt.language = "zh";
    prompt.reference_rms = 0.125f;
    prompt.sample_rate = 24000;
    prompt.denoise = true;
    prompt.reference_codes.codebooks = 8;
    prompt.reference_codes.frames = 5;
    prompt.reference_codes.data.resize(40);
    for (size_t index = 0;
         index < prompt.reference_codes.data.size();
         ++index) {
        prompt.reference_codes.data[index] =
            static_cast<int32_t>((index * 97 + 11) % 1024);
    }
    return prompt;
}

bool equal_prompts(
    const omnivoice::VoiceClonePrompt& actual,
    const omnivoice::VoiceClonePrompt& expected) {
    return actual.reference_text == expected.reference_text &&
        actual.language == expected.language &&
        std::abs(actual.reference_rms - expected.reference_rms) < 1e-7f &&
        actual.sample_rate == expected.sample_rate &&
        actual.denoise == expected.denoise &&
        actual.reference_codes.codebooks ==
            expected.reference_codes.codebooks &&
        actual.reference_codes.frames ==
            expected.reference_codes.frames &&
        actual.reference_codes.data ==
            expected.reference_codes.data;
}

bool write_bytes(
    const std::filesystem::path& path,
    const std::string& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(
        bytes.data(),
        static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(output);
}

} // namespace

int main() {
    const std::filesystem::path data_dir = OMNIVOICE_TEST_DATA_DIR;
    const std::filesystem::path path =
        data_dir / "test_voice_clone_prompt.ovprompt";
    const std::filesystem::path invalid_path =
        data_dir / "test_voice_clone_prompt_invalid.ovprompt";
    try {
        const omnivoice::VoiceClonePrompt expected = make_prompt();
        omnivoice::Status status =
            omnivoice::save_voice_clone_prompt(path.string(), expected);
        if (!status) {
            std::cerr << "save failed: " << status.message() << '\n';
            return 1;
        }
        omnivoice::VoiceClonePrompt actual;
        status = omnivoice::load_voice_clone_prompt(
            path.string(),
            actual);
        if (!status || !equal_prompts(actual, expected)) {
            std::cerr << "voice prompt round-trip mismatch\n";
            return 1;
        }

        if (!write_bytes(invalid_path, "BADMAGIC")) {
            return 1;
        }
        if (omnivoice::load_voice_clone_prompt(
                invalid_path.string(),
                actual)) {
            std::cerr << "bad magic must be rejected\n";
            return 1;
        }

        omnivoice::VoiceClonePrompt invalid = expected;
        invalid.reference_codes.codebooks = 7;
        if (omnivoice::save_voice_clone_prompt(
                invalid_path.string(),
                invalid)) {
            std::cerr << "invalid codebook count must be rejected\n";
            return 1;
        }
        invalid = expected;
        invalid.reference_codes.data[0] = 1024;
        if (omnivoice::validate_voice_clone_prompt(invalid)) {
            std::cerr << "MASK/out-of-range code must be rejected\n";
            return 1;
        }
        invalid = expected;
        invalid.reference_text.clear();
        if (omnivoice::validate_voice_clone_prompt(invalid)) {
            std::cerr << "empty reference text must be rejected\n";
            return 1;
        }
        invalid = expected;
        invalid.sample_rate = 16000;
        if (omnivoice::validate_voice_clone_prompt(invalid)) {
            std::cerr << "non-24-kHz prompt must be rejected\n";
            return 1;
        }

        std::ifstream complete(path, std::ios::binary);
        std::string truncated(
            (std::istreambuf_iterator<char>(complete)),
            std::istreambuf_iterator<char>());
        truncated.resize(truncated.size() / 2);
        if (!write_bytes(invalid_path, truncated) ||
            omnivoice::load_voice_clone_prompt(
                invalid_path.string(),
                actual)) {
            std::cerr << "truncated prompt must be rejected\n";
            return 1;
        }

        complete.close();
        std::filesystem::remove(path);
        std::filesystem::remove(invalid_path);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "voice prompt test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
