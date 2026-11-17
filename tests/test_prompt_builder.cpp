#include "config.h"
#include "gguf_model.h"
#include "prompt_builder.h"
#include "text_tokenizer.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

bool expect_equal(
    const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected,
    const char* label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << label << " mismatch\nexpected:";
    for (int32_t value : expected) {
        std::cerr << ' ' << value;
    }
    std::cerr << "\nactual:";
    for (int32_t value : actual) {
        std::cerr << ' ' << value;
    }
    std::cerr << '\n';
    return false;
}

omnivoice::AudioCodes make_reference_codes(
    int32_t codebooks,
    int32_t frames) {
    omnivoice::AudioCodes codes;
    codes.codebooks = codebooks;
    codes.frames = frames;
    codes.data.resize(codes.size());
    for (int32_t codebook = 0; codebook < codebooks; ++codebook) {
        for (int32_t frame = 0; frame < frames; ++frame) {
            codes.data[
                static_cast<size_t>(codebook) * frames +
                static_cast<size_t>(frame)] =
                100 * codebook + frame + 1;
        }
    }
    return codes;
}

} // namespace

int main() {
    const std::string data_dir = OMNIVOICE_TEST_DATA_DIR;
    std::unique_ptr<omnivoice::GGUFModel> lm_gguf;
    omnivoice::Status status = omnivoice::GGUFModel::open(
        data_dir + "/omnivoice-lm-config-m3.gguf",
        lm_gguf);
    if (!status) {
        std::cerr << "LM GGUF load failed: " << status.message() << '\n';
        return 1;
    }
    std::unique_ptr<omnivoice::GGUFModel> audio_gguf;
    status = omnivoice::GGUFModel::open(
        data_dir + "/omnivoice-audio-config-m3.gguf",
        audio_gguf);
    if (!status) {
        std::cerr << "audio GGUF load failed: " << status.message() << '\n';
        return 1;
    }

    omnivoice::OmniVoiceConfig config;
    status = omnivoice::load_omnivoice_config(
        *lm_gguf,
        *audio_gguf,
        config);
    if (!status) {
        std::cerr << "config load failed: " << status.message() << '\n';
        return 1;
    }
    omnivoice::TextTokenizer tokenizer;
    status = tokenizer.load(config.text_tokenizer);
    if (!status) {
        std::cerr << "tokenizer load failed: " << status.message() << '\n';
        return 1;
    }

    std::string combined;
    status = omnivoice::combine_reference_target_text(
        "  Hello.\n",
        "World.\t ",
        combined);
    if (!status || combined != "Hello. World.") {
        std::cerr << "English text combination failed: " << combined << '\n';
        return 1;
    }
    status = omnivoice::combine_reference_target_text(
        " 你好 ",
        "世界 ",
        combined);
    if (!status || combined != "你好世界") {
        std::cerr << "CJK text combination failed\n";
        return 1;
    }
    status = omnivoice::combine_reference_target_text(
        "a\r\nb",
        "（c）",
        combined);
    if (!status || combined != "ab (c)") {
        std::cerr << "newline or parentheses normalization failed\n";
        return 1;
    }
    status = omnivoice::combine_reference_target_text(
        "你好，世界！",
        "Hello, 中英混合。",
        combined);
    if (!status ||
        combined != "你好，世界！ Hello,中英混合。") {
        std::cerr << "mixed Chinese-English text failed: "
                  << combined << '\n';
        return 1;
    }

    omnivoice::VoiceClonePromptInput input;
    input.reference_text = "Hello.";
    input.target_text = "World.";
    input.language = "en";
    input.reference_codes =
        make_reference_codes(config.lm.num_audio_codebooks, 3);
    input.target_frames = 4;
    input.denoise = true;

    omnivoice::PromptPlan plan;
    status = omnivoice::build_voice_clone_prompt(
        tokenizer,
        config,
        input,
        plan);
    if (!status) {
        std::cerr << "prompt build failed: " << status.message() << '\n';
        return 1;
    }

    if (plan.codebooks != 8 ||
        plan.layout.style_frames <= 0 ||
        plan.layout.text_frames <= 0 ||
        plan.layout.reference_start !=
            plan.layout.style_frames + plan.layout.text_frames ||
        plan.layout.target_start !=
            plan.layout.reference_start + input.reference_codes.frames ||
        plan.layout.sequence_length !=
            plan.layout.target_start + input.target_frames) {
        std::cerr << "prompt layout mismatch\n";
        return 1;
    }
    if (plan.style_tokens.empty() ||
        plan.style_tokens[0] != 151669 ||
        plan.style_tokens.back() != 151673 ||
        plan.text_tokens.front() != 151674 ||
        plan.text_tokens.back() != 151675) {
        std::cerr << "special token layout mismatch\n";
        return 1;
    }

    const size_t sequence_length =
        static_cast<size_t>(plan.layout.sequence_length);
    if (plan.input_ids.size() !=
            static_cast<size_t>(plan.codebooks) * sequence_length ||
        plan.audio_mask.size() != sequence_length) {
        std::cerr << "prompt tensor size mismatch\n";
        return 1;
    }
    for (int32_t codebook = 0; codebook < plan.codebooks; ++codebook) {
        const size_t row =
            static_cast<size_t>(codebook) * sequence_length;
        for (int32_t index = 0;
             index < plan.layout.style_frames;
             ++index) {
            if (plan.input_ids[row + static_cast<size_t>(index)] !=
                plan.style_tokens[static_cast<size_t>(index)]) {
                std::cerr << "style token rows are inconsistent\n";
                return 1;
            }
        }
        for (int32_t index = 0;
             index < plan.layout.text_frames;
             ++index) {
            if (plan.input_ids[
                    row + static_cast<size_t>(
                        plan.layout.style_frames + index)] !=
                plan.text_tokens[static_cast<size_t>(index)]) {
                std::cerr << "text token rows are inconsistent\n";
                return 1;
            }
        }
        for (int32_t frame = 0;
             frame < input.reference_codes.frames;
             ++frame) {
            const int32_t expected = input.reference_codes.data[
                static_cast<size_t>(codebook) * input.reference_codes.frames +
                static_cast<size_t>(frame)];
            const int32_t actual = plan.input_ids[
                row + static_cast<size_t>(
                    plan.layout.reference_start + frame)];
            if (actual != expected) {
                std::cerr << "reference code placement mismatch\n";
                return 1;
            }
        }
        for (int32_t frame = 0; frame < input.target_frames; ++frame) {
            const int32_t actual = plan.input_ids[
                row + static_cast<size_t>(
                    plan.layout.target_start + frame)];
            if (actual != config.lm.audio_mask_id) {
                std::cerr << "target MASK placement mismatch\n";
                return 1;
            }
        }
    }
    for (int32_t index = 0; index < plan.layout.sequence_length; ++index) {
        const uint8_t expected =
            index >= plan.layout.reference_start ? 1 : 0;
        if (plan.audio_mask[static_cast<size_t>(index)] != expected) {
            std::cerr << "audio mask mismatch\n";
            return 1;
        }
    }

    input.target_text = "[laughter] hi";
    input.language = "None";
    status = omnivoice::build_voice_clone_prompt(
        tokenizer,
        config,
        input,
        plan);
    if (!status) {
        std::cerr << "non-verbal prompt build failed\n";
        return 1;
    }
    std::vector<int32_t> expected_text;
    std::vector<int32_t> ids;
    tokenizer.encode("<|text_start|>", ids);
    expected_text.insert(expected_text.end(), ids.begin(), ids.end());
    tokenizer.encode("Hello. ", ids);
    expected_text.insert(expected_text.end(), ids.begin(), ids.end());
    tokenizer.encode("[laughter]", ids);
    expected_text.insert(expected_text.end(), ids.begin(), ids.end());
    tokenizer.encode(" hi", ids);
    expected_text.insert(expected_text.end(), ids.begin(), ids.end());
    tokenizer.encode("<|text_end|>", ids);
    expected_text.insert(expected_text.end(), ids.begin(), ids.end());
    if (!expect_equal(plan.text_tokens, expected_text, "non-verbal text")) {
        return 1;
    }

    input.reference_text = "你好，世界！";
    input.target_text = "Hello, 中英混合。";
    status = omnivoice::build_voice_clone_prompt(
        tokenizer,
        config,
        input,
        plan);
    if (!status) {
        std::cerr << "mixed Chinese-English prompt build failed\n";
        return 1;
    }
    expected_text.clear();
    tokenizer.encode(
        "<|text_start|>你好，世界！ Hello,中英混合。<|text_end|>",
        expected_text);
    if (!expect_equal(
            plan.text_tokens,
            expected_text,
            "mixed Chinese-English text")) {
        return 1;
    }

    input.reference_text.clear();
    if (omnivoice::build_voice_clone_prompt(
            tokenizer,
            config,
            input,
            plan)) {
        std::cerr << "empty reference text must be rejected\n";
        return 1;
    }
    input.reference_text = "Hello.";
    input.target_frames = 0;
    if (omnivoice::build_voice_clone_prompt(
            tokenizer,
            config,
            input,
            plan)) {
        std::cerr << "zero target frames must be rejected\n";
        return 1;
    }
    input.target_frames = 4;
    input.reference_codes.codebooks = 7;
    if (omnivoice::build_voice_clone_prompt(
            tokenizer,
            config,
            input,
            plan)) {
        std::cerr << "wrong codebook count must be rejected\n";
        return 1;
    }
    input.reference_codes =
        make_reference_codes(config.lm.num_audio_codebooks, 3);
    input.reference_codes.data[0] = config.lm.audio_mask_id;
    if (omnivoice::build_voice_clone_prompt(
            tokenizer,
            config,
            input,
            plan)) {
        std::cerr << "MASK in reference codes must be rejected\n";
        return 1;
    }

    return 0;
}
