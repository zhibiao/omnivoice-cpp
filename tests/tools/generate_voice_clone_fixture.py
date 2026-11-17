#!/usr/bin/env python3
"""Generate a deterministic end-to-end Voice Clone reference fixture."""

from __future__ import annotations

import argparse
import math
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# The reference package imports pydub at module load even though this fixture
# only uses direct tensor/model APIs and never invokes its silence utilities.
if "pydub" not in sys.modules:
    pydub = types.ModuleType("pydub")
    silence = types.ModuleType("pydub.silence")
    pydub.AudioSegment = type("AudioSegment", (), {})
    silence.detect_leading_silence = lambda *args, **kwargs: 0
    silence.detect_nonsilent = lambda *args, **kwargs: []
    silence.split_on_silence = lambda *args, **kwargs: []
    pydub.silence = silence
    sys.modules["pydub"] = pydub
    sys.modules["pydub.silence"] = silence

from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceGenerationConfig,
    _get_time_steps,
)
from omnivoice.utils.duration import RuleDurationEstimator
from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = OmniVoice.from_pretrained(
        str(args.model_dir),
        train=True,
        local_files_only=True,
    ).eval()
    model.text_tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
    )
    model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        str(args.model_dir / "audio_tokenizer"),
        local_files_only=True,
    ).eval()
    model.duration_estimator = RuleDurationEstimator()
    model.sampling_rate = 24000
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    codebooks = model.config.num_audio_codebook
    reference_frames = 25
    target_frames = 25
    reference_codes = torch.empty(
        codebooks,
        reference_frames,
        dtype=torch.long,
    )
    for codebook in range(codebooks):
        for frame in range(reference_frames):
            reference_codes[codebook, frame] = (
                codebook * 97 + frame * 31 + 7
            ) % 1024

    reference_text = "Hello."
    target_text = "Hi."
    inputs = model._prepare_inference_inputs(
        target_text,
        target_frames,
        reference_text,
        reference_codes,
        "en",
        None,
        True,
    )
    input_ids = inputs["input_ids"]
    audio_mask = inputs["audio_mask"]
    with torch.no_grad():
        embeddings = model._prepare_embed_inputs(
            input_ids,
            audio_mask,
        )
        attention_mask = torch.ones(
            1,
            1,
            input_ids.shape[2],
            input_ids.shape[2],
            dtype=torch.bool,
        )
        hidden = model.llm(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        logits = model.audio_heads(hidden).view(
            1,
            hidden.shape[1],
            codebooks,
            model.config.audio_vocab_size,
        ).permute(0, 2, 1, 3).to(torch.float32)
        target_logits = logits[:, :, -target_frames:, :]

        config = OmniVoiceGenerationConfig(
            num_step=4,
            guidance_scale=2.0,
            t_shift=0.1,
            layer_penalty_factor=5.0,
            position_temperature=0.0,
            class_temperature=0.0,
        )

        def forward_logits(
            ids: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            sequence = ids.shape[2]
            embeds = model._prepare_embed_inputs(ids, mask)
            mask_4d = torch.ones(
                1,
                1,
                sequence,
                sequence,
                dtype=torch.bool,
            )
            current_hidden = model.llm(
                inputs_embeds=embeds,
                attention_mask=mask_4d,
                return_dict=True,
            ).last_hidden_state
            current_logits = model.audio_heads(current_hidden).view(
                1,
                sequence,
                codebooks,
                model.config.audio_vocab_size,
            ).permute(0, 2, 1, 3).to(torch.float32)
            return current_logits

        conditional_ids = input_ids.clone()
        conditional_mask = audio_mask.clone()
        unconditional_ids = conditional_ids[..., -target_frames:].clone()
        unconditional_mask = conditional_mask[..., -target_frames:].clone()
        current = torch.full(
            (1, codebooks, target_frames),
            model.config.audio_mask_id,
            dtype=torch.long,
        )
        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=config.num_step,
            t_shift=config.t_shift,
        ).tolist()
        total = codebooks * target_frames
        remaining = total
        schedule = []
        for step in range(config.num_step):
            count = (
                remaining
                if step == config.num_step - 1
                else min(
                    math.ceil(
                        total * (timesteps[step + 1] - timesteps[step])
                    ),
                    remaining,
                )
            )
            schedule.append(int(count))
            remaining -= int(count)

        layer_ids = torch.arange(codebooks).view(1, -1, 1)
        for step in range(config.num_step):
            conditional_logits = forward_logits(
                conditional_ids,
                conditional_mask,
            )
            unconditional_logits = forward_logits(
                unconditional_ids,
                unconditional_mask,
            )
            predictions, scores = model._predict_tokens_with_scoring(
                conditional_logits[:, :, -target_frames:, :],
                unconditional_logits,
                config,
            )
            scores = scores - layer_ids * config.layer_penalty_factor
            scores.masked_fill_(
                current != model.config.audio_mask_id,
                -float("inf"),
            )
            top_k = torch.topk(
                scores.flatten(),
                schedule[step],
            ).indices
            flat_tokens = current.flatten().clone()
            flat_tokens[top_k] = predictions.flatten()[top_k]
            current = flat_tokens.view_as(current)
            conditional_ids[..., -target_frames:] = current
            unconditional_ids = current.clone()
        target_codes = current[0]

        decoded = model.audio_tokenizer.decode(
            target_codes.unsqueeze(0)
        ).audio_values[0, 0].cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        file.write(
            struct.pack(
                "<8s5i",
                b"HIGCLONE",
                codebooks,
                reference_frames,
                target_frames,
                input_ids.shape[2],
                decoded.numel(),
            )
        )
        reference_codes.numpy().astype(np.int32).tofile(file)
        target_codes.numpy().astype(np.int32).tofile(file)
        input_ids[0].numpy().astype(np.int32).tofile(file)
        audio_mask[0].numpy().astype(np.uint8).tofile(file)
        hidden[0].numpy().astype(np.float32).tofile(file)
        target_logits[0].numpy().astype(np.float32).tofile(file)
        decoded.numpy().astype(np.float32).tofile(file)

    print(f"wrote Voice Clone fixture: {args.output}")
    print(f"sequence length: {input_ids.shape[2]}")
    print(f"target codes shape: {tuple(target_codes.shape)}")
    print(f"waveform samples: {decoded.numel()}")


if __name__ == "__main__":
    main()
