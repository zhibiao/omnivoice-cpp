#!/usr/bin/env python3
"""Generate a Higgs audio encoder reference fixture."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import HiggsAudioV2TokenizerModel


def time_major_to_channels_major(
    values: torch.Tensor,
) -> torch.Tensor:
    return values[0].transpose(0, 1).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = HiggsAudioV2TokenizerModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
    ).eval()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    sample_rate = 24000
    time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    waveform = (
        0.08 * torch.sin(2.0 * torch.pi * 220.0 * time)
        + 0.03 * torch.sin(2.0 * torch.pi * 437.0 * time)
        + 0.015 * torch.sin(2.0 * torch.pi * 881.0 * time)
    )
    input_values = waveform.unsqueeze(0).unsqueeze(0)

    resampled = torchaudio.functional.resample(
        waveform.unsqueeze(0),
        sample_rate,
        model.config.semantic_sample_rate,
    ).squeeze(0)
    semantic_input = F.pad(
        resampled.unsqueeze(0),
        (160, 160),
    )
    with torch.no_grad():
        feature_extracted = model.semantic_model.feature_extractor(
            semantic_input
        ).transpose(1, 2)
        projected = model.semantic_model.feature_projection(
            feature_extracted
        )
        normalized_features = model.semantic_model.feature_projection.layer_norm(
            feature_extracted
        )
        positional = model.semantic_model.encoder.pos_conv_embed(
            projected
        )
        positional_weight = (
            model.semantic_model.encoder.pos_conv_embed.conv.weight
        )
        pos_group0 = F.conv1d(
            projected.transpose(1, 2)[:, :48],
            positional_weight[:48],
            padding=64,
        )[:, :, : projected.shape[1]]
        semantic_outputs = model.semantic_model(
            semantic_input,
            output_hidden_states=True,
        )
        hidden_states = semantic_outputs.hidden_states
        hubert_mean = torch.stack(hidden_states, dim=1).mean(dim=1)
        semantic = model.encoder_semantic(
            hubert_mean[:, ::2, :].transpose(1, 2)
        )

        acoustic_output_length = model._get_conv1d_output_lengths(
            input_values.shape[2],
            model.acoustic_encoder,
        )
        if acoustic_output_length != semantic.shape[2]:
            acoustic_input = F.pad(input_values, (480, 480))
        else:
            acoustic_input = input_values
        acoustic_hidden = model.acoustic_encoder.conv1(acoustic_input)
        acoustic_stages = [
            ("ac_conv1", acoustic_hidden[0].contiguous()),
        ]
        for index, block in enumerate(model.acoustic_encoder.block):
            acoustic_hidden = block(acoustic_hidden)
            acoustic_stages.append((
                f"ac_block{index}",
                acoustic_hidden[0].contiguous(),
            ))
        acoustic_hidden = model.acoustic_encoder.snake1(
            acoustic_hidden
        )
        acoustic = model.acoustic_encoder.conv2(acoustic_hidden)
        concatenated = torch.cat([acoustic, semantic], dim=1)
        embeddings = model.fc(
            concatenated.transpose(1, 2)
        ).transpose(1, 2)
        audio_codes = model.quantizer.encode(
            embeddings,
            model.config.target_bandwidths[-1],
        ).squeeze(1)

    stages: list[tuple[str, torch.Tensor]] = [
        ("resampled", resampled.unsqueeze(0)),
        ("semantic_input", semantic_input[0].unsqueeze(0)),
        (
            "feature_conv0",
            torch.nn.functional.conv1d(
                semantic_input[:, None],
                model.semantic_model.feature_extractor
                .conv_layers[0].conv.weight,
                stride=5,
            )[0].contiguous(),
        ),
        (
            "feature_extracted",
            time_major_to_channels_major(feature_extracted),
        ),
        (
            "normalized_features",
            time_major_to_channels_major(normalized_features),
        ),
        (
            "projected",
            time_major_to_channels_major(projected),
        ),
        ("pos_group0", pos_group0[0].contiguous()),
        (
            "positional",
            time_major_to_channels_major(positional),
        ),
    ]
    stages.extend(
        (f"h{index}", time_major_to_channels_major(hidden))
        for index, hidden in enumerate(hidden_states)
    )
    stages.extend([
        (
            "hubert_mean",
            time_major_to_channels_major(hubert_mean),
        ),
        ("semantic", semantic[0].contiguous()),
        *acoustic_stages,
        ("acoustic", acoustic[0].contiguous()),
        ("concat", concatenated[0].contiguous()),
        (
            "embeddings",
            embeddings[0].contiguous(),
        ),
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        file.write(
            struct.pack(
                "<8s4i",
                b"HIGENC01",
                int(waveform.numel()),
                int(audio_codes.shape[0]),
                int(audio_codes.shape[1]),
                len(stages),
            )
        )
        waveform.numpy().astype(np.float32).tofile(file)
        audio_codes.numpy().astype(np.int32).tofile(file)
        for name, values in stages:
            array = values.detach().cpu().numpy()
            file.write(
                struct.pack(
                    "<32sii",
                    name.encode("ascii"),
                    int(array.shape[0]),
                    int(array.shape[1]),
                )
            )
            array.astype(np.float32).tofile(file)

    print(f"wrote Higgs encoder fixture: {args.output}")
    print(f"codes shape: {tuple(audio_codes.shape)}")
    print(f"stages: {len(stages)}")


if __name__ == "__main__":
    main()
