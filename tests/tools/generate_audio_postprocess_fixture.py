#!/usr/bin/env python3
"""Generate a reference cross-fade fixture."""

from __future__ import annotations

import argparse
import importlib.util
import struct
import sys
import types
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "pydub" not in sys.modules:
        pydub = types.ModuleType("pydub")
        silence = types.ModuleType("pydub.silence")
        pydub.AudioSegment = type("AudioSegment", (), {})
        silence.detect_leading_silence = lambda *a, **k: 0
        silence.detect_nonsilent = lambda *a, **k: []
        silence.split_on_silence = lambda *a, **k: []
        pydub.silence = silence
        sys.modules["pydub"] = pydub
        sys.modules["pydub.silence"] = silence

    spec = importlib.util.spec_from_file_location(
        "omnivoice_audio_reference",
        args.audio_module,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load reference audio module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    chunks = [
        np.ones((1, 100), dtype=np.float32),
        np.ones((1, 100), dtype=np.float32),
    ]
    merged = module.cross_fade_chunks(chunks, 24000, 0.001)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        file.write(struct.pack("<8sI", b"AUDCF001", merged.size))
        merged.astype(np.float32).tofile(file)
    print(f"wrote cross-fade fixture: {args.output}")
    print(f"samples: {merged.size}")


if __name__ == "__main__":
    main()
