#!/usr/bin/env python3
"""Generate punctuation chunking golden cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-module", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "omnivoice_text_reference",
        args.text_module,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load reference text module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunk_text_punctuation = module.chunk_text_punctuation
    cases = [
        ("Mr. Smith went home. He slept.", 24, 3),
        ("One. Two! Three? Four, five; six: seven.", 12, 3),
        ("你好。世界！今天很好。我们开始吧。", 8, 3),
        ("First sentence. Second sentence! Third sentence?", 36, 3),
        ('He said, "Go now." Then left.', 18, 3),
        ("A. B. C. D.", 4, 3),
        ("No Dr. Smith arrived at 5 p.m. today.", 20, 3),
        ("Nested [tag] text. More text.", 100, 3),
        ("  Short text.  ", 100, 3),
        ("你好，世界！Hello, world. 今天我们测试 OmniVoice！", 15, 3),
    ]
    output = []
    for text, chunk_len, minimum in cases:
        output.append({
            "text": text,
            "chunk_length": chunk_len,
            "minimum_chunk_length": minimum,
            "chunks": chunk_text_punctuation(
                text,
                chunk_len=chunk_len,
                min_chunk_len=minimum,
            ),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote text chunker golden: {args.output}")


if __name__ == "__main__":
    main()
