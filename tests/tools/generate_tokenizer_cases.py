#!/usr/bin/env python3
"""Generate Qwen tokenizer golden token IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    texts = [
        "Hello, world.",
        "今天，世界！",
        "Hello, 世界!",
        "你好，世界！Hello, world.",
        "One, two; three: four?",
        "e\u0301",
        "A\u030a",
        "\uac00",
        "\u304c",
        "<|text_start|>Hello<|text_end|>",
    ]
    cases = []
    for text in texts:
        cases.append({
            "text": text,
            "ids": tokenizer(
                text,
                add_special_tokens=False,
            ).input_ids,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote tokenizer cases: {args.output}")


if __name__ == "__main__":
    main()
