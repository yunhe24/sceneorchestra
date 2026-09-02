#!/usr/bin/env python3
"""Compatibility entry point for the development instruction generator."""

from __future__ import annotations

import argparse

from sceneorchestra.instructions import generate_instruction_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate indoor-scene instruction candidates")
    parser.add_argument("-n", "--num", type=int, required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--existing")
    args = parser.parse_args()
    count = generate_instruction_candidates(
        args.output,
        model=args.model,
        count=args.num,
        existing_path=args.existing,
    )
    print(f"instruction candidates: {count}")


if __name__ == "__main__":
    main()
