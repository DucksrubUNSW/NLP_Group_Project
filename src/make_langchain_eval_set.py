#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from data_loader import load_combined


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "langchain_eval_cases.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-label",
        type=int,
        default=10,
        help="Number of examples to sample for each label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_combined()

    samples = []
    for label in ["false", "mixed", "true"]:
        label_df = df[df["label"] == label]
        if len(label_df) < args.per_label:
            raise ValueError(
                f"Requested {args.per_label} examples for {label}, but only found {len(label_df)}."
            )
        samples.append(label_df.sample(n=args.per_label, random_state=args.seed))

    sample = (
        pd.concat(samples, ignore_index=True)[["text", "label"]]
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)

    print(f"Saved {len(sample)} examples to {args.output}")
    print(sample["label"].value_counts())


if __name__ == "__main__":
    main()
