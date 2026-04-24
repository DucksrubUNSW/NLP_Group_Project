#!/usr/bin/env python3

# samples a balanced eval set from the held-out test split for LangChain evaluation
# usage: python make_langchain_eval_set.py --per-label 10

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_combined


DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "MISC" / "data" / "langchain_eval_cases.csv"
TEST_SPLIT_RANDOM_STATE = 42


# parses CLI arguments for sample size, output path, and random seed
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

    # Recreate the same held-out 20% test split used by bert_model.py so the
    # LangChain evaluation samples only from data the transformers were not
    # trained or validated on.
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=TEST_SPLIT_RANDOM_STATE,
        stratify=df["label"],
    )

    samples = []
    for label in ["false", "mixed", "true"]:
        label_df = test_df[test_df["label"] == label]
        if len(label_df) < args.per_label:
            raise ValueError(
                f"Requested {args.per_label} examples for {label}, but only found {len(label_df)} in the held-out test split."
            )
        samples.append(label_df.sample(n=args.per_label, random_state=args.seed))

    sample = (
        pd.concat(samples, ignore_index=True)[["text", "label"]]
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)

    print(f"Saved {len(sample)} held-out test examples to {args.output}")
    print(sample["label"].value_counts())


if __name__ == "__main__":
    main()
