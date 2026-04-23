#!/usr/bin/env python3

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cli_test import analyze_headline_three_class, combine_predictions, predict_bert, predict_roberta


DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "MISC" / "data" / "langchain_eval_cases.csv"
LABEL_ORDER = ["false", "mixed", "true"]


def predict(model_name: str, text: str):
    if model_name == "bert":
        return predict_bert(text)
    return predict_roberta(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV with text,label columns.",
    )
    parser.add_argument(
        "--model",
        choices=["bert", "roberta"],
        default="roberta",
        help="Base transformer model to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to evaluate for a quick test.",
    )
    return parser.parse_args()


def evaluate(df: pd.DataFrame, model_name: str):
    y_true = []
    y_base = []
    y_augmented = []
    overrides = Counter()
    corrected = 0
    worsened = 0
    unchanged_correct = 0
    unchanged_wrong = 0

    for row in df.itertuples(index=False):
        base_label, base_confidence = predict(model_name, row.text)
        evidence_report = analyze_headline_three_class(row.text, base_label, base_confidence)
        final_label = combine_predictions(base_label, base_confidence, evidence_report)

        y_true.append(row.label)
        y_base.append(base_label)
        y_augmented.append(final_label)

        if final_label != base_label:
            overrides[(base_label, final_label)] += 1
            if base_label != row.label and final_label == row.label:
                corrected += 1
            elif base_label == row.label and final_label != row.label:
                worsened += 1
        elif final_label == row.label:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    return y_true, y_base, y_augmented, overrides, corrected, worsened, unchanged_correct, unchanged_wrong


def print_metrics(title: str, y_true: list[str], y_pred: list[str]):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=LABEL_ORDER))


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing evaluation file: {args.input}. Create a CSV with text,label columns."
        )

    df = pd.read_csv(args.input)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Input CSV must contain text and label columns.")

    if args.limit is not None:
        df = df.head(args.limit)

    (
        y_true,
        y_base,
        y_augmented,
        overrides,
        corrected,
        worsened,
        unchanged_correct,
        unchanged_wrong,
    ) = evaluate(df, args.model)

    print(f"Model: {args.model}")
    print(f"Examples evaluated: {len(df)}")
    print_metrics(f"{args.model.upper()} ONLY", y_true, y_base)
    print_metrics(f"{args.model.upper()} + LANGCHAIN", y_true, y_augmented)

    print("\nOverrides:")
    if not overrides:
        print("None")
    else:
        for (old_label, new_label), count in sorted(overrides.items()):
            print(f"{old_label} -> {new_label}: {count}")

    print("\nOverride Impact:")
    print(f"Corrected by LangChain: {corrected}")
    print(f"Made worse by LangChain: {worsened}")
    print(f"Unchanged correct: {unchanged_correct}")
    print(f"Unchanged wrong: {unchanged_wrong}")


if __name__ == "__main__":
    main()
