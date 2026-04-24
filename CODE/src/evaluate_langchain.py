#!/usr/bin/env python3

# compares base transformer predictions vs LangChain-augmented predictions on a held-out eval set
# supports both standard retrieval and WordNet-expanded retrieval side by side
# usage: python evaluate_langchain.py --model roberta --input path/to/eval.csv

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cli_test import analyze_headline_three_class, combine_predictions, predict_bert, predict_roberta


DEFAULT_INPUT = Path(__file__).resolve().parent.parent.parent / "MISC" / "data" / "langchain_eval_cases.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "MISC" / "test_results" / "langchain_results.txt"
LABEL_ORDER = ["false", "mixed", "true"]


# thin wrapper so evaluate() can call predict without knowing which model is selected
def predict(model_name: str, text: str):
    if model_name == "bert":
        return predict_bert(text)
    return predict_roberta(text)


# parses CLI arguments for model selection, input path, row limit, and output path
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
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Text file for per-example predictions.",
    )
    return parser.parse_args()


# runs all three pipelines (base, LangChain, LangChain+WordNet) on every row in the dataframe
def evaluate(df: pd.DataFrame, model_name: str):
    y_true = []
    y_base = []
    y_langchain = []
    y_wordnet = []
    overrides = Counter()
    wordnet_overrides = Counter()
    corrected = 0
    worsened = 0
    unchanged_correct = 0
    unchanged_wrong = 0
    wordnet_corrected = 0
    wordnet_worsened = 0
    wordnet_unchanged_correct = 0
    wordnet_unchanged_wrong = 0
    rows = []

    for row in df.itertuples(index=False):
        base_label, base_confidence = predict(model_name, row.text)
        evidence_report = analyze_headline_three_class(
            row.text,
            base_label,
            base_confidence,
            use_wordnet=False,
        )
        wordnet_report = analyze_headline_three_class(
            row.text,
            base_label,
            base_confidence,
            use_wordnet=True,
        )
        final_label = combine_predictions(base_label, base_confidence, evidence_report)
        wordnet_label = combine_predictions(base_label, base_confidence, wordnet_report)

        y_true.append(row.label)
        y_base.append(base_label)
        y_langchain.append(final_label)
        y_wordnet.append(wordnet_label)
        rows.append(
            {
                "text": row.text,
                "gold": row.label,
                "base_label": base_label,
                "base_confidence": base_confidence,
                "langchain_label": final_label,
                "langchain_evidence": evidence_report.verdict,
                "langchain_confidence": evidence_report.confidence_score,
                "wordnet_label": wordnet_label,
                "wordnet_evidence": wordnet_report.verdict,
                "wordnet_confidence": wordnet_report.confidence_score,
            }
        )

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

        if wordnet_label != base_label:
            wordnet_overrides[(base_label, wordnet_label)] += 1
            if base_label != row.label and wordnet_label == row.label:
                wordnet_corrected += 1
            elif base_label == row.label and wordnet_label != row.label:
                wordnet_worsened += 1
        elif wordnet_label == row.label:
            wordnet_unchanged_correct += 1
        else:
            wordnet_unchanged_wrong += 1

    return (
        y_true,
        y_base,
        y_langchain,
        y_wordnet,
        overrides,
        wordnet_overrides,
        corrected,
        worsened,
        unchanged_correct,
        unchanged_wrong,
        wordnet_corrected,
        wordnet_worsened,
        wordnet_unchanged_correct,
        wordnet_unchanged_wrong,
        rows,
    )


# prints accuracy, classification report, and confusion matrix with a section header
def print_metrics(title: str, y_true: list[str], y_pred: list[str]):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=LABEL_ORDER, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=LABEL_ORDER))


# writes a per-example breakdown (base + LangChain + WordNet predictions) to a text file
def write_prediction_report(
    output_path: Path,
    model_name: str,
    rows: list[dict],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Examples evaluated: {len(rows)}\n")
        file.write("=" * 80 + "\n\n")

        for index, row in enumerate(rows, start=1):
            base_confidence = row["base_confidence"]
            base_confidence_text = (
                f"{base_confidence:.4f}" if base_confidence is not None else "N/A"
            )
            file.write(f"{index}. Headline: {row['text']}\n")
            file.write(f"Gold Label: {row['gold']}\n")
            file.write(
                f"{model_name.upper()} Prediction: {row['base_label']} "
                f"(confidence: {base_confidence_text})\n"
            )
            file.write(
                "LangChain Prediction: "
                f"{row['langchain_label']} "
                f"(evidence verdict: {row['langchain_evidence']}, "
                f"confidence: {row['langchain_confidence']:.4f})\n"
            )
            file.write(
                "LangChain + WordNet Prediction: "
                f"{row['wordnet_label']} "
                f"(evidence verdict: {row['wordnet_evidence']}, "
                f"confidence: {row['wordnet_confidence']:.4f})\n"
            )
            file.write("-" * 80 + "\n")


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
        y_langchain,
        y_wordnet,
        overrides,
        wordnet_overrides,
        corrected,
        worsened,
        unchanged_correct,
        unchanged_wrong,
        wordnet_corrected,
        wordnet_worsened,
        wordnet_unchanged_correct,
        wordnet_unchanged_wrong,
        rows,
    ) = evaluate(df, args.model)

    print(f"Model: {args.model}")
    print(f"Examples evaluated: {len(df)}")
    print_metrics(f"{args.model.upper()} ONLY", y_true, y_base)
    print_metrics(f"{args.model.upper()} + LANGCHAIN", y_true, y_langchain)
    print_metrics(f"{args.model.upper()} + LANGCHAIN + WORDNET", y_true, y_wordnet)

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

    print("\nWordNet Overrides:")
    if not wordnet_overrides:
        print("None")
    else:
        for (old_label, new_label), count in sorted(wordnet_overrides.items()):
            print(f"{old_label} -> {new_label}: {count}")

    print("\nWordNet Override Impact:")
    print(f"Corrected by LangChain + WordNet: {wordnet_corrected}")
    print(f"Made worse by LangChain + WordNet: {wordnet_worsened}")
    print(f"Unchanged correct: {wordnet_unchanged_correct}")
    print(f"Unchanged wrong: {wordnet_unchanged_wrong}")

    write_prediction_report(args.output, args.model, rows)
    print(f"\nPer-example predictions written to: {args.output}")


if __name__ == "__main__":
    main()
