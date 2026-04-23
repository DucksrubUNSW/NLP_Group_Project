#!/usr/bin/env python3

# loads and combines the LIAR and FakeNewsNet (PolitiFact) datasets
# labels are mapped to three classes: true, mixed, false

import pandas as pd
from datasets import load_dataset

# LIAR has 6 fine-grained labels, we collapse them down to 3
LIAR_LABEL_MAP = {
    "true": "true",
    "mostly-true": "true",
    "half-true": "mixed",
    "barely-true": "mixed",
    "false": "false",
    "pants-fire": "false",
}

# FakeNewsNet is binary so no "mixed" class here
FAKE_NEWS_NET_LABEL_MAP = {
    "real": "true",
    "fake": "false",
}

def load_liar(data_dir="data/liar"):
    # loads LIAR from HuggingFace, uses local cache if already downloaded
    dataset = load_dataset("ucsbnlp/liar")

    frames = []
    for split in ["train", "validation", "test"]:
        df = dataset[split].to_pandas()
        df = df[["statement", "label"]].copy()
        df.columns = ["text", "label"]

        # label column is an int index, convert to string then map to our 3 classes
        label_names = dataset[split].features["label"].names
        df["label"] = df["label"].map(lambda x: label_names[x])
        df["label"] = df["label"].map(LIAR_LABEL_MAP)

        df["source"] = "liar"
        df["split"] = split
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def load_fakenewsnet(data_dir="../../MISC/data/fakenewsnet"):
    # loads the PolitiFact portion of FakeNewsNet from local csvs
    frames = []
    for label_type in ["real", "fake"]:
        filepath = f"{data_dir}/politifact_{label_type}.csv"
        df = pd.read_csv(filepath)

        # just grab the title as our text
        df = df[["title"]].copy()
        df.columns = ["text"]
        df["label"] = FAKE_NEWS_NET_LABEL_MAP[label_type]
        df["source"] = "fakenewsnet"
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def load_combined(data_dir_liar="../../MISC/data/liar", data_dir_fnn="../../MISC/data/fakenewsnet"):
    # loads both datasets and merges them into one dataframe
    liar_df = load_liar(data_dir_liar)
    fnn_df = load_fakenewsnet(data_dir_fnn)

    combined = pd.concat([liar_df, fnn_df], ignore_index=True)
    combined = combined.dropna(subset=["text"])

    print(f"LIAR samples:         {len(liar_df)}")
    print(f"FakeNewsNet samples:  {len(fnn_df)}")
    print(f"Combined total:       {len(combined)}")
    print(f"\nLabel distribution:\n{combined['label'].value_counts()}")

    return combined

if __name__ == "__main__":
    df = load_combined()
    print(f"\nSample rows:\n{df.head(10)}")