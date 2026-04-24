#!/usr/bin/env python3

# evaluates all models on the test set and prints accuracy + macro f1
# usage: cd CODE/src && python evaluate_models.py

import os
import numpy as np
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_loader import load_combined

MODELS_DIR = os.path.join("..", "..", "MISC", "models")
LABEL_TO_ID = {"false": 0, "mixed": 1, "true": 2}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# wraps text + labels into a pytorch dataset for the dataloader
class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def evaluate_transformer(model, dataloader):
    # runs model on dataloader and returns predictions + true labels
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

# prints accuracy, precision, recall, and macro f1 for a set of predictions
def print_scores(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    print(f"  accuracy:  {acc:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall:    {rec:.4f}")
    print(f"  macro f1:  {f1:.4f}")

def main():
    print(f"device: {DEVICE}")
    print("loading data...")
    df = load_combined()
    X = df["text"].values
    y = np.array([LABEL_TO_ID[label] for label in df["label"].values])

    # same split as training (60/20/20)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # baseline
    print("\n--- baseline (TF-IDF + Logistic Regression) ---")
    baseline_dir = os.path.join(MODELS_DIR, "baseline")
    model = joblib.load(os.path.join(baseline_dir, "baseline_model.pkl"))
    vectoriser = joblib.load(os.path.join(baseline_dir, "baseline_vectoriser.pkl"))
    X_test_tfidf = vectoriser.transform(X_test)
    y_pred_baseline = model.predict(X_test_tfidf)
    # map string labels to ints for consistency
    label_to_int = {"false": 0, "mixed": 1, "true": 2}
    y_pred_baseline_int = np.array([label_to_int[l] for l in y_pred_baseline])
    print_scores("baseline", y_test, y_pred_baseline_int)

    # pre-trained BERT (no fine-tuning)
    print("\n--- pre-trained BERT (no fine-tuning) ---")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_pretrained = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    ).to(DEVICE)
    test_dataset = HeadlineDataset(X_test.tolist(), y_test.tolist(), bert_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    preds, labels = evaluate_transformer(bert_pretrained, test_loader)
    print_scores("pre-trained BERT", labels, preds)
    del bert_pretrained

    # fine-tuned BERT
    print("\n--- fine-tuned BERT ---")
    bert_model = BertForSequenceClassification.from_pretrained(
        os.path.join(MODELS_DIR, "bert_finetuned")
    ).to(DEVICE)
    preds, labels = evaluate_transformer(bert_model, test_loader)
    print_scores("fine-tuned BERT", labels, preds)
    del bert_model

    # fine-tuned RoBERTa
    print("\n--- fine-tuned RoBERTa ---")
    roberta_tokenizer = RobertaTokenizer.from_pretrained(
        os.path.join(MODELS_DIR, "roberta_finetuned")
    )
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        os.path.join(MODELS_DIR, "roberta_finetuned")
    ).to(DEVICE)
    test_dataset_r = HeadlineDataset(X_test.tolist(), y_test.tolist(), roberta_tokenizer)
    test_loader_r = DataLoader(test_dataset_r, batch_size=32)
    preds, labels = evaluate_transformer(roberta_model, test_loader_r)
    print_scores("fine-tuned RoBERTa", labels, preds)

    print("\ndone!")

if __name__ == "__main__":
    main()