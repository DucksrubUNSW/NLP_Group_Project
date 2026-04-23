#!/usr/bin/env python3

# runs a headline through all 3 models and prints predictions + confidence

import os
import torch
import joblib
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

MODELS_DIR = os.path.join("..", "..", "MISC", "models")
LABELS = {0: "false", 1: "mixed", 2: "true"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# change this to whatever you want to test
HEADLINE = "The unemployment rate has dropped to its lowest point in 50 years"

def test_baseline(headline):
    baseline_dir = os.path.join(MODELS_DIR, "baseline")
    model = joblib.load(os.path.join(baseline_dir, "baseline_model.pkl"))
    vectoriser = joblib.load(os.path.join(baseline_dir, "baseline_vectoriser.pkl"))
    tfidf = vectoriser.transform([headline])
    # get probability scores for each class
    probs = model.predict_proba(tfidf)[0]
    pred_idx = probs.argmax()
    pred_label = model.classes_[pred_idx]
    print(f"  prediction:  {pred_label}")
    print(f"  confidence:  {probs[pred_idx]:.4f}")
    print(f"  all scores:  false={probs[list(model.classes_).index('false')]:.4f}, "
          f"mixed={probs[list(model.classes_).index('mixed')]:.4f}, "
          f"true={probs[list(model.classes_).index('true')]:.4f}")

def test_transformer(headline, model_name, model_dir, tokenizer_class, model_class):
    tokenizer = tokenizer_class.from_pretrained(os.path.join(MODELS_DIR, model_dir))
    model = model_class.from_pretrained(os.path.join(MODELS_DIR, model_dir)).to(DEVICE)
    model.eval()
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    print(f"  prediction:  {LABELS[pred]}")
    print(f"  confidence:  {probs[pred]:.4f}")
    print(f"  all scores:  false={probs[0]:.4f}, mixed={probs[1]:.4f}, true={probs[2]:.4f}")

if __name__ == "__main__":
    print(f"headline: \"{HEADLINE}\"\n")
    print("--- baseline (TF-IDF + Logistic Regression) ---")
    test_baseline(HEADLINE)
    print("\n--- fine-tuned BERT ---")
    test_transformer(HEADLINE, "BERT", "bert_finetuned", BertTokenizer, BertForSequenceClassification)
    print("\n--- fine-tuned RoBERTa ---")
    test_transformer(HEADLINE, "RoBERTa", "roberta_finetuned", RobertaTokenizer, RobertaForSequenceClassification)