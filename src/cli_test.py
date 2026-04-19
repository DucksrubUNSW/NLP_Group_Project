#!/usr/bin/env python3

# Example usage:
# python src/cli_test.py --model bert --text "The unemployment rate has dropped to its lowest point in 50 years"
# python src/cli_test.py --model roberta --text "The unemployment rate has dropped to its lowest point in 50 years"

import argparse
from pathlib import Path
import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field

import torch
import joblib
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)


# Free key (can sign up to gemini api free tier) - Rate limited
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFCzFAPqyEDdxL2ew5B26ChS-9svCJBns"

# paid key (Course gave $50 credit)
os.environ["GOOGLE_API_KEY"] = ""
os.environ["SERPER_API_KEY"] = "551da4c7939253fd9f113d0c8fb9612eeeba4103"


LABELS = {0: "false", 1: "mixed", 2: "true"}
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
search = GoogleSerperAPIWrapper()

class FakeNewsAnalysis(BaseModel):
    is_real: bool = Field(description="The news headline is either fake or real, determine whether the article is real/fake. True = real, False = fake.")
    confidence_score: float = Field(description="A score from 0 to 1 indicating the model's certainty.")
    reasoning: str = Field(description="A brief explanation based on search results.")

structured_llm = model.with_structured_output(FakeNewsAnalysis)

@tool
def google_search(query: str) -> str:
    """Search Google for fact-checking evidence."""
    return search.run(query)

agent = create_agent(model, tools=[google_search])

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def analyze_headline(headline: str):
    prompt = f"""
You are an expert adversarial fact-checker. Your goal is to debunk misinformation. Do not trust the headline's tone.

Identify the core factual claim.

Use the search tool to find reputable primary sources (Associated Press, Reuters, official government sites) that either confirm or explicitly refute this claim.

If no reputable source mentions this 'major' news, treat that silence as a strong signal that it is fake.

Headline: {headline}

    """
    result = agent.invoke({"messages": [("user", prompt)]})

    final_report = structured_llm.invoke(result["messages"])

    return final_report

def predict_baseline(text: str):
    model = joblib.load(PROJECT_ROOT / "baseline_model.pkl")
    vectoriser = joblib.load(PROJECT_ROOT / "baseline_vectoriser.pkl")
    pred_id = int(model.predict(vectoriser.transform([text]))[0])
    return LABELS.get(pred_id, str(pred_id)), None

def predict_bert(headline: str):
    model_dir = PROJECT_ROOT / "bert_finetuned"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_id].item())
    return LABELS.get(pred_id, str(pred_id)), confidence


def predict_roberta(headline: str):
    model_dir = PROJECT_ROOT / "roberta_finetuned"
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_id].item())
    return LABELS.get(pred_id, str(pred_id)), confidence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["baseline", "bert", "roberta"],
        required=True,
        help="Which saved model to use.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Headline text to classify.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    text = args.text.strip()

    if not text:
        raise ValueError("`--text` cannot be empty.")

    if args.model == "baseline":
        label, confidence = predict_baseline(text)
    elif args.model == "bert":
        label, confidence = predict_bert(text)
    else:
        label, confidence = predict_roberta(text)

    print(f"Model: {args.model}")
    print(f"Device: {DEVICE}")
    print(f"Headline: {text}")
    print(f"Prediction: {label}")
    if confidence is None:
        print("Model Confidence: N/A (baseline model confidence not exposed)")
    else:
        print(f"Model Confidence: {confidence:.4f}")

    print()
    report = analyze_headline(text)
    print(f"Is Real: {report.is_real}")
    print(f"Score: {report.confidence_score}")
    print(f"Reason: {report.reasoning}")


if __name__ == "__main__":
    main()
