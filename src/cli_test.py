#!/usr/bin/env python3

# Example usage:
# python src/cli_test.py --model bert --text "The unemployment rate has dropped to its lowest point in 50 years"
# python src/cli_test.py --model roberta --text "The unemployment rate has dropped to its lowest point in 50 years"

import argparse
from pathlib import Path
import os
from typing import Literal
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

from wordnet_query_expansion import expand_search_queries


# Free key (can sign up to gemini api free tier) - Rate limited
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2Ki8AX-K0dzSwHK971TuRusvIKPa407c"

# paid key (Course gave $50 credit)
#os.environ["GOOGLE_API_KEY"] = ""
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


class ThreeClassEvidenceAnalysis(BaseModel):
    verdict: Literal["true", "mixed", "false"] = Field(
        description="Evidence-based verdict using the project's true/mixed/false label scheme."
    )
    confidence_score: float = Field(description="A score from 0 to 1 indicating the evidence verdict certainty.")
    reasoning: str = Field(description="A brief explanation grounded in the retrieved evidence.")
    sources: list[str] = Field(description="Source names or URLs used as evidence.")

structured_llm = model.with_structured_output(FakeNewsAnalysis)
structured_three_class_llm = model.with_structured_output(ThreeClassEvidenceAnalysis)

@tool
def google_search(query: str) -> str:
    """Search Google for fact-checking evidence."""
    return search.run(query)


@tool
def google_search_wordnet(query: str) -> str:
    """Search Google for fact-checking evidence using WordNet-expanded query variants."""
    results = []
    for expanded_query in expand_search_queries(query):
        results.append(f"Query: {expanded_query}\n{search.run(expanded_query)}")
    return "\n\n".join(results)

agent = create_agent(model, tools=[google_search])
wordnet_agent = create_agent(model, tools=[google_search_wordnet])

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def analyze_headline(headline: str):
    prompt = f"""
You are an expert adversarial fact-checker. Your goal is to debunk misinformation. Do not trust the headline's tone.

Identify the core factual claim.

Use the search tool to find reputable primary sources (Associated Press, Reuters, official government sites) that either confirm or explicitly refute this claim. The search tool may use WordNet-expanded query variants to retrieve evidence with different wording.

If no reputable source mentions this 'major' news, treat that silence as a strong signal that it is fake.

Headline: {headline}

    """
    result = agent.invoke({"messages": [("user", prompt)]})

    final_report = structured_llm.invoke(result["messages"])

    return final_report


def analyze_headline_three_class(
    headline: str,
    base_label: str,
    base_confidence: float | None,
    use_wordnet: bool = False,
):
    confidence_text = (
        f"{base_confidence:.4f}" if base_confidence is not None else "not available"
    )
    prompt = f"""
You are an expert adversarial fact-checker. Your goal is to evaluate a headline using the project's three labels.

The only allowed labels are:
- true: reliable evidence supports the main claim
- mixed: the claim is partially true, ambiguous, lacks enough evidence, or sources conflict
- false: reliable evidence refutes the main claim or indicates the claim is fabricated

Base model prediction: {base_label}
Base model confidence: {confidence_text}

Identify the core factual claim.

Use the search tool to find reputable primary sources (Associated Press, Reuters, official government sites, official statistics sources, or established fact-checkers) that confirm, partially support, complicate, or refute this claim.

If the retrieved evidence is unclear, incomplete, or conflicting, use the mixed label.

Headline: {headline}
    """
    selected_agent = wordnet_agent if use_wordnet else agent
    result = selected_agent.invoke({"messages": [("user", prompt)]})

    final_report = structured_three_class_llm.invoke(result["messages"])

    return final_report


def combine_predictions(
    base_label: str,
    base_confidence: float | None,
    evidence_report: ThreeClassEvidenceAnalysis | None,
) -> str:
    if evidence_report is None:
        return base_label

    evidence_label = evidence_report.verdict
    evidence_confidence = evidence_report.confidence_score

    # Keep the model prediction by default. LLM confidence is often
    # overconfident, so retrieval should only override uncertain model outputs.
    if evidence_label == base_label:
        return base_label

    if base_confidence is None:
        if evidence_confidence >= 0.95:
            return evidence_label
        return base_label

    if base_confidence < 0.50 and evidence_confidence >= 0.95:
        return evidence_label

    if base_confidence < 0.60 and evidence_label == "mixed" and evidence_confidence >= 0.95:
        return "mixed"

    return base_label

def predict_baseline(text: str):
    model = joblib.load(PROJECT_ROOT / "baseline_model.pkl")
    vectoriser = joblib.load(PROJECT_ROOT / "baseline_vectoriser.pkl")
    prediction = model.predict(vectoriser.transform([text]))[0]
    return str(prediction), None

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
    parser.add_argument(
        "--use-wordnet",
        action="store_true",
        help="Use WordNet-expanded query variants in the retrieval-augmented classification step.",
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

    print()
    evidence_report = analyze_headline_three_class(
        text,
        label,
        confidence,
        use_wordnet=args.use_wordnet,
    )
    final_prediction = combine_predictions(label, confidence, evidence_report)
    print("Retrieval-Augmented Classification:")
    print(f"WordNet Query Expansion: {args.use_wordnet}")
    print(f"Evidence Verdict: {evidence_report.verdict}")
    print(f"Evidence Confidence: {evidence_report.confidence_score:.4f}")
    print(f"Final Prediction: {final_prediction}")
    print(f"Evidence Reason: {evidence_report.reasoning}")
    if evidence_report.sources:
        print("Sources:")
        for source in evidence_report.sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()
