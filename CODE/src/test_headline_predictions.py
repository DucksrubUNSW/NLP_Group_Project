#!/usr/bin/env python3

# checks predictions of fine-tuned BERT and RoBERTa on known true/false headlines
# current pass rate is 27/40
# usage: cd CODE/src && python -m pytest test_headline_predictions.py -v

import os
import pytest
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

LABELS = {0: "false", 1: "mixed", 2: "true"}
MODELS_DIR = os.path.join("..", "..", "MISC", "models")
RESULTS_DIR = os.path.join("..", "..", "MISC", "test_results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "headline_results.txt")
BERT_DIR = os.path.join(MODELS_DIR, "bert_finetuned")
ROBERTA_DIR = os.path.join(MODELS_DIR, "roberta_finetuned")

os.makedirs(RESULTS_DIR, exist_ok=True)

# stores results as tests run, writes to file at the end
_results = []

@pytest.fixture(autouse=True)
def record_result(request):
    yield
    outcome = "PASSED" if request.node.rep_call.passed else "FAILED"
    _results.append(f"{outcome} - {request.node.name}")


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def predict_bert(text: str) -> str:
    tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
    model = BertForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=1).item())
    return LABELS.get(pred_id, str(pred_id))

def predict_roberta(text: str) -> str:
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_DIR)
    model = RobertaForSequenceClassification.from_pretrained(ROBERTA_DIR).to(DEVICE)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=1).item())
    return LABELS.get(pred_id, str(pred_id))

TRUE_CASES = [
    "Watch Carrie Underwood Open Super Bowl 52 With Her New Video for \u201cThe Champion\u201d",
    "Kylie Jenner is reportedly 'nervous' to give birth but 'excited' for motherhood",
    "Real Housewives Mysteries: The Stolen Homes, Mystery Lovers and Friendship-Ending Fights That Rocked the Franchise",
    "Kathy Griffin \u2018No Longer Sorry\u2019 About Trump Photo Shoot \u2013 Rolling Stone",
    "A Complete Timeline of Selena Gomez and Justin Bieber's Relationship",
    "When Will \u2018Claws\u2019 Season 2 Be On Hulu?",
    "Critics' Choice Awards - Critics' Choice Awards",
    "Bellamy Young Opens Up About Being Adopted, Her Real First Name and How She Almost Missed Out on Scandal",
    "Find Out Which of The Real Housewives of Orange County Won't Be Returning",
    "The Bachelorette Rachel Lindsay's Dog Copper and What Happened to His Leg",
]

FALSE_CASES = [
    "Did Miley Cyrus and Liam Hemsworth secretly get married?",
    "Paris Jackson & Cara Delevingne Enjoy Night Out In Matching Outfits: They Have \u2018Amazing Chemistry\u2019",
    "Celebrities Join Tax March in Protest of Donald Trump",
    "Cindy Crawford's daughter Kaia Gerber wears a wig after dining with Harry Styles",
    "Biggest celebrity scandals of 2016",
    "Caitlyn Jenner Addresses Rumored Romance With Sophia Hutchins",
    "Taylor Swift Reportedly Reacts To Tom Hiddleston\u2019s Golden Globes Win",
    "For The Love Of God, Why Can't Anyone Write Kate McKinnon A Good Movie Role?",
    "Miley Cyrus, Liam Hemsworth Did NOT Get Married On Tybee Island, Despite Report",
    "Miley Cyrus Claims \u2018Satan Is A Nice Guy; He\u2019s Misunderstood\u2019",
]

@pytest.mark.parametrize("headline", TRUE_CASES)
def test_true_cases_with_bert(headline: str):
    assert predict_bert(headline) == "true"

@pytest.mark.parametrize("headline", TRUE_CASES)
def test_true_cases_with_roberta(headline: str):
    assert predict_roberta(headline) == "true"

@pytest.mark.parametrize("headline", FALSE_CASES)
def test_false_cases_with_bert(headline: str):
    assert predict_bert(headline) == "false"

@pytest.mark.parametrize("headline", FALSE_CASES)
def test_false_cases_with_roberta(headline: str):
    assert predict_roberta(headline) == "false"