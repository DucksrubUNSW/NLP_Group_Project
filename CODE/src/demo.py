#!/usr/bin/env python3

# Usage: python demo.py

import os
import gradio as gr
from pathlib import Path
from typing import Literal

import torch
import joblib
from pydantic import BaseModel, Field
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

LABELS = {0: "false", 1: "mixed", 2: "true"}
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MISC_MODELS = Path(__file__).resolve().parent.parent.parent / "MISC" / "models"

LABEL_COLOURS = {"true": "✅ True", "mixed": "⚠️ Mixed", "false": "❌ False"}


# ---------- prediction functions ----------

def predict_baseline(text: str):
    mdl = joblib.load(MISC_MODELS / "baseline" / "baseline_model.pkl")
    vec = joblib.load(MISC_MODELS / "baseline" / "baseline_vectoriser.pkl")
    prediction = mdl.predict(vec.transform([text]))[0]
    return str(prediction), None


def predict_bert(headline: str):
    model_dir = MISC_MODELS / "bert_finetuned"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    mdl = BertForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    mdl.eval()
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_id].item())
    return LABELS.get(pred_id, str(pred_id)), confidence


def predict_roberta(headline: str):
    model_dir = MISC_MODELS / "roberta_finetuned"
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    mdl = RobertaForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    mdl.eval()
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_id].item())
    return LABELS.get(pred_id, str(pred_id)), confidence


# ---------- LLM / RAG helpers (lazy-imported so the demo works without API keys) ----------

class ThreeClassEvidenceAnalysis(BaseModel):
    verdict: Literal["true", "mixed", "false"] = Field(
        description="Evidence-based verdict: true / mixed / false."
    )
    confidence_score: float = Field(description="Score 0–1 for verdict certainty.")
    reasoning: str = Field(description="Brief explanation grounded in retrieved evidence.")
    sources: list[str] = Field(description="Source names or URLs used as evidence.")


def _build_rag_chain(google_api_key: str):
    from langchain_community.utilities import GoogleSerperAPIWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.tools import tool
    from langchain.agents import create_agent

    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["SERPER_API_KEY"] = "551da4c7939253fd9f113d0c8fb9612eeeba4103"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    search = GoogleSerperAPIWrapper()
    structured_llm = llm.with_structured_output(ThreeClassEvidenceAnalysis)

    @tool
    def google_search(query: str) -> str:
        """Search Google for fact-checking evidence."""
        return search.run(query)

    agent = create_agent(llm, tools=[google_search])
    return agent, structured_llm


def run_rag_analysis(headline: str, base_label: str, base_confidence, google_api_key: str):
    agent, structured_llm = _build_rag_chain(google_api_key)
    confidence_text = f"{base_confidence:.4f}" if base_confidence is not None else "not available"
    prompt = f"""
You are an expert adversarial fact-checker. Your goal is to evaluate a headline using the project's three labels.

The only allowed labels are:
- true: reliable evidence supports the main claim
- mixed: the claim is partially true, ambiguous, lacks enough evidence, or sources conflict
- false: reliable evidence refutes the main claim or indicates the claim is fabricated

Base model prediction: {base_label}
Base model confidence: {confidence_text}

Identify the core factual claim. Use the search tool to find reputable primary sources that confirm,
partially support, complicate, or refute this claim. If evidence is unclear or conflicting, use mixed.

Headline: {headline}
    """
    result = agent.invoke({"messages": [("user", prompt)]})
    report: ThreeClassEvidenceAnalysis = structured_llm.invoke(result["messages"])
    return report


def combine_predictions(base_label, base_confidence, evidence_report):
    if evidence_report is None:
        return base_label
    evidence_label = evidence_report.verdict
    evidence_confidence = evidence_report.confidence_score
    if evidence_label == base_label:
        return base_label
    if base_confidence is None:
        return evidence_label if evidence_confidence >= 0.95 else base_label
    if base_confidence < 0.50 and evidence_confidence >= 0.95:
        return evidence_label
    if base_confidence < 0.60 and evidence_label == "mixed" and evidence_confidence >= 0.95:
        return "mixed"
    return base_label


# ---------- main predict function called by Gradio ----------

def predict(headline: str, model_choice: str, use_rag: bool, google_api_key: str):
    headline = headline.strip()
    if not headline:
        return "Please enter a headline.", "", "", "", "", ""

    # Base model prediction
    if model_choice == "Baseline (TF-IDF + LogReg)":
        label, confidence = predict_baseline(headline)
    elif model_choice == "BERT (fine-tuned)":
        label, confidence = predict_bert(headline)
    else:
        label, confidence = predict_roberta(headline)

    display_label = LABEL_COLOURS.get(label, label)
    conf_str = f"{confidence:.1%}" if confidence is not None else "N/A (baseline)"

    if not use_rag:
        return display_label, conf_str, "", "", "", ""

    if not google_api_key.strip():
        return display_label, conf_str, "⚠️ Enter a Google API key to enable RAG analysis.", "", "", ""

    try:
        report = run_rag_analysis(headline, label, confidence, google_api_key.strip())
    except Exception as e:
        return display_label, conf_str, f"RAG error: {e}", "", "", ""

    final = combine_predictions(label, confidence, report)
    final_display = LABEL_COLOURS.get(final, final)
    rag_verdict = LABEL_COLOURS.get(report.verdict, report.verdict)
    sources_md = "\n".join(f"- {s}" for s in report.sources) if report.sources else "None"

    return (
        display_label,
        conf_str,
        rag_verdict,
        f"{report.confidence_score:.1%}",
        report.reasoning,
        sources_md,
    )


# ---------- Gradio UI ----------

CUSTOM_CSS = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
}
"""

with gr.Blocks(title="Fake News Detector", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# 📰 Fake News Headline Detector")
    gr.Markdown(
        "Enter a news headline, choose a model, and optionally enable "
        "Retrieval-Augmented Generation (RAG) for evidence-based fact-checking."
    )

    with gr.Row():
        with gr.Column(scale=2):
            headline_input = gr.Textbox(
                label="Headline",
                placeholder="e.g. The unemployment rate has dropped to its lowest point in 50 years",
                lines=2,
            )
            model_dropdown = gr.Dropdown(
                choices=[
                    "Baseline (TF-IDF + LogReg)",
                    "BERT (fine-tuned)",
                    "RoBERTa (fine-tuned)",
                ],
                value="BERT (fine-tuned)",
                label="Model",
            )
            with gr.Accordion("RAG / LLM Analysis (optional)", open=False):
                use_rag = gr.Checkbox(label="Enable RAG analysis", value=False)
                google_api_key = gr.Textbox(
                    label="Google API Key",
                    placeholder="Paste your Gemini API key here",
                    type="password",
                )

            submit_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Base Model")
            out_label = gr.Textbox(label="Prediction", interactive=False)
            out_conf = gr.Textbox(label="Confidence", interactive=False)

            gr.Markdown("### RAG / Evidence Analysis")
            out_rag_verdict = gr.Textbox(label="Evidence Verdict", interactive=False)
            out_rag_conf = gr.Textbox(label="Evidence Confidence", interactive=False)
            out_reasoning = gr.Textbox(label="Reasoning", interactive=False, lines=4)
            out_sources = gr.Textbox(label="Sources", interactive=False, lines=3)

    submit_btn.click(
        fn=predict,
        inputs=[headline_input, model_dropdown, use_rag, google_api_key],
        outputs=[out_label, out_conf, out_rag_verdict, out_rag_conf, out_reasoning, out_sources],
    )

    gr.Examples(
        examples=[
            ["The unemployment rate has dropped to its lowest point in 50 years", "BERT (fine-tuned)", False, ""],
            ["Scientists discover cure for all cancers using household vinegar", "RoBERTa (fine-tuned)", False, ""],
            ["Federal Reserve raises interest rates by 0.25%", "Baseline (TF-IDF + LogReg)", False, ""],
        ],
        inputs=[headline_input, model_dropdown, use_rag, google_api_key],
    )


if __name__ == "__main__":
    demo.launch()
