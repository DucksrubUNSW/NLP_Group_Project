# Misinformation Detection in News Headlines

**Team Name:** News Group
**Course:** COMP6713 - Natural Language Processing (2026 T1)

## Team Members

- Lachlan Johnston (z5477322)
- Sebastian Litchfield (z5477323)
- Jaiden Brocklebank (z5421748)

## Project Overview

Detects misinformation in political news statements using NLP models trained on the LIAR and FakeNewsNet datasets. Classifies statements as **true**, **mixed**, or **false**.

## Setup

```bash
cd CODE
pip install -r requirements.txt
cd src
```

## Getting the Models

You can either train the models yourself or download pre-trained versions.

### Option A: Train the models

This will take a while, especially the BERT/RoBERTa training on CPU.

```bash
python baseline_model.py
python bert_model.py
```

Trained models will be saved to `MISC/models/`.

### Option B: Download pre-trained models

Download the models from [\[Models\]](https://drive.google.com/file/d/1NwbpDr42JVU8C4C--t8SmLaKSikfS77F/view?usp=sharing) and place them in the `MISC/models/` directory so the structure looks like:

```
MISC/
└── models/
    ├── baseline/
    │   ├── baseline_model.pkl
    │   └── baseline_vectoriser.pkl
    ├── bert_finetuned/
    └── roberta_finetuned/
```

## Classifying a Headline

Once the models are available, you can test all three on a headline by editing the `HEADLINE` variable in `test_models.py` and running:

```bash
python test_models.py
```

## CLI with LangChain Fact-Checking

`cli_test.py` runs a headline through a model and then uses LangChain + Google Search to fact-check it against external sources.

### API Keys

You'll need two API keys. Open `cli_test.py` and paste them into the relevant lines at the top of the file:

1. **Google Gemini API key** (free tier available)
   - Go to https://aistudio.google.com/apikey
   - Click "Create API key"
   - Copy the key and paste it into the `GOOGLE_API_KEY` line

2. **Serper API key** (for Google Search)
   - Go to https://serper.dev and sign up
   - Copy your API key and paste it into the `SERPER_API_KEY` line

### Usage

```bash
python cli_test.py --model bert --text "The unemployment rate has dropped to its lowest point in 50 years"
python cli_test.py --model roberta --text "The unemployment rate has dropped to its lowest point in 50 years"
python cli_test.py --model baseline --text "The unemployment rate has dropped to its lowest point in 50 years"
```

## Gradio Demo

A web-based demo lets you enter a headline, pick a model, and see the prediction and confidence in your browser.

```bash
python demo.py
```

This opens a local webpage with:
- A text box for the headline
- A dropdown to select **Baseline**, **BERT**, or **RoBERTa**
- Prediction and confidence displayed below

Optionally, expand the **RAG / LLM Analysis** section, tick the checkbox, and paste your Google Gemini API key to also run evidence-based fact-checking via LangChain + Google Search (uses the same keys described in the CLI section above).

## Running Tests

```bash
python -m pytest test_headline_predictions.py -v
```

Results are saved to `MISC/test_results/headline_results.txt`.

## Directory Structure

```
NewsGroup/
├── README.md
├── CONTRIBUTION.md
├── REPORT.docx
├── PRESENTATION.pdf
├── CODE/
│   ├── requirements.txt
│   └── src/
│       ├── data_loader.py
│       ├── baseline_model.py
│       ├── bert_model.py
│       ├── cli_test.py
│       ├── conftest.py
│       ├── demo.py
│       ├── evaluate_langchain.py
│       ├── evaluate_models.py
│       ├── make_langchain_eval_set.py
│       ├── test_headline_predictions.py
│       ├── wordnet_query_expansion.py
│       └── test_models.py
└── MISC/
    ├── data/
    │   ├── langchain_eval_cases.csv
    │   ├── liar/
    │   └── fakenewsnet/
    ├── models/
    │   ├── baseline/
    │   ├── bert_finetuned/
    │   └── roberta_finetuned/
    └── test_results/
```