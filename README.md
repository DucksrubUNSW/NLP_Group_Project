# Misinformation Detection in News Headlines

Detects misinformation in political news statements using NLP models trained on the LIAR and FakeNewsNet datasets. Classifies statements as **true**, **mixed**, or **false**.

## Setup

```bash
pip install -r requirements.txt
```

## Classifying a headline

### Baseline (TF-IDF + Logistic Regression)

Make sure you've trained the baseline first (`python src/baseline_model.py`), then:

```python
import joblib

model = joblib.load("baseline_model.pkl")
vectoriser = joblib.load("baseline_vectoriser.pkl")

headline = "The unemployment rate has dropped to its lowest point in 50 years"
prediction = model.predict(vectoriser.transform([headline]))[0]
print(prediction)  # true, mixed, or false
```

### Fine-tuned BERT

Make sure you've trained BERT first (`python src/bert_model.py`), then:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert_finetuned")
model = BertForSequenceClassification.from_pretrained("bert_finetuned")
model.eval()

headline = "The unemployment rate has dropped to its lowest point in 50 years"
inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

labels = {0: "false", 1: "mixed", 2: "true"}
print(labels[pred])
```

### Fine-tuned RoBERTa

Same as BERT but swap the imports and model path:

```python
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained("roberta_finetuned")
model = RobertaForSequenceClassification.from_pretrained("roberta_finetuned")
model.eval()

headline = "The unemployment rate has dropped to its lowest point in 50 years"
inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

labels = {0: "false", 1: "mixed", 2: "true"}
print(labels[pred])
```

## Training the models

```bash
python src/baseline_model.py   # trains TF-IDF + Logistic Regression
python src/bert_model.py        # trains BERT and RoBERTa
```

## Team

- Lachlan Johnston (z5477322)
- Sebastian Litchfield (z5477323)
- Jaiden Brocklebank (z5421748)