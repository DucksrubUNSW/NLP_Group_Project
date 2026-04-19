# fine-tunes BERT (and optionally RoBERTa) on the combined dataset
# also evaluates a pre-trained (non-fine-tuned) BERT for comparison

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_combined

LABEL_TO_ID = {"false": 0, "mixed": 1, "true": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class HeadlineDataset(Dataset):
    # wraps our text + labels into a pytorch dataset for the dataloader
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

def evaluate(model, dataloader):
    # runs the model on a dataloader and returns predictions + true labels
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

def print_results(title, y_true, y_pred):
    # prints accuracy, classification report, and confusion matrix
    label_names = ["false", "mixed", "true"]
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-5, patience=2):
    # training loop with linear lr warmup and early stopping
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch+1}/{epochs}, batch {i+1}/{len(train_loader)}, loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"  epoch {epoch+1}/{epochs} done, avg loss: {avg_loss:.4f}")
        # check validation accuracy and track best model
        preds, labels = evaluate(model, val_loader)
        acc = accuracy_score(labels, preds)
        print(f"  validation accuracy: {acc:.4f}")
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  new best model! (val acc: {acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  no improvement for {patience} epochs, stopping early")
                break
    # restore the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  restored best model (val acc: {best_val_acc:.4f})")
    return model

def main():
    print(f"Using device: {DEVICE}")
    print("Loading data...")
    df = load_combined()
    X = df["text"].values
    y = np.array([LABEL_TO_ID[label] for label in df["label"].values])
    # split into train (60%), val (20%), test (20%)
    # test set stays untouched until final evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    print(f"\nTrain size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")

    # 1. pre-trained BERT without fine-tuning
    print("\n--- evaluating pre-trained BERT (no fine-tuning) ---")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_pretrained = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    ).to(DEVICE)
    test_dataset = HeadlineDataset(X_test.tolist(), y_test.tolist(), bert_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    # will basically be random since the classification head is untrained
    preds, labels = evaluate(bert_pretrained, test_loader)
    print_results("PRE-TRAINED BERT (no fine-tuning)", labels, preds)
    del bert_pretrained

    # 2. fine-tuned BERT
    print("\n--- fine-tuning BERT ---")
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    ).to(DEVICE)
    train_dataset = HeadlineDataset(X_train.tolist(), y_train.tolist(), bert_tokenizer)
    val_dataset = HeadlineDataset(X_val.tolist(), y_val.tolist(), bert_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    bert_model = train_model(bert_model, train_loader, val_loader)
    # final evaluation on held-out test set
    preds, labels = evaluate(bert_model, test_loader)
    print_results("FINE-TUNED BERT", labels, preds)
    bert_model.save_pretrained("bert_finetuned")
    bert_tokenizer.save_pretrained("bert_finetuned")
    print("Fine-tuned BERT saved to bert_finetuned/")
    del bert_model

    # 3. fine-tuned RoBERTa
    print("\n--- fine-tuning RoBERTa ---")
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=3
    ).to(DEVICE)
    train_dataset_r = HeadlineDataset(X_train.tolist(), y_train.tolist(), roberta_tokenizer)
    val_dataset_r = HeadlineDataset(X_val.tolist(), y_val.tolist(), roberta_tokenizer)
    test_dataset_r = HeadlineDataset(X_test.tolist(), y_test.tolist(), roberta_tokenizer)
    train_loader_r = DataLoader(train_dataset_r, batch_size=32, shuffle=True)
    val_loader_r = DataLoader(val_dataset_r, batch_size=32)
    test_loader_r = DataLoader(test_dataset_r, batch_size=32)
    roberta_model = train_model(roberta_model, train_loader_r, val_loader_r)
    # final evaluation on held-out test set
    preds, labels = evaluate(roberta_model, test_loader_r)
    print_results("FINE-TUNED RoBERTa", labels, preds)
    roberta_model.save_pretrained("roberta_finetuned")
    roberta_tokenizer.save_pretrained("roberta_finetuned")
    print("Fine-tuned RoBERTa saved to roberta_finetuned/")

if __name__ == "__main__":
    main()