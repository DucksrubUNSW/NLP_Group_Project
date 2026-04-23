#!/usr/bin/env python3

# baseline_model.py
# baseline model: TF-IDF + Logistic Regression for misinformation detection

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import joblib

# relative to CODE/src/
MODELS_DIR = os.path.join("..", "..", "MISC", "models")

from data_loader import load_combined


def train_baseline():
    print("Loading data...")
    df = load_combined()

    X = df["text"].values
    y = df["label"].values

    # stratified split so class ratios stay consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")

    # tfidf with bigrams and english stopwords removed
    print("\nFitting TF-IDF vectoriser...")
    vectoriser = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_tfidf = vectoriser.fit_transform(X_train)
    X_test_tfidf = vectoriser.transform(X_test)

    # balanced class weights to handle the fact that "mixed" has fewer samples
    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS: TF-IDF + Logistic Regression")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # save so we can load later for comparison
    baseline_dir = os.path.join(MODELS_DIR, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    joblib.dump(model, os.path.join(baseline_dir, "baseline_model.pkl"))
    joblib.dump(vectoriser, os.path.join(baseline_dir, "baseline_vectoriser.pkl"))
    print(f"\nModel saved to {baseline_dir}/baseline_model.pkl")
    print(f"Vectoriser saved to {baseline_dir}/baseline_vectoriser.pkl")

    return model, vectoriser

if __name__ == "__main__":
    train_baseline()