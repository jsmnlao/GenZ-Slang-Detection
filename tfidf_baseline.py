# -*- coding: utf-8 -*-
"""
tfidf_baseline.py

Implements a TF-IDF sentence classifier for detecting Gen Z slang in text.

Usage:
    python tfidf_baseline.py
    python tfidf_baseline.py --train output/train.csv --dev output/dev.csv --test output/test.csv --generalization output/generalization_test.csv --output_dir tfidf_baseline

Output:
    tfidf_baseline/tfidf_baseline_report.txt
"""

import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

NOT_SLANG = "NOT_SLANG"
SLANG = "SLANG"
RANDOM_SEED = 42


def sentence_level_eval(gold_labels, pred_labels):
    label_map = {SLANG: 1, NOT_SLANG: 0}
    y_true = [label_map[x] for x in gold_labels]
    y_pred = [label_map[x] for x in pred_labels]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_split(df, pipeline, split_name):
    texts = df["text"].fillna("").astype(str).tolist()
    gold_labels = df["label"].tolist()

    print(f"[{split_name}] Running predictions on {len(texts)} samples...")
    pred_labels = pipeline.predict(texts).tolist()

    results = sentence_level_eval(gold_labels, pred_labels)
    print(f"\n=== {split_name} Eval Results ===")
    print(
        f"Sentence - Accuracy: {results['accuracy']:.4f}  Precision: {results['precision']:.4f}  Recall: {results['recall']:.4f}  F1: {results['f1']:.4f}"
    )
    return results, pred_labels


def save_predictions(df, pred_labels, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    out = pd.DataFrame({
        "text": df["text"].fillna("").astype(str).tolist(),
        "gold_tags": df["label"].tolist(),
        "pred_tags": pred_labels,
        "exact_match": [g == p for g, p in zip(df["label"].tolist(), pred_labels)],
    })
    out.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def build_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def top_weighted_features(pipeline, top_k=20):
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    positive_idx = coefficients.argsort()[-top_k:][::-1]
    negative_idx = coefficients.argsort()[:top_k]

    positive = [(feature_names[i], coefficients[i]) for i in positive_idx]
    negative = [(feature_names[i], coefficients[i]) for i in negative_idx]
    return positive, negative


def save_report(results, positive_features, negative_features, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("TF-IDF BASELINE EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        for split_name, split_results in results.items():
            f.write(f"=== {split_name.upper()} ===\n")
            f.write("  Sentence-level:\n")
            f.write(f"    Accuracy  : {split_results['accuracy']:.4f}\n")
            f.write(f"    Precision : {split_results['precision']:.4f}\n")
            f.write(f"    Recall    : {split_results['recall']:.4f}\n")
            f.write(f"    F1        : {split_results['f1']:.4f}\n\n")

        f.write("=== TOP POSITIVE FEATURES (predicting SLANG) ===\n")
        for feature, weight in positive_features:
            f.write(f"  {feature:<25} {weight:.4f}\n")
        f.write("\n")

        f.write("=== TOP NEGATIVE FEATURES (predicting NOT_SLANG) ===\n")
        for feature, weight in negative_features:
            f.write(f"  {feature:<25} {weight:.4f}\n")

    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="output/train.csv")
    parser.add_argument("--dev", default="output/dev.csv")
    parser.add_argument("--test", default="output/test.csv")
    parser.add_argument("--generalization", default="output/generalization_test.csv")
    parser.add_argument("--output_dir", default="tfidf_baseline")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train)
    df_dev = pd.read_csv(args.dev)
    df_test = pd.read_csv(args.test)
    df_generalization = pd.read_csv(args.generalization)

    print("Training TF-IDF baseline on TRAIN split...")
    pipeline = build_pipeline()
    pipeline.fit(df_train["text"].fillna("").astype(str), df_train["label"])
    print("Training complete.")

    print("Evaluating TRAIN...")
    train_results, _ = evaluate_split(df_train, pipeline, "TRAIN")
    print("Evaluating DEV...")
    dev_results, _ = evaluate_split(df_dev, pipeline, "DEV")
    print("Evaluating TEST...")
    test_results, test_preds = evaluate_split(df_test, pipeline, "TEST")
    save_predictions(df_test, test_preds, os.path.join(args.output_dir, "test_predictions.csv"))
    print("Evaluating GENERALIZATION_TEST...")
    generalization_results, gen_preds = evaluate_split(
        df_generalization, pipeline, "GENERALIZATION_TEST"
    )
    save_predictions(df_generalization, gen_preds, os.path.join(args.output_dir, "generalization_test_predictions.csv"))

    positive_features, negative_features = top_weighted_features(pipeline)
    report_path = os.path.join(args.output_dir, "tfidf_baseline_report.txt")
    save_report(
        {
            "train": train_results,
            "dev": dev_results,
            "test": test_results,
            "generalization_test": generalization_results,
        },
        positive_features,
        negative_features,
        report_path,
    )


if __name__ == "__main__":
    main()
