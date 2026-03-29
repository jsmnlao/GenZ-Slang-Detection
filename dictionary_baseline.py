# -*- coding: utf-8 -*-
"""
dictionary_baseline.py

Implements rule-based, dictionary look up model for detecting Gen Z slang in text using the MLBTrio slang dataset.

Usage:
    python dictionary_baseline.py
    python dictionary_baseline.py --mlbtrio data/mlbtrio_cleaned.csv --train output/train.csv --dev output/dev.csv --test output/test.csv --output_dir dictionary_baseline

Output:
    dictionary_baseline/dictionary_baseline_report.txt
"""

import os
import argparse
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

NOT_SLANG = "NOT_SLANG"
SLANG = "SLANG"


def normalize(slang_term):
    slang_term = slang_term.lower().strip()
    slang_term = re.sub(r"\s+", " ", slang_term)
    return slang_term


def build_slang_list(df_mlbtrio, slang_col="Slang"):
    slang_list = sorted(
        set(normalize(t) for t in df_mlbtrio[slang_col]),
        key=len,
        reverse=True,  # sort longest first
    )
    return slang_list


def predict_bio_tags(text, patterns):
    tokens = text.split()
    lowered = [t.lower() for t in tokens]
    tags = ["O"] * len(tokens)
    for pattern in patterns:
        pattern_tokens = pattern.split()
        n = len(pattern_tokens)
        for i in range(len(tokens) - n + 1):
            if lowered[i : i + n] == pattern_tokens:
                if all(tag == "O" for tag in tags[i : i + n]):
                    tags[i] = "B-SLANG"
                    for j in range(1, n):
                        tags[i + j] = "I-SLANG"
    return tags


def predict_sentence_tag(tags):
    return SLANG if any(t != "O" for t in tags) else NOT_SLANG


def make_predictions(texts, slang_list):
    pred_tags, pred_labels = [], []

    for text in texts:
        tags = predict_bio_tags(str(text), slang_list)
        pred_tags.append(tags)
        pred_labels.append(predict_sentence_tag(tags))

    return pred_tags, pred_labels


def sentence_level_eval(gold_labels, pred_labels):
    label_map = {"SLANG": 1, "NOT_SLANG": 0}
    y_true = [label_map[x] for x in gold_labels]
    y_pred = [label_map[x] for x in pred_labels]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def token_level_eval(gold_tags_list, pred_tags_list):
    gold_flat, pred_flat = [], []
    for gold_tags, pred_tags in zip(gold_tags_list, pred_tags_list):
        gold_seq = str(gold_tags).split()
        if len(gold_seq) != len(pred_tags):
            continue
        gold_flat.extend(gold_seq)
        pred_flat.extend(pred_tags)
    return {
        "accuracy": accuracy_score(gold_flat, pred_flat),
        "precision": precision_score(
            gold_flat,
            pred_flat,
            labels=["B-SLANG", "I-SLANG"],
            average="micro",
            zero_division=0,
        ),
        "recall": recall_score(
            gold_flat,
            pred_flat,
            labels=["B-SLANG", "I-SLANG"],
            average="micro",
            zero_division=0,
        ),
        "f1": f1_score(
            gold_flat,
            pred_flat,
            labels=["B-SLANG", "I-SLANG"],
            average="micro",
            zero_division=0,
        ),
    }


def evaluate_dataset(df, slang_list, split_name=""):
    texts = df["text"].tolist()
    gold_labels = df["label"].tolist()
    gold_tags = df["tags"].tolist()

    print(f"[{split_name}] Running predictions on {len(texts)} samples...")
    pred_tags, pred_labels = make_predictions(texts, slang_list)
    print(f"[{split_name}] Predictions done. Running sentence-level eval...")

    sentence_eval_results = sentence_level_eval(gold_labels, pred_labels)
    print(f"[{split_name}] Sentence eval done. Running token-level eval...")

    token_eval_results = token_level_eval(gold_tags, pred_tags)
    print(f"[{split_name}] Token eval done.")

    print(f"\n=== {split_name} Eval Results ===")
    print(
        f"Sentence — Accuracy: {sentence_eval_results['accuracy']:.4f}  Precision: {sentence_eval_results['precision']:.4f}  Recall: {sentence_eval_results['recall']:.4f}  F1: {sentence_eval_results['f1']:.4f}"
    )
    print(
        f"Token    — A: {token_eval_results['accuracy']:.4f}  Precision: {token_eval_results['precision']:.4f}  Recall: {token_eval_results['recall']:.4f}  F1: {token_eval_results['f1']:.4f}"
    )

    return sentence_eval_results, token_eval_results


def run_evaluations(train_path, dev_path, test_path, slang_list):
    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    df_test = pd.read_csv(test_path)

    print("Evalutating TRAIN...")
    train_sentence_out, train_token_out = evaluate_dataset(
        df_train, slang_list, "TRAIN"
    )
    print("Evalutating DEV...")
    dev_sentence_out, dev_token_out = evaluate_dataset(df_dev, slang_list, "DEV")
    print("Evalutating TEST...")
    test_sentence_out, test_token_out = evaluate_dataset(df_test, slang_list, "TEST")

    return {
        "train": {"sentence": train_sentence_out, "token": train_token_out},
        "dev": {"sentence": dev_sentence_out, "token": dev_token_out},
        "test": {"sentence": test_sentence_out, "token": test_token_out},
    }


def save_report(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DICTIONARY BASELINE EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write("\n")

        for split in ["train", "dev", "test"]:
            f.write(f"=== {split.upper()} ===\n")

            s = results[split]["sentence"]
            f.write(f"  Sentence-level:\n")
            f.write(f"    Accuracy  : {s['accuracy']:.4f}\n")
            f.write(f"    Precision : {s['precision']:.4f}\n")
            f.write(f"    Recall    : {s['recall']:.4f}\n")
            f.write(f"    F1        : {s['f1']:.4f}\n")

            t = results[split]["token"]
            f.write(f"  Token-level:\n")
            f.write(f"    Accuracy  : {t['accuracy']:.4f}\n")
            f.write(f"    Precision : {t['precision']:.4f}\n")
            f.write(f"    Recall    : {t['recall']:.4f}\n")
            f.write(f"    F1        : {t['f1']:.4f}\n")
            f.write("\n")
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlbtrio", default="data/mlbtrio_cleaned.csv")
    parser.add_argument("--train", default="output/train.csv")
    parser.add_argument("--dev", default="output/dev.csv")
    parser.add_argument("--test", default="output/test.csv")
    parser.add_argument("--output_dir", default="dictionary_baseline")
    args = parser.parse_args()

    df_mlbtrio = pd.read_csv(args.mlbtrio)
    slang_list = build_slang_list(df_mlbtrio)

    print("Running evaluations...")
    results = run_evaluations(args.train, args.dev, args.test, slang_list)

    report_path = os.path.join(args.output_dir, "dictionary_baseline_report.txt")
    save_report(results, report_path)


if __name__ == "__main__":
    main()
