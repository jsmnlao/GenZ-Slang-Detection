"""Sentence-level error analysis for BERTweet slang classification.

Input CSV must have columns:
  - text       : the full sentence / sequence
  - gold_tags  : SLANG or NOT_SLANG label from the ground truth
  - pred_tags  : SLANG or NOT_SLANG label from the model
  - exact_match: (optional) True/False match

By default, runs analysis on both:
  - bertweet_binary_results/test_predictions.csv
      -> error_analysis/bert_sentence_test_report.txt
  - bertweet_binary_results/generalization_test_predictions.csv
      -> error_analysis/bert_sentence_generalization_report.txt

Usage:
  python error_analysis_BERT_sentence.py
  python error_analysis_BERT_sentence.py --csv path/to/file.csv --out path/to/report.txt
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd

SLANG = "SLANG"
NOT_SLANG = "NOT_SLANG"

POLYSEMY_WORDS = [
    "fire",
    "sick",
    "slay",
    "extra",
    "basic",
    "beta",
    "squad",
    "lore",
    "ghost",
    "mood",
    "woke",
    "caps",
    "big",
    "sus",
    "cringe",
]

DEFAULT_RUNS = [
    (
        "bertweet_binary_results/test_predictions.csv",
        "error_analysis/bert_sentence_test_report.txt",
    ),
    (
        "bertweet_binary_results/generalization_test_predictions.csv",
        "error_analysis/bert_sentence_generalization_report.txt",
    ),
]


def load_data(csv_path):
    if not csv_path:
        raise ValueError("CSV file is missing")

    df = pd.read_csv(csv_path)
    required = {"text", "gold_tags", "pred_tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    for col in ("text", "gold_tags", "pred_tags"):
        df[col] = df[col].fillna("").astype(str)

    return df


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_label(value):
    value = str(value).strip().upper()
    if value == SLANG:
        return SLANG
    if value == NOT_SLANG:
        return NOT_SLANG
    raise ValueError(f"Unexpected label: {value}")


def prepare_rows(df):
    out = df.copy()
    out["gold_label"] = out["gold_tags"].map(normalize_label)
    out["pred_label"] = out["pred_tags"].map(normalize_label)
    return out


def classify_sentences(df):
    buckets = {"TP": [], "FP": [], "TN": [], "FN": []}
    for _, row in df.iterrows():
        gold_is_slang = row["gold_label"] == SLANG
        pred_is_slang = row["pred_label"] == SLANG

        if gold_is_slang and pred_is_slang:
            buckets["TP"].append(row)
        elif (not gold_is_slang) and pred_is_slang:
            buckets["FP"].append(row)
        elif (not gold_is_slang) and (not pred_is_slang):
            buckets["TN"].append(row)
        else:
            buckets["FN"].append(row)

    return {key: pd.DataFrame(rows) for key, rows in buckets.items()}


def fmt_examples(rows, n=10):
    if rows.empty:
        return "  (none)\n"

    lines = []
    for _, row in rows.head(n).iterrows():
        lines.append(
            f'  gold={row["gold_label"]:<10}  pred={row["pred_label"]:<10}  | "{row["text"]}"'
        )
    return "\n".join(lines) + "\n"


def sentence_level_section(df):
    lines = ["-" * 70, "SENTENCE-LEVEL EXACT MATCH", "-" * 70]
    if "exact_match" in df.columns:
        exact = df["exact_match"].astype(str).str.lower().eq("true")
        correct = int(exact.sum())
        total = len(df)
        pct = (correct / total * 100.0) if total else 0.0
        lines.append(f"  Exact matches : {correct} / {total}  ({pct:.1f}%)")
    else:
        lines.append("  (no exact_match column in CSV)")
    lines.append("")
    return "\n".join(lines)


def polysemy_section(df):
    lines = [
        "=" * 70,
        "POLYSEMOUS WORD ANALYSIS",
        "=" * 70,
        f"Tracking {len(POLYSEMY_WORDS)} words: " + ", ".join(POLYSEMY_WORDS),
        "",
    ]

    word_stats = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        tokens = set(re.findall(r"\b\w+\b", row["text"].lower()))
        present_words = [word for word in POLYSEMY_WORDS if word in tokens]
        if not present_words:
            continue

        gold_is_slang = row["gold_label"] == SLANG
        pred_is_slang = row["pred_label"] == SLANG

        if gold_is_slang and pred_is_slang:
            category = "TP"
        elif (not gold_is_slang) and pred_is_slang:
            category = "FP"
        elif (not gold_is_slang) and (not pred_is_slang):
            category = "TN"
        else:
            category = "FN"

        for word in present_words:
            word_stats[word][category].append(row)

    if not word_stats:
        lines.append("  No tracked polysemous words found in dataset.\n")
        return "\n".join(lines)

    lines.append(f"  {'Word':<12} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  Verdict")
    lines.append("  " + "-" * 55)

    for word in POLYSEMY_WORDS:
        if word not in word_stats:
            lines.append(
                f"  {word:<12} {'-':>5} {'-':>5} {'-':>5} {'-':>5}  not in dataset"
            )
            continue

        counts = {cat: len(word_stats[word][cat]) for cat in ("TP", "FP", "TN", "FN")}
        verdict = "x errors" if (counts["FP"] + counts["FN"]) > 0 else "ok correct"
        lines.append(
            f"  {word:<12} {counts['TP']:>5} {counts['FP']:>5} "
            f"{counts['TN']:>5} {counts['FN']:>5}  {verdict}"
        )

    lines.extend(["", "Detailed sentence rows for polysemous words:", ""])

    for word in POLYSEMY_WORDS:
        if word not in word_stats:
            continue
        lines.append(f"  [{word}]")
        for category in ("TP", "FP", "TN", "FN"):
            for _, row in pd.DataFrame(word_stats[word][category]).iterrows():
                lines.append(
                    f'    {category}  gold={row["gold_label"]:<10}  '
                    f'pred={row["pred_label"]:<10}  | "{row["text"]}"'
                )
        lines.append("")

    return "\n".join(lines)


def build_report(df, source_label, n_examples=10):
    df = prepare_rows(df)
    buckets = classify_sentences(df)

    tp = len(buckets["TP"])
    fp = len(buckets["FP"])
    tn = len(buckets["TN"])
    fn = len(buckets["FN"])
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0

    out = [
        "=" * 70,
        "SLANG SENTENCE CLASSIFICATION - ERROR ANALYSIS REPORT",
        "=" * 70,
        f"Source        : {source_label}",
        f"Sentences     : {len(df)}",
        "",
        "-" * 70,
        "SENTENCE-LEVEL CONFUSION MATRIX",
        "-" * 70,
        f"  True  Positives (TP) - slang sentence correctly identified   : {tp:>6}",
        f"  False Positives (FP) - non-slang sentence called slang       : {fp:>6}",
        f"  True  Negatives (TN) - non-slang sentence correctly rejected : {tn:>6}",
        f"  False Negatives (FN) - slang sentence missed by model        : {fn:>6}",
        "",
        "-" * 70,
        "SENTENCE-LEVEL METRICS",
        "-" * 70,
        f"  Accuracy  : {accuracy:.4f}",
        f"  Precision : {precision:.4f}",
        f"  Recall    : {recall:.4f}",
        f"  F1 Score  : {f1:.4f}",
        "",
        sentence_level_section(df),
    ]

    defs = {
        "TP": "True  Positives - slang sentence correctly identified",
        "FP": "False Positives - non-slang sentence incorrectly labelled as slang",
        "TN": "True  Negatives - non-slang sentence correctly rejected",
        "FN": "False Negatives - slang sentence the model failed to catch",
    }

    for category in ("TP", "FP", "TN", "FN"):
        out.extend(
            [
                "-" * 70,
                f"{category} EXAMPLES  ({defs[category]})",
                f"  Count: {len(buckets[category])}",
                "",
                fmt_examples(buckets[category], n=n_examples),
            ]
        )

    out.extend(["", polysemy_section(df), "=" * 70, "END OF REPORT", "=" * 70])
    return "\n".join(out)


def run_single(csv_path, out_path, n_examples=10):
    print(f"\n{'=' * 70}")
    print(f"Processing : {csv_path}")
    print(f"Output     : {out_path}")
    print(f"{'=' * 70}")

    df = load_data(csv_path)
    report = build_report(df, source_label=csv_path, n_examples=n_examples)

    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    # print(report)
    print(f"\nReport being genereated...")
    print(f"\nReport finished and written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sentence-level error analysis for BERTweet slang classification"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to a single predictions CSV. If omitted, runs both default files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for single-file mode (requires --csv).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Max examples shown per bucket (default: 10)",
    )
    args = parser.parse_args()

    if args.examples < 0:
        parser.error("--examples must be >= 0")

    if args.csv:
        out_path = args.out if args.out else "error_analysis/bert_sentence_report.txt"
        run_single(args.csv, out_path, n_examples=args.examples)
    else:
        for csv_path, out_path in DEFAULT_RUNS:
            run_single(csv_path, out_path, n_examples=args.examples)


if __name__ == "__main__":
    main()
