"""
Slang Token Classification — Error Analysis (BIO format)
=========================================================
Input CSV must have columns:
  - text       : the full sentence / sequence
  - gold_tags  : space-separated BIO tags (e.g. "O O B-SLANG I-SLANG O")
  - pred_tags  : space-separated BIO tags from the model
  - exact_match: (optional) True/False sentence-level match

Labels treated as SLANG: any tag starting with B- or I-
Labels treated as NON-SLANG: O

By default, runs analysis on both:
  - test_predictions.csv                → error_analysis/bert_bio_test_report.txt
  - generalization_test_predictions.csv → error_analysis/bert_bio_generalization_report.txt

Usage:
  python error_analysis_BERT_bio.py                        # runs both default files
  python error_analysis_BERT_bio.py --csv path/to/file.csv --out path/to/report.txt
"""

import argparse
import os
import pandas as pd
from collections import defaultdict
from seqeval.metrics import classification_report

POLYSEMY_WORDS = [
    "fire", "sick", "slay", "extra", "basic", "beta",
    "squad", "lore", "ghost", "mood", "woke", "caps", "big", "sus", "cringe"
]

DEFAULT_RUNS = [
    ("bert_bio/test_predictions.csv", "error_analysis/bert_bio_test_report.txt"),
    ("bert_bio/generalization_test_predictions.csv", "error_analysis/bert_bio_generalization_report.txt")
]


def is_slang_tag(tag):
    return tag.upper().startswith("B-") or tag.upper().startswith("I-")


def load_data(csv_path):
    if csv_path:
        df = pd.read_csv(csv_path)
        required = {"text", "gold_tags", "pred_tags"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        return df
    # if no csv file provided raise error
    raise ValueError(f"CSV file is missing")


def expand_to_tokens(df):
    records = []
    for sid, row in df.iterrows():
        tokens = str(row["text"]).split()
        gold = str(row["gold_tags"]).split()
        pred = str(row["pred_tags"]).split()
        n = min(len(tokens), len(gold), len(pred))
        if not (len(tokens) == len(gold) == len(pred)):
            print(
                f"  [WARN] Row {sid}: mismatch (tokens={len(tokens)}, gold={len(gold)}, pred={len(pred)}) — using first {n}")
        for i in range(n):
            records.append({
                "sentence_id": sid,
                "sentence": row["text"],
                "token": tokens[i],
                "gold_tag": gold[i],
                "pred_tag": pred[i],
                "gold_label": int(is_slang_tag(gold[i])),
                "pred_label": int(is_slang_tag(pred[i]))
            })
    return pd.DataFrame(records)


def classify_tokens(tdf):
    buckets = {"TP": [], "FP": [], "TN": [], "FN": []}
    for _, row in tdf.iterrows():
        g, p = row["gold_label"], row["pred_label"]
        if g == 1 and p == 1:
            buckets["TP"].append(row)
        elif g == 0 and p == 1:
            buckets["FP"].append(row)
        elif g == 0 and p == 0:
            buckets["TN"].append(row)
        elif g == 1 and p == 0:
            buckets["FN"].append(row)
    return {k: pd.DataFrame(v) for k, v in buckets.items()}


def fmt_examples(rows, n=10):
    if rows.empty:
        return "  (none)\n"
    lines = []
    for _, r in rows.head(n).iterrows():
        lines.append(
            f"  token: {str(r['token']):<18} "
            f"gold={r['gold_tag']:<12} pred={r['pred_tag']:<12} "
            f'| "{r["sentence"]}"'
        )
    return "\n".join(lines) + "\n"


def sentence_level_section(df):
    lines = ["─" * 70, "SENTENCE-LEVEL EXACT MATCH", "─" * 70]
    if "exact_match" in df.columns:
        total = len(df)
        correct = df["exact_match"].astype(str).str.lower().eq("true").sum()
        lines.append(f"  Exact matches : {correct} / {total}  ({correct / total * 100:.1f}%)")
    else:
        lines.append("  (no exact_match column in CSV)")
    lines.append("")
    return "\n".join(lines)


def span_level_section(df):
    lines = ["─" * 70, "SPAN-LEVEL METRICS", "─" * 70]
    y_true = []
    y_pred = []
    for _, row in df.iterrows():
        y_true.append(str(row['gold_tags']).split())
        y_pred.append(str(row['pred_tags']).split())
    lines.append(classification_report(y_true, y_pred))
    lines.append("")
    return "\n".join(lines)


def polysemy_section(tdf):
    lines = [
        "=" * 70, "POLYSEMOUS WORD ANALYSIS", "=" * 70,
        f"Tracking {len(POLYSEMY_WORDS)} words: " + ", ".join(POLYSEMY_WORDS), "",
    ]
    tdf = tdf.copy()
    tdf["token_lower"] = tdf["token"].astype(str).str.strip().str.lower()
    poly_set = {w.lower() for w in POLYSEMY_WORDS}
    poly_df = tdf[tdf["token_lower"].isin(poly_set)]

    if poly_df.empty:
        lines.append("  No polysemous words found in dataset.\n")
        return "\n".join(lines)

    word_stats = defaultdict(lambda: defaultdict(list))
    for _, row in poly_df.iterrows():
        w = row["token_lower"]
        g, p = row["gold_label"], row["pred_label"]
        if g == 1 and p == 1:
            cat = "TP"
        elif g == 0 and p == 1:
            cat = "FP"
        elif g == 0 and p == 0:
            cat = "TN"
        else:
            cat = "FN"
        word_stats[w][cat].append(row)

    lines.append(f"  {'Word':<12} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  Verdict")
    lines.append("  " + "-" * 55)
    for word in POLYSEMY_WORDS:
        w = word.lower()
        if w not in word_stats:
            lines.append(f"  {word:<12} {'—':>5} {'—':>5} {'—':>5} {'—':>5}  not in dataset")
            continue
        counts = {c: len(word_stats[w][c]) for c in ("TP", "FP", "TN", "FN")}
        verdict = "✗ errors" if (counts["FP"] + counts["FN"]) > 0 else "✓ correct"
        lines.append(f"  {word:<12} {counts['TP']:>5} {counts['FP']:>5} {counts['TN']:>5} {counts['FN']:>5}  {verdict}")

    lines += ["", "Detailed token rows for polysemous words:", ""]
    for word in POLYSEMY_WORDS:
        w = word.lower()
        if w not in word_stats:
            continue
        lines.append(f"  [{word}]")
        for cat in ("TP", "FP", "TN", "FN"):
            for r in word_stats[w][cat]:
                lines.append(
                    f"    {cat}  token={str(r['token']):<12} "
                    f"gold={r['gold_tag']:<12} pred={r['pred_tag']:<12} "
                    f'| "{r["sentence"]}"'
                )
        lines.append("")
    return "\n".join(lines)


def build_report(df, source_label, n_examples=10):
    tdf = expand_to_tokens(df)
    buckets = classify_tokens(tdf)

    tp, fp = len(buckets["TP"]), len(buckets["FP"])
    tn, fn = len(buckets["TN"]), len(buckets["FN"])
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    out = [
        "=" * 70,
        "SLANG TOKEN CLASSIFICATION — ERROR ANALYSIS REPORT  (BIO format)",
        "=" * 70,
        f"Source        : {source_label}",
        f"Sentences     : {len(df)}",
        f"Total tokens  : {total}",
        "",
        "─" * 70, "TOKEN-LEVEL CONFUSION MATRIX", "─" * 70,
        f"  True  Positives (TP) — slang token correctly identified   : {tp:>6}",
        f"  False Positives (FP) — non-slang token called slang       : {fp:>6}",
        f"  True  Negatives (TN) — non-slang token correctly rejected : {tn:>6}",
        f"  False Negatives (FN) — slang token missed by model        : {fn:>6}",
        "",
        "─" * 70, "TOKEN-LEVEL METRICS", "─" * 70,
        f"  Accuracy  : {accuracy:.4f}",
        f"  Precision : {precision:.4f}",
        f"  Recall    : {recall:.4f}",
        f"  F1 Score  : {f1:.4f}",
        "",
        span_level_section(df),
        sentence_level_section(df),
    ]

    DEFS = {
        "TP": "True  Positives — slang token correctly identified",
        "FP": "False Positives — non-slang token incorrectly labelled as slang",
        "TN": "True  Negatives — non-slang token correctly rejected",
        "FN": "False Negatives — slang token the model failed to catch",
    }
    for cat in ("TP", "FP", "TN", "FN"):
        out += [
            "─" * 70,
            f"{cat} EXAMPLES  ({DEFS[cat]})",
            f"  Count: {len(buckets[cat])}",
            "",
            fmt_examples(buckets[cat], n=n_examples),
        ]

    out += ["", polysemy_section(tdf), "=" * 70, "END OF REPORT", "=" * 70]
    return "\n".join(out)


def run_single(csv_path, out_path, n_examples=10):
    print(f"\n{'=' * 70}")
    print(f"Processing : {csv_path}")
    print(f"Output     : {out_path}")
    print(f"{'=' * 70}")
    df = load_data(csv_path)
    report = build_report(df, source_label=csv_path, n_examples=n_examples)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n Report written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Slang BIO-tag classifier error analysis")
    parser.add_argument("--csv", default=None,
                        help="Path to a single predictions CSV. If omitted, runs both default files.")
    parser.add_argument("--out", default=None,
                        help="Output path for single-file mode (requires --csv).")
    parser.add_argument("--examples", type=int, default=10,
                        help="Max examples shown per bucket (default: 10)")
    args = parser.parse_args()

    if args.csv:
        # Single-file mode
        out_path = args.out if args.out else "error_analysis/bert_bio_report.txt"
        run_single(args.csv, out_path, n_examples=args.examples)
    else:
        # Default mode: run both files
        for csv_path, out_path in DEFAULT_RUNS:
            run_single(csv_path, out_path, n_examples=args.examples)


if __name__ == "__main__":
    main()
