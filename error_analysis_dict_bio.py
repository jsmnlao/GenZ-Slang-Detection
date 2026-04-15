"""
Dictionary Baseline — Token-level BIO Error Analysis
=====================================================
Generates BIO predictions from the dictionary lookup baseline and runs
the same error analysis as error_analysis_bert_bio.py (reuses its
run_single / build_report functions directly).

Predictions are saved alongside the analysis so they can be inspected
or fed into other pipelines.

Usage:
  python error_analysis_dict_bio.py
  python error_analysis_dict_bio.py \
      --mlbtrio data/mlbtrio_cleaned.csv \
      --test output/test.csv \
      --generalization output/generalization_test.csv \
      --output_dir dictionary_baseline \
      --report_dir error_analysis

Output:
  dictionary_baseline/test_predictions.csv
  dictionary_baseline/generalization_test_predictions.csv
  error_analysis/dict_bio_test_report.txt
  error_analysis/dict_bio_generalization_report.txt
"""

import argparse
import os
import re
import subprocess
import sys

import pandas as pd

# ── Dictionary baseline helpers (kept in sync with dictionary_baseline.py) ────

def normalize(slang_term: str) -> str:
    slang_term = slang_term.lower().strip()
    slang_term = re.sub(r"\s+", " ", slang_term)
    return slang_term


def build_slang_list(df_mlbtrio: pd.DataFrame, slang_col: str = "Slang"):
    """Return slang terms sorted longest-first so multi-word phrases match first."""
    return sorted(
        set(normalize(t) for t in df_mlbtrio[slang_col]),
        key=len,
        reverse=True,
    )


def predict_bio_tags(text: str, patterns: list) -> list:
    tokens = str(text).split()
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


# ── Prediction generation ──────────────────────────────────────────────────────

def generate_predictions(split_path: str, slang_list: list) -> pd.DataFrame:
    """Load a split CSV and return a DataFrame with gold_tags / pred_tags columns."""
    df = pd.read_csv(split_path).dropna(subset=["tags"]).copy()
    text_col = "text" if "text" in df.columns else "sentence"
    df[text_col] = df[text_col].fillna("").astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)

    rows = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        gold_tags = str(row["tags"]).split()
        pred_tags = predict_bio_tags(text, slang_list)

        # skip sentences where lengths don't align (shouldn't happen)
        if len(gold_tags) != len(pred_tags):
            continue

        exact = gold_tags == pred_tags
        rows.append({
            "text":        text,
            "gold_tags":   " ".join(gold_tags),
            "pred_tags":   " ".join(pred_tags),
            "exact_match": exact,
        })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlbtrio",      default="data/mlbtrio_cleaned.csv")
    parser.add_argument("--test",         default="output/test.csv")
    parser.add_argument("--generalization", default="output/generalization_test.csv")
    parser.add_argument("--output_dir",   default="dictionary_baseline",
                        help="Where to save predictions CSVs")
    parser.add_argument("--report_dir",   default="error_analysis",
                        help="Where to save the text reports")
    parser.add_argument("--examples",     type=int, default=10)
    args = parser.parse_args()

    # Build slang list
    print("Loading slang list from", args.mlbtrio)
    df_mlbtrio = pd.read_csv(args.mlbtrio)
    slang_list = build_slang_list(df_mlbtrio)
    print(f"  {len(slang_list)} unique slang terms loaded")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    runs = [
        (args.test,           "test",               "dict_bio_test_report.txt"),
        (args.generalization, "generalization_test", "dict_bio_generalization_report.txt"),
    ]

    for split_path, split_name, report_filename in runs:
        if not split_path or not os.path.exists(split_path):
            print(f"Skipping {split_name}: {split_path} not found")
            continue

        print(f"\nGenerating dictionary predictions for {split_name} ({split_path})…")
        pred_df = generate_predictions(split_path, slang_list)
        print(f"  {len(pred_df)} sentences processed")

        # Save predictions CSV
        csv_path = os.path.join(args.output_dir, f"{split_name}_predictions.csv")
        pred_df.to_csv(csv_path, index=False)
        print(f"  Predictions saved to {csv_path}")

        # Run the shared analysis via the bert_bio analysis script
        report_path = os.path.join(args.report_dir, report_filename)
        analysis_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "error_analysis_bert_bio.py")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(
            [sys.executable, analysis_script,
             "--csv", csv_path,
             "--out", report_path,
             "--examples", str(args.examples)],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
