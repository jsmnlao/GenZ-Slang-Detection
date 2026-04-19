# -*- coding: utf-8 -*-
"""
error_analysis_llm.py

Error analysis for LLM models (Qwen 2.5 Plus and DeepSeek-V3.2).
Identifies false positives/negatives caused by polysemy (e.g., "fire", "sick", "slay").
Analyzes per-model error patterns and documents with examples.

Usage:
    python error_analysis_llm.py
    python error_analysis_llm.py --output_dir error_analysis

    # Generalization set:
    python error_analysis_llm.py \
        --qwen_jsonl llm_evaluation/outputs/qwen_generalization/qwen-plus_sentence_cot_raw.jsonl \
        --deepseek_jsonl llm_evaluation/outputs/deepseek_generalization/deepseek-v3.2_sentence_cot_raw.jsonl \
        --output_dir error_analysis --suffix generalization

Output:
    error_analysis/llm_sentence_test_report.txt
    error_analysis/llm_sentence_generalization_report.txt
"""

import argparse
import json
import os
import re
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── Constants ─────────────────────────────────────────────────────────────────

NOT_SLANG = "NOT_SLANG"
SLANG     = "SLANG"

POLYSEMY_WORDS = [
    "fire", "sick", "slay", "extra", "basic", "beta",
    "squad", "lore", "ghost", "mood", "woke", "caps", "big", "sus", "cringe",
]

DEFAULT_QWEN_JSONL     = "llm_evaluation/outputs/qwen/qwen-plus_sentence_cot_raw.jsonl"
DEFAULT_DEEPSEEK_JSONL = "llm_evaluation/outputs/deepseek/deepseek-v3.2_sentence_cot_raw.jsonl"

SEP  = "=" * 70
SEP2 = "-" * 70

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_predictions(jsonl_path):
    """Load LLM predictions from a raw JSONL file.
    Drops rows where pred_label is not SLANG/NOT_SLANG (parse failures).
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df = df[df["pred_label"].isin([SLANG, NOT_SLANG])].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def word_is_present(word, text):
    """True iff `word` appears as a whole token in text (case-insensitive)."""
    return bool(re.search(r"\b" + re.escape(word) + r"\b", str(text).lower()))


def compute_metrics(y_true, y_pred):
    label_map = {SLANG: 1, NOT_SLANG: 0}
    yt = [label_map[y] for y in y_true]
    yp = [label_map[y] for y in y_pred]
    exact = sum(a == b for a, b in zip(y_true, y_pred))
    return {
        "accuracy":  accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall":    recall_score(yt, yp, zero_division=0),
        "f1":        f1_score(yt, yp, zero_division=0),
        "exact":     exact,
        "total":     len(y_true),
    }


def get_error_buckets(df):
    """Return (tp_df, tn_df, fp_df, fn_df)."""
    gold = df["gold_label"]
    pred = df["pred_label"]
    tp = df[(gold == SLANG)     & (pred == SLANG)]
    tn = df[(gold == NOT_SLANG) & (pred == NOT_SLANG)]
    fp = df[(gold == NOT_SLANG) & (pred == SLANG)]
    fn = df[(gold == SLANG)     & (pred == NOT_SLANG)]
    return tp, tn, fp, fn


def analyze_polysemy(df, poly_words, max_examples=3):
    """For each polysemy word return counts + example rows (with reasoning)."""
    tp_df, tn_df, fp_df, fn_df = get_error_buckets(df)

    def rows_with_word(sub_df, word, n=max_examples):
        mask = sub_df["text"].apply(lambda t: word_is_present(word, t))
        return sub_df[mask].head(n)

    def count_in(sub_df, word):
        if len(sub_df) == 0:
            return 0
        return int(sub_df["text"].fillna("").apply(lambda t: word_is_present(word, t)).astype(int).sum())

    result = {}
    for word in poly_words:
        result[word] = {
            "tp_count": count_in(tp_df, word),
            "tn_count": count_in(tn_df, word),
            "fp_count": count_in(fp_df, word),
            "fn_count": count_in(fn_df, word),
            "tp_rows":  rows_with_word(tp_df, word),
            "tn_rows":  rows_with_word(tn_df, word),
            "fp_rows":  rows_with_word(fp_df, word),
            "fn_rows":  rows_with_word(fn_df, word),
        }
    return result


# ── Formatting helpers ────────────────────────────────────────────────────────

def _example_line(bucket, gold, pred, text, reasoning=None, max_text=110, max_reason=180):
    """Single row in BERT-style format, with optional indented reasoning."""
    display = str(text)
    if len(display) > max_text:
        display = display[:max_text - 3] + "..."
    line = f"  {bucket:<2}  gold={gold:<10}  pred={pred:<10}  | \"{display}\"\n"
    if reasoning and str(reasoning) not in ("", "nan", "None"):
        r = str(reasoning)
        if len(r) > max_reason:
            r = r[:max_reason - 3] + "..."
        line += f"       Reasoning: {r}\n"
    return line


def _write_bucket_block(f, label, rows, gold_lbl, pred_lbl,
                        show_reasoning=False, max_rows=10):
    f.write(SEP2 + "\n")
    f.write(f"{label}\n")
    f.write(SEP2 + "\n")
    f.write(f"  Count: {len(rows)}\n\n")
    for row in rows.head(max_rows).itertuples():
        reasoning = str(getattr(row, "reasoning", "")) if show_reasoning else None
        f.write(_example_line(
            label[:2], gold_lbl, pred_lbl, row.text, reasoning
        ))
    f.write("\n")


# ── Per-model report section ──────────────────────────────────────────────────

def _write_model_section(f, model_name, source_path, df, metrics,
                         errors, poly, top_n=10):
    tp, tn, fp, fn = errors

    # ── Header ────────────────────────────────────────────────────────────
    f.write(SEP + "\n")
    f.write("SLANG SENTENCE CLASSIFICATION - ERROR ANALYSIS REPORT\n")
    f.write(SEP + "\n")
    f.write(f"Source        : {source_path}\n")
    f.write(f"Model         : {model_name}\n")
    f.write(f"Sentences     : {len(df)}\n\n")

    # ── Confusion matrix ──────────────────────────────────────────────────
    f.write(SEP2 + "\n")
    f.write("SENTENCE-LEVEL CONFUSION MATRIX\n")
    f.write(SEP2 + "\n")
    f.write(f"  True  Positives (TP) - slang sentence correctly identified   : {len(tp):>6}\n")
    f.write(f"  False Positives (FP) - non-slang sentence called slang       : {len(fp):>6}\n")
    f.write(f"  True  Negatives (TN) - non-slang sentence correctly rejected : {len(tn):>6}\n")
    f.write(f"  False Negatives (FN) - slang sentence missed by model        : {len(fn):>6}\n\n")

    # ── Metrics ───────────────────────────────────────────────────────────
    f.write(SEP2 + "\n")
    f.write("SENTENCE-LEVEL METRICS\n")
    f.write(SEP2 + "\n")
    f.write(f"  Accuracy  : {metrics['accuracy']:.4f}\n")
    f.write(f"  Precision : {metrics['precision']:.4f}\n")
    f.write(f"  Recall    : {metrics['recall']:.4f}\n")
    f.write(f"  F1 Score  : {metrics['f1']:.4f}\n\n")

    # ── Exact match ───────────────────────────────────────────────────────
    f.write(SEP2 + "\n")
    f.write("SENTENCE-LEVEL EXACT MATCH\n")
    f.write(SEP2 + "\n")
    f.write(f"  Exact matches : {metrics['exact']} / {metrics['total']}"
            f"  ({metrics['exact'] / metrics['total']:.1%})\n\n")

    # ── TP / FP / TN / FN examples ────────────────────────────────────────
    _write_bucket_block(f,
        "TP EXAMPLES  (True  Positives - slang sentence correctly identified)",
        tp, SLANG, SLANG, show_reasoning=False, max_rows=top_n)
    _write_bucket_block(f,
        "FP EXAMPLES  (False Positives - non-slang sentence incorrectly labelled as slang)",
        fp, NOT_SLANG, SLANG, show_reasoning=True, max_rows=top_n)
    _write_bucket_block(f,
        "TN EXAMPLES  (True  Negatives - non-slang sentence correctly rejected)",
        tn, NOT_SLANG, NOT_SLANG, show_reasoning=False, max_rows=top_n)
    _write_bucket_block(f,
        "FN EXAMPLES  (False Negatives - slang sentence the model failed to catch)",
        fn, SLANG, NOT_SLANG, show_reasoning=True, max_rows=top_n)

    # ── Polysemy summary table ────────────────────────────────────────────
    f.write("\n")
    f.write(SEP + "\n")
    f.write("POLYSEMOUS WORD ANALYSIS\n")
    f.write(SEP + "\n")
    f.write(f"Tracking {len(POLYSEMY_WORDS)} words: {', '.join(POLYSEMY_WORDS)}\n\n")

    f.write(f"  {'Word':<16} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  Verdict\n")
    f.write(f"  {'─' * 53}\n")
    for word in POLYSEMY_WORDS:
        wa = poly[word]
        has_errors = (wa['fp_count'] + wa['fn_count']) > 0
        verdict = "x errors" if has_errors else "ok correct"
        f.write(f"  {word:<16} {wa['tp_count']:>5} {wa['fp_count']:>5}"
                f" {wa['tn_count']:>5} {wa['fn_count']:>5}  {verdict}\n")
    f.write("\n")

    # ── Polysemy detailed rows ────────────────────────────────────────────
    f.write("Detailed sentence rows for polysemous words:\n\n")
    for word in POLYSEMY_WORDS:
        wa = poly[word]
        total = wa['tp_count'] + wa['fp_count'] + wa['tn_count'] + wa['fn_count']
        if total == 0:
            continue
        errors = wa['fp_count'] + wa['fn_count']
        f.write(f"  [{word}]\n")
        for _, row in wa["tp_rows"].iterrows():
            f.write(_example_line("TP", SLANG, SLANG, row["text"]))
        for _, row in wa["fp_rows"].iterrows():
            f.write(_example_line("FP", NOT_SLANG, SLANG, row["text"],
                                  row.get("reasoning")))
        for _, row in wa["tn_rows"].iterrows():
            f.write(_example_line("TN", NOT_SLANG, NOT_SLANG, row["text"]))
        for _, row in wa["fn_rows"].iterrows():
            f.write(_example_line("FN", SLANG, NOT_SLANG, row["text"],
                                  row.get("reasoning")))
        f.write("\n")


# ── Cross-model section ───────────────────────────────────────────────────────

def _write_cross_model_section(f, df_qwen, df_deepseek,
                                qwen_errors, deepseek_errors, sample_n=5):
    tp_q, tn_q, fp_q, fn_q = qwen_errors
    tp_d, tn_d, fp_d, fn_d = deepseek_errors

    fp_q_idx = set(fp_q["idx"].tolist()) if "idx" in fp_q.columns else set(fp_q.index)
    fn_q_idx = set(fn_q["idx"].tolist()) if "idx" in fn_q.columns else set(fn_q.index)
    fp_d_idx = set(fp_d["idx"].tolist()) if "idx" in fp_d.columns else set(fp_d.index)
    fn_d_idx = set(fn_d["idx"].tolist()) if "idx" in fn_d.columns else set(fn_d.index)

    both_fp   = fp_q_idx & fp_d_idx
    both_fn   = fn_q_idx & fn_d_idx
    only_fp_q = fp_q_idx - fp_d_idx
    only_fn_q = fn_q_idx - fn_d_idx
    only_fp_d = fp_d_idx - fp_q_idx
    only_fn_d = fn_d_idx - fn_q_idx

    f.write(SEP + "\n")
    f.write("CROSS-MODEL ERROR COMPARISON (Qwen 2.5 Plus vs DeepSeek-V3.2)\n")
    f.write(SEP + "\n\n")

    f.write("Both models wrong (agree on error):\n")
    f.write(f"  Both predict SLANG when NOT_SLANG (shared FP) : {len(both_fp)}\n")
    f.write(f"  Both predict NOT_SLANG when SLANG (shared FN) : {len(both_fn)}\n\n")

    f.write("Qwen wrong, DeepSeek correct:\n")
    f.write(f"  Qwen FP / DeepSeek TN : {len(only_fp_q)}"
            f"  (Qwen false alarms that DeepSeek avoided)\n")
    f.write(f"  Qwen FN / DeepSeek TP : {len(only_fn_q)}"
            f"  (slang Qwen missed but DeepSeek found)\n\n")

    f.write("DeepSeek wrong, Qwen correct:\n")
    f.write(f"  DeepSeek FP / Qwen TN : {len(only_fp_d)}"
            f"  (DeepSeek false alarms that Qwen avoided)\n")
    f.write(f"  DeepSeek FN / Qwen TP : {len(only_fn_d)}"
            f"  (slang DeepSeek missed but Qwen found)\n\n")

    # Sample shared errors with reasoning
    f.write(SEP2 + "\n")
    f.write(f"Shared False Positives (both predicted SLANG, gold NOT_SLANG)"
            f" — up to {sample_n}\n")
    f.write(SEP2 + "\n\n")
    for i, idx in enumerate(sorted(both_fp)[:sample_n], 1):
        q_row = df_qwen[df_qwen["idx"] == idx]
        d_row = df_deepseek[df_deepseek["idx"] == idx]
        if len(q_row) == 0:
            continue
        text = str(q_row.iloc[0]["text"])
        if len(text) > 107:
            text = text[:104] + "..."
        q_r = str(q_row.iloc[0].get("reasoning", ""))[:180]
        d_r = str(d_row.iloc[0].get("reasoning", ""))[:180] if len(d_row) > 0 else ""
        f.write(f"  [{i}] gold=NOT_SLANG  pred=SLANG  | \"{text}\"\n")
        f.write(f"       Qwen reasoning:     {q_r}\n")
        f.write(f"       DeepSeek reasoning: {d_r}\n\n")

    f.write(SEP2 + "\n")
    f.write(f"Shared False Negatives (both predicted NOT_SLANG, gold SLANG)"
            f" — up to {sample_n}\n")
    f.write(SEP2 + "\n\n")
    for i, idx in enumerate(sorted(both_fn)[:sample_n], 1):
        q_row = df_qwen[df_qwen["idx"] == idx]
        d_row = df_deepseek[df_deepseek["idx"] == idx]
        if len(q_row) == 0:
            continue
        text = str(q_row.iloc[0]["text"])
        if len(text) > 107:
            text = text[:104] + "..."
        q_r = str(q_row.iloc[0].get("reasoning", ""))[:180]
        d_r = str(d_row.iloc[0].get("reasoning", ""))[:180] if len(d_row) > 0 else ""
        f.write(f"  [{i}] gold=SLANG  pred=NOT_SLANG  | \"{text}\"\n")
        f.write(f"       Qwen reasoning:     {q_r}\n")
        f.write(f"       DeepSeek reasoning: {d_r}\n\n")

    f.write(SEP + "\n")
    f.write("END OF REPORT\n")
    f.write(SEP + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for Qwen and DeepSeek LLM models"
    )
    parser.add_argument("--qwen_jsonl",     default=DEFAULT_QWEN_JSONL)
    parser.add_argument("--deepseek_jsonl", default=DEFAULT_DEEPSEEK_JSONL)
    parser.add_argument("--output_dir",     default="error_analysis")
    parser.add_argument("--suffix",         default="test",
                        help="Report filename suffix: test or generalization")
    parser.add_argument("--top_n",  type=int, default=10)
    parser.add_argument("--max_ex", type=int, default=3)
    args = parser.parse_args()

    # Load
    print(f"Loading Qwen predictions from {args.qwen_jsonl}...")
    df_qwen = load_predictions(args.qwen_jsonl)
    print(f"  {len(df_qwen)} valid predictions")

    print(f"Loading DeepSeek predictions from {args.deepseek_jsonl}...")
    df_deepseek = load_predictions(args.deepseek_jsonl)
    print(f"  {len(df_deepseek)} valid predictions")

    # Metrics
    print("Computing metrics...")
    qwen_metrics     = compute_metrics(df_qwen["gold_label"].tolist(),
                                       df_qwen["pred_label"].tolist())
    deepseek_metrics = compute_metrics(df_deepseek["gold_label"].tolist(),
                                       df_deepseek["pred_label"].tolist())
    print(f"  Qwen     Acc={qwen_metrics['accuracy']:.4f}  F1={qwen_metrics['f1']:.4f}")
    print(f"  DeepSeek Acc={deepseek_metrics['accuracy']:.4f}  F1={deepseek_metrics['f1']:.4f}")

    # Error buckets
    print("Computing error buckets...")
    qwen_errors     = get_error_buckets(df_qwen)
    deepseek_errors = get_error_buckets(df_deepseek)
    tp_q, tn_q, fp_q, fn_q = qwen_errors
    tp_d, tn_d, fp_d, fn_d = deepseek_errors
    print(f"  Qwen     FP={len(fp_q)}  FN={len(fn_q)}")
    print(f"  DeepSeek FP={len(fp_d)}  FN={len(fn_d)}")

    # Polysemy
    print("Analyzing polysemy words...")
    qwen_poly     = analyze_polysemy(df_qwen,     POLYSEMY_WORDS, args.max_ex)
    deepseek_poly = analyze_polysemy(df_deepseek, POLYSEMY_WORDS, args.max_ex)

    # Write
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"llm_sentence_{args.suffix}_report.txt")
    print(f"Writing report to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        _write_model_section(
            f, "Qwen 2.5 Plus", args.qwen_jsonl,
            df_qwen, qwen_metrics, qwen_errors, qwen_poly, args.top_n
        )
        f.write("\n\n")
        _write_model_section(
            f, "DeepSeek-V3.2", args.deepseek_jsonl,
            df_deepseek, deepseek_metrics, deepseek_errors, deepseek_poly, args.top_n
        )
        f.write("\n\n")
        _write_cross_model_section(
            f, df_qwen, df_deepseek, qwen_errors, deepseek_errors
        )

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
