# -*- coding: utf-8 -*-
"""
error_analysis_baseline.py

Error analysis for dictionary and TF-IDF baseline models.
Identifies false positives/negatives caused by polysemy (e.g., "fire", "sick", "slay").
Analyzes per-model error patterns and documents with examples.

Usage:
    python error_analysis_baseline.py
    python error_analysis_baseline.py --train output/train.csv --test output/test.csv \
        --mlbtrio data/mlbtrio_cleaned.csv --output_dir error_analysis

Output:
    error_analysis/baseline_report.txt
"""

import argparse
import os
import re
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

NOT_SLANG = "NOT_SLANG"
SLANG = "SLANG"
RANDOM_SEED = 42

POLYSEMY_WORDS = [
    "fire", "sick", "slay", "extra", "basic", "beta",
    "squad", "lore", "ghost", "mood", "woke", "caps", "big", "sus", "cringe",
]

# ── Reproduced from dictionary_baseline.py — keep in sync ────────────────────

def normalize(slang_term):
    slang_term = slang_term.lower().strip()
    slang_term = re.sub(r"\s+", " ", slang_term)
    return slang_term


def build_slang_list(df_mlbtrio, slang_col="Slang"):
    slang_list = sorted(
        set(normalize(t) for t in df_mlbtrio[slang_col]),
        key=len,
        reverse=True,  # longest first so multi-word phrases match before single words
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


# ── Reproduced from tfidf_baseline.py — keep in sync ─────────────────────────

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


# ── New helpers ───────────────────────────────────────────────────────────────

def word_is_present(word, text):
    """True iff `word` appears as a complete token in text (case-insensitive).
    Uses \\b word-boundary to avoid subword hits like 'fire' in 'Firefox'."""
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, str(text).lower()))


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def get_error_buckets(df_test, pred_col):
    """Return (tp_df, tn_df, fp_df, fn_df) based on gold label vs. pred_col."""
    gold = df_test["label"]
    pred = df_test[pred_col]
    tp = df_test[(gold == SLANG)     & (pred == SLANG)]
    tn = df_test[(gold == NOT_SLANG) & (pred == NOT_SLANG)]
    fp = df_test[(gold == NOT_SLANG) & (pred == SLANG)]     # false alarm: called SLANG, was NOT
    fn = df_test[(gold == SLANG)     & (pred == NOT_SLANG)] # missed slang: called NOT, was SLANG
    return tp, tn, fp, fn


def analyze_polysemy(df_test, pred_col, poly_words, max_examples=3):
    """For each polysemy word, return TP/TN/FP/FN counts and example sentences."""
    tp_df, tn_df, fp_df, fn_df = get_error_buckets(df_test, pred_col)

    def examples_from(sub_df, word, n=max_examples):
        mask = sub_df["text"].apply(lambda t: word_is_present(word, t))
        return sub_df["text"][mask].head(n).tolist()

    result = {}
    for word in poly_words:
        def count_in(sub_df, w=word):
            return int(sub_df["text"].apply(lambda t: word_is_present(w, t)).sum())

        result[word] = {
            "tp_count":    count_in(tp_df),
            "tn_count":    count_in(tn_df),
            "fp_count":    count_in(fp_df),
            "fn_count":    count_in(fn_df),
            "tp_examples": examples_from(tp_df, word),
            "tn_examples": examples_from(tn_df, word),
            "fp_examples": examples_from(fp_df, word),
            "fn_examples": examples_from(fn_df, word),
        }
    return result


def get_tfidf_coefficients(pipeline, words):
    """Return {word: coeff} for each word. None if word is OOV (min_df filtered)."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    vocab = vectorizer.vocabulary_
    coef = classifier.coef_[0]
    return {
        word: float(coef[vocab[word]]) if word in vocab else None
        for word in words
    }


# ── Report writing ────────────────────────────────────────────────────────────

SEP = "=" * 68
SEP2 = "-" * 68


def _fmt_examples(examples, indent="      "):
    if not examples:
        return indent + "(no examples found)\n"
    lines = ""
    for i, ex in enumerate(examples, 1):
        # truncate long lines for readability
        display = str(ex)
        if len(display) > 120:
            display = display[:117] + "..."
        lines += f"{indent}[{i}] {display}\n"
    return lines


def write_report(
    output_path,
    df_test,
    dict_metrics, tfidf_metrics,
    dict_errors, tfidf_errors,
    dict_poly, tfidf_poly,
    tfidf_coeffs,
    top_n=10,
    max_ex=3,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tp_d, tn_d, fp_d, fn_d = dict_errors
    tp_t, tn_t, fp_t, fn_t = tfidf_errors

    label_map = {SLANG: 1, NOT_SLANG: 0}
    n_total  = len(df_test)
    n_slang  = (df_test["label"] == SLANG).sum()
    n_not    = (df_test["label"] == NOT_SLANG).sum()

    with open(output_path, "w", encoding="utf-8") as f:

        # ── Header ────────────────────────────────────────────────────────────
        f.write(SEP + "\n")
        f.write("ERROR ANALYSIS REPORT — Gen Z Slang Detection (Baselines)\n")
        f.write(SEP + "\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test set  : output/test.csv  ({n_total} rows, "
                f"{n_slang} SLANG / {n_not} NOT_SLANG)\n")
        f.write(f"Models    : Dictionary Baseline, TF-IDF Baseline\n\n")

        # ── Section 1: Metrics ────────────────────────────────────────────────
        f.write(SEP + "\n")
        f.write("SECTION 1: MODEL METRICS SUMMARY (TEST SET)\n")
        f.write(SEP + "\n\n")
        f.write(f"{'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}\n")
        f.write(SEP2 + "\n")
        for name, m in [("Dictionary Baseline", dict_metrics), ("TF-IDF Baseline", tfidf_metrics)]:
            f.write(f"{name:<28} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
                    f"{m['recall']:>8.4f} {m['f1']:>8.4f}\n")
        f.write("\n")

        # ── Section 2: Confusion Matrices ─────────────────────────────────────
        f.write(SEP + "\n")
        f.write("SECTION 2: CONFUSION MATRICES\n")
        f.write(SEP + "\n\n")

        for name, tp, tn, fp, fn in [
            ("Dictionary Baseline", tp_d, tn_d, fp_d, fn_d),
            ("TF-IDF Baseline",     tp_t, tn_t, fp_t, fn_t),
        ]:
            f.write(f"--- {name} ---\n")
            f.write(f"{'':>28} {'Pred SLANG':>12} {'Pred NOT_SLANG':>15}\n")
            f.write(f"{'Actual SLANG':>28} {'TP='+str(len(tp)):>12} {'FN='+str(len(fn)):>15}\n")
            f.write(f"{'Actual NOT_SLANG':>28} {'FP='+str(len(fp)):>12} {'TN='+str(len(tn)):>15}\n")
            f.write(f"  False Positive Rate : {len(fp) / max(len(fp)+len(tn), 1):.3f}  "
                    f"(predicted SLANG when actually NOT_SLANG)\n")
            f.write(f"  False Negative Rate : {len(fn) / max(len(fn)+len(tp), 1):.3f}  "
                    f"(predicted NOT_SLANG when actually SLANG)\n\n")

        # ── Section 3: Top-N FP / FN Examples ────────────────────────────────
        f.write(SEP + "\n")
        f.write(f"SECTION 3: TOP-{top_n} FALSE POSITIVES AND FALSE NEGATIVES\n")
        f.write(SEP + "\n\n")

        for name, fp, fn in [
            ("Dictionary Baseline", fp_d, fn_d),
            ("TF-IDF Baseline",     fp_t, fn_t),
        ]:
            f.write(f"--- {name}: Top {top_n} False Positives ---\n")
            f.write("    (Predicted SLANG, Actual NOT_SLANG)\n")
            if len(fp) == 0:
                f.write("    (none)\n")
            else:
                for i, row in enumerate(fp.head(top_n).itertuples(), 1):
                    display = str(row.text)
                    if len(display) > 110:
                        display = display[:107] + "..."
                    f.write(f"    [{i:2d}] {display}\n")
            f.write("\n")

            f.write(f"--- {name}: Top {top_n} False Negatives ---\n")
            f.write("    (Predicted NOT_SLANG, Actual SLANG)\n")
            if len(fn) == 0:
                f.write("    (none)\n")
            else:
                for i, row in enumerate(fn.head(top_n).itertuples(), 1):
                    display = str(row.text)
                    if len(display) > 110:
                        display = display[:107] + "..."
                    f.write(f"    [{i:2d}] {display}\n")
            f.write("\n")

        # ── Section 4: Polysemy Word Analysis ─────────────────────────────────
        f.write(SEP + "\n")
        f.write("SECTION 4: POLYSEMY WORD ANALYSIS\n")
        f.write(SEP + "\n")
        f.write("For each polysemous word: TP/TN/FP/FN counts and up to "
                f"{max_ex} example sentences.\n")
        f.write("FP = model predicted SLANG but label was NOT_SLANG (false alarm).\n")
        f.write("FN = model predicted NOT_SLANG but label was SLANG (missed slang).\n\n")

        for word in POLYSEMY_WORDS:
            da = dict_poly[word]
            ta = tfidf_poly[word]
            coeff = tfidf_coeffs.get(word)
            coeff_str = f"{coeff:+.4f}" if coeff is not None else "N/A (OOV)"

            f.write(f"{'─'*68}\n")
            f.write(f'Word: "{word}"  |  TF-IDF coeff: {coeff_str}\n')
            f.write(f"{'─'*68}\n\n")

            # Dictionary model
            f.write(f"  DICTIONARY BASELINE\n")
            f.write(f"    Counts: TP={da['tp_count']}  TN={da['tn_count']}  "
                    f"FP={da['fp_count']}  FN={da['fn_count']}\n")
            if da["fp_count"] > 0:
                f.write(f"    False Positives (predicted SLANG, was NOT_SLANG):\n")
                f.write(_fmt_examples(da["fp_examples"]))
            if da["fn_count"] > 0:
                f.write(f"    False Negatives (predicted NOT_SLANG, was SLANG):\n")
                f.write(_fmt_examples(da["fn_examples"]))
            if da["tp_count"] > 0:
                f.write(f"    True Positives (correctly SLANG):\n")
                f.write(_fmt_examples(da["tp_examples"]))
            if da["tn_count"] > 0:
                f.write(f"    True Negatives (correctly NOT_SLANG):\n")
                f.write(_fmt_examples(da["tn_examples"]))
            f.write("\n")

            # TF-IDF model
            f.write(f"  TF-IDF BASELINE  (coeff {coeff_str} → "
                    f"{'biased toward NOT_SLANG' if coeff is not None and coeff < 0 else 'biased toward SLANG' if coeff is not None else 'OOV'})\n")
            f.write(f"    Counts: TP={ta['tp_count']}  TN={ta['tn_count']}  "
                    f"FP={ta['fp_count']}  FN={ta['fn_count']}\n")
            if ta["fp_count"] > 0:
                f.write(f"    False Positives (predicted SLANG, was NOT_SLANG):\n")
                f.write(_fmt_examples(ta["fp_examples"]))
            if ta["fn_count"] > 0:
                f.write(f"    False Negatives (predicted NOT_SLANG, was SLANG):\n")
                f.write(_fmt_examples(ta["fn_examples"]))
            if ta["tp_count"] > 0:
                f.write(f"    True Positives (correctly SLANG):\n")
                f.write(_fmt_examples(ta["tp_examples"]))
            if ta["tn_count"] > 0:
                f.write(f"    True Negatives (correctly NOT_SLANG):\n")
                f.write(_fmt_examples(ta["tn_examples"]))
            f.write("\n")

        # ── Section 5: Cross-Model Comparison ────────────────────────────────
        f.write(SEP + "\n")
        f.write("SECTION 5: CROSS-MODEL ERROR COMPARISON\n")
        f.write(SEP + "\n\n")

        # Index-based set operations
        both_fp  = fp_d.index.intersection(fp_t.index)
        both_fn  = fn_d.index.intersection(fn_t.index)
        only_fp_d = fp_d.index.difference(fp_t.index)   # dict FP, tfidf TN
        only_fn_d = fn_d.index.difference(fn_t.index)   # dict FN, tfidf TP
        only_fp_t = fp_t.index.difference(fp_d.index)   # tfidf FP, dict TN
        only_fn_t = fn_t.index.difference(fn_d.index)   # tfidf FN, dict TP

        f.write(f"Both models wrong (agree on error):\n")
        f.write(f"  Both predict SLANG when NOT_SLANG (shared FP): {len(both_fp)}\n")
        f.write(f"  Both predict NOT_SLANG when SLANG (shared FN): {len(both_fn)}\n\n")

        f.write(f"Dictionary wrong, TF-IDF correct:\n")
        f.write(f"  Dict FP / TF-IDF TN : {len(only_fp_d)}  "
                f"(dict false alarms that TF-IDF caught)\n")
        f.write(f"  Dict FN / TF-IDF TP : {len(only_fn_d)}  "
                f"(slang dict missed but TF-IDF found)\n\n")

        f.write(f"TF-IDF wrong, Dictionary correct:\n")
        f.write(f"  TF-IDF FP / Dict TN : {len(only_fp_t)}  "
                f"(TF-IDF false alarms that dict avoided)\n")
        f.write(f"  TF-IDF FN / Dict TP : {len(only_fn_t)}  "
                f"(slang TF-IDF missed but dict found)\n\n")

        # Sample disagreement examples
        sample_n = 8
        f.write(f"--- Sample sentences where models DISAGREE (up to {sample_n} each) ---\n\n")

        f.write(f"Dict=SLANG, TF-IDF=NOT_SLANG (dict false alarm):\n")
        for i, idx in enumerate(only_fp_d[:sample_n], 1):
            display = str(df_test.loc[idx, "text"])
            if len(display) > 105:
                display = display[:102] + "..."
            gold = df_test.loc[idx, "label"]
            f.write(f"  [{i}] gold={gold}  |  {display}\n")
        f.write("\n")

        f.write(f"Dict=NOT_SLANG, TF-IDF=SLANG (tfidf false alarm):\n")
        for i, idx in enumerate(only_fp_t[:sample_n], 1):
            display = str(df_test.loc[idx, "text"])
            if len(display) > 105:
                display = display[:102] + "..."
            gold = df_test.loc[idx, "label"]
            f.write(f"  [{i}] gold={gold}  |  {display}\n")
        f.write("\n")

        f.write(f"Dict missed SLANG, TF-IDF found it:\n")
        for i, idx in enumerate(only_fn_d[:sample_n], 1):
            display = str(df_test.loc[idx, "text"])
            if len(display) > 105:
                display = display[:102] + "..."
            f.write(f"  [{i}] {display}\n")
        f.write("\n")

        f.write(f"TF-IDF missed SLANG, Dict found it:\n")
        for i, idx in enumerate(only_fn_t[:sample_n], 1):
            display = str(df_test.loc[idx, "text"])
            if len(display) > 105:
                display = display[:102] + "..."
            f.write(f"  [{i}] {display}\n")
        f.write("\n")

        # ── Section 6: Polysemy Summary Table ────────────────────────────────
        f.write(SEP + "\n")
        f.write("SECTION 6: POLYSEMY SUMMARY TABLE\n")
        f.write(SEP + "\n\n")
        f.write("Negative TF-IDF coefficients indicate words the model learned as NOT_SLANG\n")
        f.write("signals — these cause false negatives when the word is used as slang.\n")
        f.write("Positive TF-IDF coefficients can cause false positives on literal uses.\n\n")

        hdr = f"{'Word':<10} {'TF-IDF Coeff':>13} {'Dict FP':>8} {'Dict FN':>8} {'TFIDF FP':>9} {'TFIDF FN':>9}   Error Pattern"
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")

        for word in POLYSEMY_WORDS:
            da = dict_poly[word]
            ta = tfidf_poly[word]
            coeff = tfidf_coeffs.get(word)
            coeff_str = f"{coeff:+.4f}" if coeff is not None else "     N/A"

            # Determine dominant error pattern
            if coeff is not None and coeff < -2.0 and ta["fn_count"] > 0:
                pattern = "FN-prone: slang meaning suppressed by negative coeff"
            elif coeff is not None and coeff > 2.0 and ta["fp_count"] > 0:
                pattern = "FP-prone: literal use triggers high SLANG coeff"
            elif da["fp_count"] > da["fn_count"]:
                pattern = "Dict over-predicts SLANG (in slang lexicon)"
            elif da["fn_count"] > da["fp_count"]:
                pattern = "Dict under-predicts SLANG (missing slang form)"
            else:
                pattern = "Balanced / low occurrence"

            f.write(
                f"{word:<10} {coeff_str:>13} {da['fp_count']:>8} {da['fn_count']:>8} "
                f"{ta['fp_count']:>9} {ta['fn_count']:>9}   {pattern}\n"
            )

        f.write("\n")
        f.write(SEP + "\n")
        f.write("END OF REPORT\n")
        f.write(SEP + "\n")

    print(f"Report saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlbtrio",    default="data/mlbtrio_cleaned.csv")
    parser.add_argument("--train",      default="output/train.csv")
    parser.add_argument("--test",       default="output/test.csv")
    parser.add_argument("--output_dir", default="error_analysis")
    parser.add_argument("--top_n",      type=int, default=10)
    parser.add_argument("--max_ex",     type=int, default=3)
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df_mlbtrio = pd.read_csv(args.mlbtrio)
    df_train   = pd.read_csv(args.train)
    df_test    = pd.read_csv(args.test)

    # ── Dictionary model ──────────────────────────────────────────────────────
    print("Building slang list...")
    slang_list = build_slang_list(df_mlbtrio)
    print(f"  Slang list size: {len(slang_list)} terms")

    print("Running Dictionary model on test set...")
    _, dict_pred_labels = make_predictions(df_test["text"].tolist(), slang_list)
    df_test["dict_pred"] = dict_pred_labels

    # ── TF-IDF model ──────────────────────────────────────────────────────────
    print("Training TF-IDF model on train set...")
    pipeline = build_pipeline()
    pipeline.fit(
        df_train["text"].fillna("").astype(str),
        df_train["label"],
    )
    print("Running TF-IDF model on test set...")
    df_test["tfidf_pred"] = pipeline.predict(
        df_test["text"].fillna("").astype(str)
    ).tolist()

    # ── Metrics ───────────────────────────────────────────────────────────────
    label_map = {SLANG: 1, NOT_SLANG: 0}
    y_true = df_test["label"].map(label_map).tolist()

    dict_metrics  = compute_metrics(y_true, df_test["dict_pred"].map(label_map).tolist())
    tfidf_metrics = compute_metrics(y_true, df_test["tfidf_pred"].map(label_map).tolist())

    print(f"  Dict  — Acc={dict_metrics['accuracy']:.4f}  F1={dict_metrics['f1']:.4f}")
    print(f"  TF-IDF— Acc={tfidf_metrics['accuracy']:.4f}  F1={tfidf_metrics['f1']:.4f}")

    # ── Error buckets ──────────────────────────────────────────────────────────
    print("Computing error buckets...")
    dict_errors  = get_error_buckets(df_test, "dict_pred")
    tfidf_errors = get_error_buckets(df_test, "tfidf_pred")

    tp_d, tn_d, fp_d, fn_d = dict_errors
    tp_t, tn_t, fp_t, fn_t = tfidf_errors
    print(f"  Dict  FP={len(fp_d)}  FN={len(fn_d)}")
    print(f"  TF-IDF FP={len(fp_t)}  FN={len(fn_t)}")

    # ── Polysemy analysis ─────────────────────────────────────────────────────
    print("Analyzing polysemy words...")
    dict_poly  = analyze_polysemy(df_test, "dict_pred",  POLYSEMY_WORDS, args.max_ex)
    tfidf_poly = analyze_polysemy(df_test, "tfidf_pred", POLYSEMY_WORDS, args.max_ex)

    tfidf_coeffs = get_tfidf_coefficients(pipeline, POLYSEMY_WORDS)

    # ── Write report ──────────────────────────────────────────────────────────
    output_path = os.path.join(args.output_dir, "baseline_report.txt")
    print(f"Writing report to {output_path}...")
    write_report(
        output_path,
        df_test,
        dict_metrics, tfidf_metrics,
        dict_errors,  tfidf_errors,
        dict_poly,    tfidf_poly,
        tfidf_coeffs,
        top_n=args.top_n,
        max_ex=args.max_ex,
    )


if __name__ == "__main__":
    main()
