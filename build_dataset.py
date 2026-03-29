"""
build_dataset.py

Constructs a unified train/dev/test dataset from three source datasets.
Usage:
    python build_dataset.py
    python build_dataset.py --mlbtrio PATH --gen_alpha PATH --wsj PATH --output_dir DIR
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

TARGET_MLBTRIO_POS = 1500    # sample from ~1562 available
TARGET_GEN_TEST    = 1000    # held-out generalization set (not in main data)
# Gen Alpha pos for main = 10000 - TARGET_MLBTRIO_POS = 8500
# Gen Alpha neg = use all unique (~7209)
# WSJ neg = 10000 - len(ga_neg_unique) to hit exactly 10k negatives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_slang_spans(sentence, tags):
    """Return the set of lowercased slang spans extracted via BIO tags."""
    words = sentence.split()
    tag_list = tags.split()
    spans, current = [], []
    for word, tag in zip(words, tag_list):
        if tag == "B-SLANG":
            if current:
                spans.append(" ".join(current))
            current = [word]
        elif tag == "I-SLANG" and current:
            current.append(word)
        else:
            if current:
                spans.append(" ".join(current))
            current = []
    if current:
        spans.append(" ".join(current))
    return set(s.lower() for s in spans)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source", "label", "tags"])
        writer.writeheader()
        writer.writerows(rows)


def check_overlap(splits):
    """Verify zero sentence overlap across all split pairs. Returns True if clean."""
    seen = {}
    duplicates = []
    for name, rows in splits.items():
        for row in rows:
            key = row["text"]
            if key in seen:
                duplicates.append((key, seen[key], name))
            else:
                seen[key] = name
    if duplicates:
        print(f"  WARNING: {len(duplicates)} overlapping sentences found!")
        for text, s1, s2 in duplicates[:5]:
            print(f"    '{text[:60]}' in {s1} and {s2}")
        return False
    return True


def stats_block(name, rows):
    total = len(rows)
    label_counts  = defaultdict(int)
    source_counts = defaultdict(int)
    for r in rows:
        label_counts[r["label"]] += 1
        source_counts[r["source"]] += 1
    lines = [f"\n[{name}]  total={total}", "  Labels:"]
    for lbl in sorted(label_counts):
        lines.append(f"    {lbl}: {label_counts[lbl]}  ({100*label_counts[lbl]/total:.1f}%)")
    lines.append("  Sources:")
    for src in sorted(source_counts):
        lines.append(f"    {src}: {source_counts[src]}  ({100*source_counts[src]/total:.1f}%)")
    return "\n".join(lines)


def make_row(sentence, label, source, tags):
    return {"text": sentence, "label": label, "source": source, "tags": tags}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlbtrio",    default="data/mlbtrio_labeled.csv")
    parser.add_argument("--gen_alpha",  default="data/gen_alpha_tweets_labeled.csv")
    parser.add_argument("--wsj",        default="data/wsj_labeled.csv")
    parser.add_argument("--mlbtrio_bio", default="data/mlbtrio_bio_tagged.csv",
                        help="Used to extract slang term list for overlap detection")
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    # ---- Validate input files ------------------------------------------------
    missing = []
    for name, path in [("mlbtrio",    args.mlbtrio),
                        ("gen_alpha",  args.gen_alpha),
                        ("wsj",        args.wsj),
                        ("mlbtrio_bio", args.mlbtrio_bio)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: expected at '{path}'")
    if missing:
        print("ERROR: Missing source file(s):")
        print("\n".join(missing))
        sys.exit(1)

    rng = random.Random(RANDOM_SEED)

    # ---- Build MLBTrio slang term set ----------------------------------------
    print("Loading MLBTrio...")
    bio_rows = load_csv(args.mlbtrio_bio)
    mlbtrio_slang_terms = set()
    for r in bio_rows:
        term = r.get("slang", "").strip().lower()
        if term:
            mlbtrio_slang_terms.add(term)
        if r.get("tokens") and r.get("tags"):
            mlbtrio_slang_terms.update(extract_slang_spans(r["tokens"], r["tags"]))
    print(f"  MLBTrio slang terms: {len(mlbtrio_slang_terms)}")

    mlbtrio_rows = [r for r in load_csv(args.mlbtrio)
                    if r.get("binary_label", "").strip() == "1"]
    rng.shuffle(mlbtrio_rows)

    # ---- Load Gen Alpha Tweets -----------------------------------------------
    print("Loading Gen Alpha tweets...")
    ga_rows  = load_csv(args.gen_alpha)
    ga_pos   = [r for r in ga_rows if r.get("binary_label", "").strip() == "1"]
    ga_neg   = [r for r in ga_rows if r.get("binary_label", "").strip() == "0"]
    print(f"  Positives: {len(ga_pos)}, Negatives: {len(ga_neg)}")

    # Partition positives: MLBTrio-overlap first, then others
    rng.shuffle(ga_pos)
    overlap_pos, other_pos = [], []
    for r in ga_pos:
        spans = extract_slang_spans(r["sentence"], r.get("tags", ""))
        (overlap_pos if spans & mlbtrio_slang_terms else other_pos).append(r)
    print(f"  Gen Alpha pos overlapping with MLBTrio: {len(overlap_pos)}")
    print(f"  Gen Alpha pos non-overlapping:          {len(other_pos)}")

    # ---- Load WSJ ------------------------------------------------------------
    print("Loading WSJ sentences...")
    wsj_rows = [r for r in load_csv(args.wsj) if r.get("binary_label", "").strip() == "0"]
    rng.shuffle(wsj_rows)
    print(f"  WSJ available: {len(wsj_rows)}")

    # ---- Build exactly-counted, globally-deduplicated pools ------------------
    # global seen_texts prevents any sentence from appearing more than once
    # across all pools (main + gen_test)
    seen_texts = set()

    def add_unique(pool, rows, sentence_key, max_count=None):
        """Append unique rows (by sentence_key) into pool up to max_count."""
        for r in rows:
            if max_count is not None and len(pool) >= max_count:
                break
            txt = r[sentence_key]
            if txt not in seen_texts:
                seen_texts.add(txt)
                pool.append(r)

    # 1. MLBTrio positives (up to 1500)
    mlbtrio_pool = []
    add_unique(mlbtrio_pool, mlbtrio_rows, "sentence", TARGET_MLBTRIO_POS)
    mlbtrio_count = len(mlbtrio_pool)
    print(f"\n  MLBTrio unique selected: {mlbtrio_count}")

    # 2. Gen Alpha positives: overlap-first pool, then fill from others
    #    Need: (10000 - mlbtrio_count) for main  +  TARGET_GEN_TEST held out
    ga_pos_total_needed = (10000 - mlbtrio_count) + TARGET_GEN_TEST
    ga_pos_pool = []
    add_unique(ga_pos_pool, overlap_pos + other_pos, "sentence", ga_pos_total_needed)

    available = len(ga_pos_pool)
    if available < ga_pos_total_needed:
        print(f"  WARNING: only {available} unique gen_alpha positives available "
              f"(need {ga_pos_total_needed})")

    # Split: gen_test gets the tail (non-overlap examples), main gets the front
    ga_main_pos_count = min(10000 - mlbtrio_count, available - TARGET_GEN_TEST)
    ga_main_pos_rows  = ga_pos_pool[:ga_main_pos_count]
    ga_gen_test_rows  = ga_pos_pool[ga_main_pos_count:ga_main_pos_count + TARGET_GEN_TEST]
    print(f"  Gen Alpha → main positives:     {len(ga_main_pos_rows)}")
    print(f"  Gen Alpha → generalization test: {len(ga_gen_test_rows)}")

    # 3. Gen Alpha negatives (all unique)
    ga_neg_pool = []
    add_unique(ga_neg_pool, ga_neg, "sentence")
    print(f"  Gen Alpha negatives unique: {len(ga_neg_pool)}")

    # 4. WSJ negatives — take exactly enough to reach 10k negatives total
    wsj_needed = 10000 - len(ga_neg_pool)
    wsj_pool = []
    add_unique(wsj_pool, wsj_rows, "sentence", wsj_needed)
    print(f"  WSJ unique selected: {len(wsj_pool)} (needed {wsj_needed})")

    # ---- Assemble main dataset -----------------------------------------------
    all_main = (
        [make_row(r["sentence"], "SLANG",     "gen_z",     r["tags"]) for r in mlbtrio_pool]
      + [make_row(r["sentence"], "SLANG",     "gen_alpha", r["tags"]) for r in ga_main_pos_rows]
      + [make_row(r["sentence"], "NOT_SLANG", "gen_alpha", r["tags"]) for r in ga_neg_pool]
      + [make_row(r["sentence"], "NOT_SLANG", "wsj",       r["tags"]) for r in wsj_pool]
    )
    gen_test_data = [
        make_row(r["sentence"], "SLANG", "gen_alpha", r["tags"]) for r in ga_gen_test_rows
    ]

    total = len(all_main)
    pos_total = sum(1 for r in all_main if r["label"] == "SLANG")
    neg_total = total - pos_total
    print(f"\nMain dataset: {total} total  |  {pos_total} SLANG  |  {neg_total} NOT_SLANG")
    if total != 20000:
        print(f"  WARNING: expected 20000 examples, got {total}")

    # ---- Stratified 70 / 15 / 15 split via sklearn ---------------------------
    # Stratify on combined (label, source) key for proportional preservation
    strat_keys = [r["label"] + "_" + r["source"] for r in all_main]

    train, temp, _, temp_keys = train_test_split(
        all_main, strat_keys,
        test_size=0.30, random_state=RANDOM_SEED, stratify=strat_keys
    )
    dev, test = train_test_split(
        temp, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_keys
    )

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    print(f"Split → Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

    # ---- Validate no cross-split overlap -------------------------------------
    print("\nValidating split overlap...")
    splits_to_check = {"train": train, "dev": dev, "test": test, "gen_test": gen_test_data}
    no_overlap = check_overlap(splits_to_check)
    print(f"  No overlap: {no_overlap}")

    # ---- Write outputs -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "train.csv"),               train)
    write_csv(os.path.join(args.output_dir, "dev.csv"),                 dev)
    write_csv(os.path.join(args.output_dir, "test.csv"),                test)
    write_csv(os.path.join(args.output_dir, "generalization_test.csv"), gen_test_data)
    print(f"CSV files written to '{args.output_dir}/'")

    # ---- Dataset stats -------------------------------------------------------
    stats_lines = ["=" * 60, "DATASET STATISTICS", "=" * 60]

    for name, rows in [("train", train), ("dev", dev), ("test", test),
                        ("generalization_test", gen_test_data)]:
        stats_lines.append(stats_block(name, rows))

    stats_lines.append("\n[Overlap validation]")
    stats_lines.append(f"  Zero overlap across all splits: {no_overlap}")
    split_names = list(splits_to_check.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            common = {r["text"] for r in splits_to_check[a]} & {r["text"] for r in splits_to_check[b]}
            stats_lines.append(f"  {a} ∩ {b}: {len(common)} shared sentences")

    stats_path = os.path.join(args.output_dir, "dataset_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")

    print("\n" + "\n".join(stats_lines))
    print(f"\nStats written to '{stats_path}'")


if __name__ == "__main__":
    main()
