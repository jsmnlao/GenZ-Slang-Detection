# -*- coding: utf-8 -*-
"""
bertweet_binary.py

Fine-tunes vinai/bertweet-base with a [CLS]-based binary classification head
for sentence-level slang detection (SLANG vs NOT_SLANG).

Includes:
  - Hyperparameter sweep over lr, batch size, max_seq_length, num_epochs
  - Dataset size ablation (25/50/75/100% of training data)
  - Final evaluation on test and generalization_test splits
  - Results saved to bertweet_binary_results/

Usage:
    # Full pipeline (sweep + ablation + final eval)
    python bertweet_binary.py

    # Sweep only (save best_config.json, stop before ablation/final eval)
    python bertweet_binary.py --sweep_only

    # Resume after sweep using saved best config
    python bertweet_binary.py --skip_sweep \\
        --best_config_json '{"learning_rate": 3e-05, "per_device_train_batch_size": 32, "max_seq_length": 128, "num_train_epochs": 3}'

    # Skip sweep and ablation, run only final eval
    python bertweet_binary.py --skip_sweep --skip_ablation \\
        --best_config_json '{"learning_rate": 3e-05, "per_device_train_batch_size": 32, "max_seq_length": 128, "num_train_epochs": 3}'
"""

import argparse
import csv
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOT_SLANG = "NOT_SLANG"
SLANG = "SLANG"
RANDOM_SEED = 42
MODEL_NAME = "vinai/bertweet-base"
OUTPUT_DIR = "bertweet_binary_results"

LABEL_MAP = {SLANG: 1, NOT_SLANG: 0}
ID2LABEL = {0: NOT_SLANG, 1: SLANG}
LABEL2ID = {NOT_SLANG: 0, SLANG: 1}

SWEEP_GRID = {
    "learning_rate": [2e-5, 3e-5, 5e-5],
    "per_device_train_batch_size": [16, 32],
    "max_seq_length": [64, 128],
    "num_train_epochs": [3, 5],
}

ABLATION_FRACTIONS = [0.25, 0.50, 0.75, 1.00]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SlangDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].fillna("").astype(str).tolist()
        self.labels = [LABEL_MAP[l] for l in df["label"].tolist()]
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
        # BERTweet has no segment embeddings — do not include token_type_ids
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def load_data(path):
    return pd.read_csv(path)


def find_latest_checkpoint(checkpoint_root):
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_root}")

    checkpoint_dirs = []
    for entry in os.listdir(checkpoint_root):
        path = os.path.join(checkpoint_root, entry)
        match = re.fullmatch(r"checkpoint-(\d+)", entry)
        if os.path.isdir(path) and match:
            checkpoint_dirs.append((int(match.group(1)), path))

    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No checkpoint-* directories found in {checkpoint_root}"
        )

    checkpoint_dirs.sort(key=lambda item: item[0])
    return checkpoint_dirs[-1][1]


def fresh_model():
    """Load a fresh copy of BERTweet with a binary classification head."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def make_output_csv(df, preds, out_path, text_col="text", gold_col="label"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gold_tags = df[gold_col].astype(str)
    pred_tags = pd.Series(preds).map(ID2LABEL)

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "gold_tags": gold_tags,
            "pred_tags": pred_tags,
            "exact_match": gold_tags == pred_tags,
        }
    )
    out.to_csv(out_path, index=False)
    return out_path


def get_training_args(config, run_dir, do_save=False):
    return TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=config["learning_rate"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch" if do_save else "no",
        load_best_model_at_end=do_save,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=RANDOM_SEED,
        # fp16 is unsafe on Apple Silicon MPS; use CUDA only
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )


def _cleanup(model, trainer):
    """Explicitly free GPU/MPS memory after each training run."""
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(train_df, dev_df, tokenizer):
    configs = list(
        itertools.product(
            SWEEP_GRID["learning_rate"],
            SWEEP_GRID["per_device_train_batch_size"],
            SWEEP_GRID["max_seq_length"],
            SWEEP_GRID["num_train_epochs"],
        )
    )
    records = []
    best_f1 = -1.0
    best_config = None

    for i, (lr, bs, seq_len, epochs) in enumerate(configs):
        config = {
            "learning_rate": lr,
            "per_device_train_batch_size": bs,
            "max_seq_length": seq_len,
            "num_train_epochs": epochs,
        }
        run_dir = os.path.join(OUTPUT_DIR, "checkpoints", f"run_{i:02d}")
        print(f"\n[Sweep {i + 1}/{len(configs)}] {config}")

        train_ds = SlangDataset(train_df, tokenizer, seq_len)
        dev_ds = SlangDataset(dev_df, tokenizer, seq_len)

        model = fresh_model()
        t_args = get_training_args(config, run_dir, do_save=False)
        trainer = Trainer(
            model=model,
            args=t_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()

        record = {**config, **{k.replace("eval_", ""): v for k, v in metrics.items()}}
        records.append(record)
        print(
            f"  Dev F1={metrics.get('eval_f1', 0):.4f}  "
            f"Acc={metrics.get('eval_accuracy', 0):.4f}"
        )

        if metrics.get("eval_f1", 0) > best_f1:
            best_f1 = metrics["eval_f1"]
            best_config = config.copy()

        _cleanup(model, trainer)

    print(f"\nBest dev F1: {best_f1:.4f}  Config: {best_config}")
    return best_config, records


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def run_ablation(train_df, dev_df, test_df, tokenizer, best_config):
    records = []
    seq_len = best_config["max_seq_length"]

    for frac in ABLATION_FRACTIONS:
        if frac < 1.0:
            subset_df = (
                train_df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=RANDOM_SEED))
                .reset_index(drop=True)
            )
        else:
            subset_df = train_df

        n_train = len(subset_df)
        run_dir = os.path.join(
            OUTPUT_DIR, "checkpoints", f"ablation_frac{int(frac * 100):03d}"
        )
        print(f"\n[Ablation] frac={frac:.2f}  n_train={n_train}")

        train_ds = SlangDataset(subset_df, tokenizer, seq_len)
        dev_ds = SlangDataset(dev_df, tokenizer, seq_len)
        test_ds = SlangDataset(test_df, tokenizer, seq_len)

        model = fresh_model()
        t_args = get_training_args(best_config, run_dir, do_save=False)
        trainer = Trainer(
            model=model,
            args=t_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        dev_metrics = trainer.evaluate(dev_ds)
        test_metrics = trainer.evaluate(test_ds)

        record = {
            "fraction": frac,
            "n_train": n_train,
            **{f"dev_{k.replace('eval_', '')}": v for k, v in dev_metrics.items()},
            **{f"test_{k.replace('eval_', '')}": v for k, v in test_metrics.items()},
        }
        records.append(record)
        print(
            f"  Dev F1={dev_metrics.get('eval_f1', 0):.4f}  "
            f"Test F1={test_metrics.get('eval_f1', 0):.4f}"
        )

        _cleanup(model, trainer)

    return records


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------


def run_final_eval(train_df, dev_df, test_df, gen_df, tokenizer, best_config):
    seq_len = best_config["max_seq_length"]
    run_dir = os.path.join(OUTPUT_DIR, "checkpoints", "final")
    print(f"\n[Final Eval] Training with best config: {best_config}")

    train_ds = SlangDataset(train_df, tokenizer, seq_len)
    dev_ds = SlangDataset(dev_df, tokenizer, seq_len)
    test_ds = SlangDataset(test_df, tokenizer, seq_len)
    gen_ds = SlangDataset(gen_df, tokenizer, seq_len)

    model = fresh_model()
    t_args = get_training_args(best_config, run_dir, do_save=True)
    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_metrics = trainer.evaluate(test_ds)
    gen_metrics = trainer.evaluate(gen_ds)
    test_prediction_results = trainer.predict(test_ds)
    gen_prediction_results = trainer.predict(gen_ds)

    test_predictions = np.argmax(test_prediction_results.predictions, axis=-1)
    gen_predictions = np.argmax(gen_prediction_results.predictions, axis=-1)

    test_predictions_path = make_output_csv(
        test_df,
        test_predictions,
        os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        text_col="text",
        gold_col="label",
    )
    gen_predictions_path = make_output_csv(
        gen_df,
        gen_predictions,
        os.path.join(OUTPUT_DIR, "generalization_test_predictions.csv"),
        text_col="text",
        gold_col="label",
    )

    print(f"  Test  F1={test_metrics.get('eval_f1', 0):.4f}")
    print(f"  Gen   F1={gen_metrics.get('eval_f1', 0):.4f}")
    print(f"  Test predictions saved to {test_predictions_path}")
    print(f"  Gen predictions saved to {gen_predictions_path}")

    return test_metrics, gen_metrics


def export_predictions_from_checkpoint(
    test_df, gen_df, tokenizer, checkpoint_path, best_config
):
    seq_len = best_config["max_seq_length"]
    print(f"\n[Predict Only] Loading checkpoint from: {checkpoint_path}")

    test_ds = SlangDataset(test_df, tokenizer, seq_len)
    gen_ds = SlangDataset(gen_df, tokenizer, seq_len)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=32,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    test_prediction_results = trainer.predict(test_ds)
    gen_prediction_results = trainer.predict(gen_ds)

    test_predictions = np.argmax(test_prediction_results.predictions, axis=-1)
    gen_predictions = np.argmax(gen_prediction_results.predictions, axis=-1)

    test_predictions_path = make_output_csv(
        test_df,
        test_predictions,
        os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        text_col="text",
        gold_col="label",
    )
    gen_predictions_path = make_output_csv(
        gen_df,
        gen_predictions,
        os.path.join(OUTPUT_DIR, "generalization_test_predictions.csv"),
        text_col="text",
        gold_col="label",
    )

    test_metrics = compute_metrics(
        (test_prediction_results.predictions, test_prediction_results.label_ids)
    )
    gen_metrics = compute_metrics(
        (gen_prediction_results.predictions, gen_prediction_results.label_ids)
    )

    print(f"  Test  F1={test_metrics.get('f1', 0):.4f}")
    print(f"  Gen   F1={gen_metrics.get('f1', 0):.4f}")
    print(f"  Test predictions saved to {test_predictions_path}")
    print(f"  Gen predictions saved to {gen_predictions_path}")

    return (
        {f"eval_{k}": v for k, v in test_metrics.items()},
        {f"eval_{k}": v for k, v in gen_metrics.items()},
    )


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------


def save_sweep_results(records, path):
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Sweep results saved to {path}")


def save_ablation_results(records, path):
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Ablation results saved to {path}")


def save_final_report(test_metrics, gen_metrics, best_config, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def m(metrics, key):
        return metrics.get(f"eval_{key}", metrics.get(key, float("nan")))

    test_f1 = m(test_metrics, "f1")
    gen_f1 = m(gen_metrics, "f1")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BERTWEET BINARY FINE-TUNING REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("=== BEST HYPERPARAMETERS ===\n")
        f.write(f"  Learning Rate             : {best_config['learning_rate']}\n")
        f.write(
            f"  Batch Size                : {best_config['per_device_train_batch_size']}\n"
        )
        f.write(f"  Max Sequence Length       : {best_config['max_seq_length']}\n")
        f.write(f"  Num Epochs                : {best_config['num_train_epochs']}\n\n")

        f.write("=== TEST SET RESULTS ===\n")
        f.write(f"  Accuracy  : {m(test_metrics, 'accuracy'):.4f}\n")
        f.write(f"  Precision : {m(test_metrics, 'precision'):.4f}\n")
        f.write(f"  Recall    : {m(test_metrics, 'recall'):.4f}\n")
        f.write(f"  F1        : {m(test_metrics, 'f1'):.4f}\n\n")

        f.write("=== GENERALIZATION TEST RESULTS ===\n")
        f.write(f"  Accuracy  : {m(gen_metrics, 'accuracy'):.4f}\n")
        f.write(f"  Precision : {m(gen_metrics, 'precision'):.4f}\n")
        f.write(f"  Recall    : {m(gen_metrics, 'recall'):.4f}\n")
        f.write(f"  F1        : {m(gen_metrics, 'f1'):.4f}\n\n")

        f.write("=== COMPARISON TO BASELINES ===\n")
        f.write(f"  {'Model':<30} {'Test F1':>10}  {'Gen Test F1':>12}\n")
        f.write("  " + "-" * 56 + "\n")
        f.write(f"  {'Dictionary Baseline':<30} {'0.6770':>10}  {'0.7437':>12}\n")
        f.write(f"  {'TF-IDF + LogReg':<30} {'0.8472':>10}  {'0.8796':>12}\n")
        f.write(f"  {'BERTweet (this run)':<30} {test_f1:>10.4f}  {gen_f1:>12.4f}\n")

    print(f"Final report saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTweet for sentence-level slang classification"
    )
    parser.add_argument("--train", default="output/train.csv")
    parser.add_argument("--dev", default="output/dev.csv")
    parser.add_argument("--test", default="output/test.csv")
    parser.add_argument("--generalization", default="output/generalization_test.csv")
    parser.add_argument("--output_dir", default="bertweet_binary_results")
    parser.add_argument(
        "--skip_sweep",
        action="store_true",
        help="Skip hyperparameter sweep; requires --best_config_json",
    )
    parser.add_argument(
        "--best_config_json",
        default=None,
        help="JSON string of best config, e.g. '{\"learning_rate\": 3e-05, ...}'",
    )
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument(
        "--sweep_only",
        action="store_true",
        help="Run sweep only; skip ablation and final eval. "
        "Best config is printed and saved to best_config.json for later use.",
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Skip training and export predictions using a saved final checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to a saved checkpoint directory. Defaults to the latest checkpoint in output_dir/checkpoints/final.",
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    # use_fast=False: BERTweet BPE slow tokenizer is required for normalization=True
    # normalization=True: applies TweetNormalizer matching BERTweet pretraining conditions
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=False, normalization=True
    )

    print("Loading data...")
    train_df = load_data(args.train)
    dev_df = load_data(args.dev)
    test_df = load_data(args.test)
    gen_df = load_data(args.generalization)
    print(
        f"  train={len(train_df)}  dev={len(dev_df)}  "
        f"test={len(test_df)}  gen={len(gen_df)}"
    )

    if args.predict_only:
        if args.best_config_json is not None:
            best_config = json.loads(args.best_config_json)
        else:
            best_config_path = os.path.join(OUTPUT_DIR, "best_config.json")
            if not os.path.exists(best_config_path):
                parser.error(
                    "--predict_only requires --best_config_json or an existing best_config.json in output_dir"
                )
            with open(best_config_path, encoding="utf-8") as f:
                best_config = json.load(f)

        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(
                os.path.join(OUTPUT_DIR, "checkpoints", "final")
            )

        test_metrics, gen_metrics = export_predictions_from_checkpoint(
            test_df, gen_df, tokenizer, checkpoint_path, best_config
        )
        save_final_report(
            test_metrics,
            gen_metrics,
            best_config,
            os.path.join(OUTPUT_DIR, "final_report.txt"),
        )
        return

    # ------------------------------------------------------------------
    # Hyperparameter sweep
    # ------------------------------------------------------------------
    if not args.skip_sweep:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SWEEP")
        print("=" * 60)
        best_config, sweep_records = run_sweep(train_df, dev_df, tokenizer)
        save_sweep_results(sweep_records, os.path.join(OUTPUT_DIR, "sweep_results.csv"))
        best_config_path = os.path.join(OUTPUT_DIR, "best_config.json")
        with open(best_config_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2)
        print(f"Best config saved to {best_config_path}")
    else:
        if args.best_config_json is None:
            parser.error("--skip_sweep requires --best_config_json")
        best_config = json.loads(args.best_config_json)
        print(f"\nSkipping sweep. Using provided config: {best_config}")

    if args.sweep_only:
        print("\n--sweep_only set. Stopping after sweep.")
        print(
            f"To continue later, run:\n  python bertweet_binary.py --skip_sweep "
            f"--best_config_json '{json.dumps(best_config)}'"
        )
        return

    # ------------------------------------------------------------------
    # Dataset size ablation
    # ------------------------------------------------------------------
    if not args.skip_ablation:
        print("\n" + "=" * 60)
        print("DATASET SIZE ABLATION")
        print("=" * 60)
        ablation_records = run_ablation(
            train_df, dev_df, test_df, tokenizer, best_config
        )
        save_ablation_results(
            ablation_records, os.path.join(OUTPUT_DIR, "ablation_results.csv")
        )
    else:
        print("\nSkipping dataset size ablation.")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    test_metrics, gen_metrics = run_final_eval(
        train_df, dev_df, test_df, gen_df, tokenizer, best_config
    )
    save_final_report(
        test_metrics,
        gen_metrics,
        best_config,
        os.path.join(OUTPUT_DIR, "final_report.txt"),
    )


if __name__ == "__main__":
    main()
