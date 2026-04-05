"""
bert_bio.py

Fine-tunes a BERT-style token classification model for BIO tagging of Gen Z / Gen Alpha slang terms.

Usage:
    python bert_bio.py
    python bert_bio.py --train_path output/train.csv --dev_path output/dev.csv --test_path output/test.csv --generalization_path output/generalization_test.csv --output_dir bert_bio
    python bert_bio.py --output_dir bert_bio --predict_only

Output:
    bert_bio/model.safetensors
    bert_bio/config.json
    bert_bio/tokenizer_config.json
    bert_bio/special_tokens_map.json
    bert_bio/vocab.txt
    bert_bio/bpe.codes
    bert_bio/label_map.json
    bert_bio/metrics.json
    bert_bio/bert_bio_report.txt
    bert_bio/test_predictions.csv
    bert_bio/generalization_test_predictions.csv
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

MAX_LENGTH = 128
MODEL_NAME = "vinai/bertweet-base"
RANDOM_SEED = 42
IGNORE_INDEX = -100
TEXT_COLUMN_CANDIDATES = ("text", "sentence")
TAG_COLUMN = "tags"
TAG_NORMALIZATION = {
    "B-I-SLANG": "B-SLANG",
}


def resolve_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        f"Expected one of {TEXT_COLUMN_CANDIDATES} in dataframe columns, got {list(df.columns)}"
    )


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=[TAG_COLUMN]).copy()
    text_column = resolve_text_column(df)
    df[text_column] = df[text_column].fillna("").astype(str)
    df[TAG_COLUMN] = df[TAG_COLUMN].fillna("").astype(str)

    valid_rows = []
    for _, row in df.iterrows():
        words = row[text_column].split()
        tags = normalize_tag_sequence(row[TAG_COLUMN]).split()
        if words and len(words) == len(tags):
            valid_rows.append(
                {
                    "text": row[text_column],
                    "tags": " ".join(tags),
                }
            )

    if not valid_rows:
        raise ValueError(f"No valid examples found in {path}")

    return pd.DataFrame(valid_rows)


def normalize_tag_sequence(tag_sequence: str) -> str:
    tags = str(tag_sequence).split()
    normalized_tags = [TAG_NORMALIZATION.get(tag, tag) for tag in tags]
    return " ".join(normalized_tags)


def build_tag_list(df_train: pd.DataFrame) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    tags = set()
    for tag_sequence in df_train[TAG_COLUMN]:
        tags.update(tag_sequence.split())

    tag_list = sorted(tags)
    tag2id = {tag: idx for idx, tag in enumerate(tag_list)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag_list, tag2id, id2tag


def tokenize_and_align_labels(batch, tokenizer, tag2id, max_length=MAX_LENGTH):
    if not getattr(tokenizer, "is_fast", False):
        return tokenize_and_align_labels_slow(
            batch,
            tokenizer=tokenizer,
            tag2id=tag2id,
            max_length=max_length,
        )

    words_batch = [text.split() for text in batch["text"]]
    tags_batch = [tag_sequence.split() for tag_sequence in batch["tags"]]

    tokenized = tokenizer(
        words_batch,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned_labels = []
    for batch_index, word_tags in enumerate(tags_batch):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(IGNORE_INDEX)
            elif word_id != previous_word_id:
                label_ids.append(tag2id[word_tags[word_id]])
            else:
                label_ids.append(IGNORE_INDEX)
            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def tokenize_and_align_labels_slow(batch, tokenizer, tag2id, max_length=MAX_LENGTH):
    words_batch = [text.split() for text in batch["text"]]
    tags_batch = [tag_sequence.split() for tag_sequence in batch["tags"]]

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    unk_token = tokenizer.unk_token or "[UNK]"

    for words, tags in zip(words_batch, tags_batch):
        input_ids = []
        label_ids = []

        for word, tag in zip(words, tags):
            pieces = tokenizer.tokenize(word)
            if not pieces:
                pieces = [unk_token]

            piece_ids = tokenizer.convert_tokens_to_ids(pieces)
            input_ids.extend(piece_ids)
            label_ids.append(tag2id[tag])
            label_ids.extend([IGNORE_INDEX] * (len(piece_ids) - 1))

        if cls_token_id is not None:
            input_ids = [cls_token_id] + input_ids
            label_ids = [IGNORE_INDEX] + label_ids
        if sep_token_id is not None:
            input_ids = input_ids + [sep_token_id]
            label_ids = label_ids + [IGNORE_INDEX]

        input_ids = input_ids[:max_length]
        label_ids = label_ids[:max_length]
        attention_mask = [1] * len(input_ids)

        pad_length = max_length - len(input_ids)
        if pad_length > 0:
            input_ids.extend([pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
            label_ids.extend([IGNORE_INDEX] * pad_length)

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(label_ids)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
    }


def build_dataset(
    train_path: str,
    dev_path: str,
    model_name: str,
    max_length: int,
    test_path: Optional[str] = None,
    generalization_path: Optional[str] = None,
):
    df_train = load_split(train_path)
    df_dev = load_split(dev_path)

    dataset_splits = {
        "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
        "dev": Dataset.from_pandas(df_dev.reset_index(drop=True)),
    }

    if test_path and os.path.exists(test_path):
        dataset_splits["test"] = Dataset.from_pandas(load_split(test_path).reset_index(drop=True))
    if generalization_path and os.path.exists(generalization_path):
        dataset_splits["generalization_test"] = Dataset.from_pandas(
            load_split(generalization_path).reset_index(drop=True)
        )

    dataset = DatasetDict(dataset_splits)
    tag_list, tag2id, id2tag = build_tag_list(df_train)
    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)

    tokenized_dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(
            batch,
            tokenizer=tokenizer,
            tag2id=tag2id,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset, tokenizer, tag_list, tag2id, id2tag


def flatten_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    id2tag: Dict[int, str],
) -> Tuple[List[str], List[str]]:
    predictions = np.argmax(logits, axis=2)
    gold_tags = []
    pred_tags = []

    for prediction_row, label_row in zip(predictions, labels):
        for prediction_id, label_id in zip(prediction_row, label_row):
            if label_id == IGNORE_INDEX:
                continue
            gold_tags.append(id2tag[int(label_id)])
            pred_tags.append(id2tag[int(prediction_id)])

    return gold_tags, pred_tags


def compute_micro_metrics(gold_tags: List[str], pred_tags: List[str], positive_labels=None):
    if positive_labels is None:
        correct = sum(1 for gold, pred in zip(gold_tags, pred_tags) if gold == pred)
        total = len(gold_tags)
        accuracy = correct / total if total else 0.0
        return accuracy, accuracy, accuracy, accuracy

    positive_labels = set(positive_labels)
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold, pred in zip(gold_tags, pred_tags):
        if pred in positive_labels and gold in positive_labels:
            if pred == gold:
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1
        elif pred in positive_labels and gold not in positive_labels:
            false_positive += 1
        elif pred not in positive_labels and gold in positive_labels:
            false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    support = sum(1 for gold in gold_tags if gold in positive_labels)
    return precision, recall, f1, float(support)


def compute_binary_metrics(gold_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    total = len(gold_labels)
    correct = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred)
    true_positive = sum(
        1 for gold, pred in zip(gold_labels, pred_labels) if gold == 1 and pred == 1
    )
    false_positive = sum(
        1 for gold, pred in zip(gold_labels, pred_labels) if gold == 0 and pred == 1
    )
    false_negative = sum(
        1 for gold, pred in zip(gold_labels, pred_labels) if gold == 1 and pred == 0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def is_slang_tag(tag: str) -> bool:
    return tag in {"B-SLANG", "I-SLANG"}


def compute_sentence_metrics_from_sequences(
    gold_sequences: List[List[str]],
    pred_sequences: List[List[str]],
) -> Dict[str, float]:
    gold_labels = [1 if any(is_slang_tag(tag) for tag in seq) else 0 for seq in gold_sequences]
    pred_labels = [1 if any(is_slang_tag(tag) for tag in seq) else 0 for seq in pred_sequences]
    return compute_binary_metrics(gold_labels, pred_labels)


def compute_token_metrics_from_sequences(
    gold_sequences: List[List[str]],
    pred_sequences: List[List[str]],
) -> Dict[str, float]:
    gold_tags = [tag for seq in gold_sequences for tag in seq]
    pred_tags = [tag for seq in pred_sequences for tag in seq]
    accuracy = (
        sum(1 for gold, pred in zip(gold_tags, pred_tags) if gold == pred) / len(gold_tags)
        if gold_tags
        else 0.0
    )
    precision, recall, f1, _ = compute_micro_metrics(
        gold_tags,
        pred_tags,
        positive_labels=["B-SLANG", "I-SLANG"],
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_split_metrics_from_dataframe(df_split: pd.DataFrame, tokenizer, model, id2label, max_length: int):
    gold_sequences = []
    pred_sequences = []

    for _, row in df_split.iterrows():
        text = str(row["text"])
        gold_tags = normalize_tag_sequence(row["tags"]).split()
        pred_tags = predict_word_tags(text, tokenizer, model, id2label, max_length)
        gold_sequences.append(gold_tags)
        pred_sequences.append(pred_tags[: len(gold_tags)])

    return {
        "sentence": compute_sentence_metrics_from_sequences(gold_sequences, pred_sequences),
        "token": compute_token_metrics_from_sequences(gold_sequences, pred_sequences),
    }


def compute_metrics_builder(id2tag: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        gold_tags, pred_tags = flatten_predictions(logits, labels, id2tag)

        accuracy = (
            sum(1 for gold, pred in zip(gold_tags, pred_tags) if gold == pred) / len(gold_tags)
            if gold_tags
            else 0.0
        )
        precision_all, recall_all, f1_all, _ = compute_micro_metrics(
            gold_tags,
            pred_tags,
            positive_labels=list(id2tag.values()),
        )
        precision_slang, recall_slang, f1_slang, slang_support = compute_micro_metrics(
            gold_tags,
            pred_tags,
            positive_labels=["B-SLANG", "I-SLANG"],
        )

        return {
            "token_accuracy": accuracy,
            "token_precision_all": precision_all,
            "token_recall_all": recall_all,
            "token_f1_all": f1_all,
            "token_precision_slang": precision_slang,
            "token_recall_slang": recall_slang,
            "token_f1_slang": f1_slang,
            "token_support_slang": slang_support,
        }

    return compute_metrics


def write_metrics_report(metrics_by_split: Dict[str, Dict[str, float]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BERT BIO TAGGING REPORT\n")
        f.write("=" * 60 + "\n\n")

        for split_name, metrics in metrics_by_split.items():
            f.write(f"=== {split_name.upper()} ===\n")
            if "sentence" in metrics:
                sentence_metrics = metrics["sentence"]
                f.write("  Sentence-level:\n")
                f.write(f"    Accuracy  : {sentence_metrics['accuracy']:.4f}\n")
                f.write(f"    Precision : {sentence_metrics['precision']:.4f}\n")
                f.write(f"    Recall    : {sentence_metrics['recall']:.4f}\n")
                f.write(f"    F1        : {sentence_metrics['f1']:.4f}\n")
            if "token" in metrics:
                token_metrics = metrics["token"]
                f.write("  Token-level:\n")
                f.write(f"    Accuracy  : {token_metrics['accuracy']:.4f}\n")
                f.write(f"    Precision : {token_metrics['precision']:.4f}\n")
                f.write(f"    Recall    : {token_metrics['recall']:.4f}\n")
                f.write(f"    F1        : {token_metrics['f1']:.4f}\n")
            f.write("\n")


def predict_word_tags(text: str, tokenizer, model, id2label: Dict[int, str], max_length: int):
    words = str(text).split()
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    unk_token = tokenizer.unk_token or "[UNK]"

    input_ids = []
    word_starts = []
    current_index = 0

    for word in words:
        pieces = tokenizer.tokenize(word)
        if not pieces:
            pieces = [unk_token]
        piece_ids = tokenizer.convert_tokens_to_ids(pieces)
        word_starts.append(current_index)
        input_ids.extend(piece_ids)
        current_index += len(piece_ids)

    if cls_token_id is not None:
        input_ids = [cls_token_id] + input_ids
        word_starts = [idx + 1 for idx in word_starts]
    if sep_token_id is not None:
        input_ids = input_ids + [sep_token_id]

    input_ids = input_ids[:max_length]
    attention_mask = [1] * len(input_ids)

    import torch
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([input_ids], device=device),
            attention_mask=torch.tensor([attention_mask], device=device),
        )

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    pred_tags = []
    for start_idx in word_starts:
        if start_idx >= len(pred_ids):
            break
        pred_tags.append(id2label[int(pred_ids[start_idx])])

    if len(pred_tags) < len(words):
        pred_tags.extend(["TRUNCATED"] * (len(words) - len(pred_tags)))

    return pred_tags


def export_predictions(split_name: str, df_split: pd.DataFrame, tokenizer, model, id2label, output_dir: str, max_length: int):
    rows = []
    for _, row in df_split.iterrows():
        text = str(row["text"])
        gold_tags = normalize_tag_sequence(row["tags"]).split()
        pred_tags = predict_word_tags(text, tokenizer, model, id2label, max_length)
        rows.append(
            {
                "text": text,
                "gold_tags": " ".join(gold_tags),
                "pred_tags": " ".join(pred_tags),
                "exact_match": gold_tags == pred_tags,
            }
        )

    output_path = os.path.join(output_dir, f"{split_name}_predictions.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a BERT-style model for BIO tagging of slang terms."
    )
    parser.add_argument("--train_path", default="output/train.csv")
    parser.add_argument("--dev_path", default="output/dev.csv")
    parser.add_argument("--test_path", default="output/test.csv")
    parser.add_argument("--generalization_path", default="output/generalization_test.csv")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--output_dir", default="bert_bio_model")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training when supported by your hardware.",
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Skip training and export predictions using an existing saved model in output_dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    report_path = os.path.join(args.output_dir, "bert_bio_report.txt")
    report_split_metrics = {}

    if args.predict_only:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, normalization=True)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "label_map.json"), encoding="utf-8") as f:
            label_map = json.load(f)
        label_list = label_map["labels"]
        label2id = label_map["label2id"]
        id2label = {int(idx): label for idx, label in label_map["id2label"].items()}
        trainer = None
    else:
        tokenized_datasets, tokenizer, label_list, label2id, id2label = build_dataset(
            train_path=args.train_path,
            dev_path=args.dev_path,
            test_path=args.test_path,
            generalization_path=args.generalization_path,
            model_name=args.model_name,
            max_length=args.max_length,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="token_f1_slang",
            greater_is_better=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            report_to="none",
            fp16=args.fp16,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["dev"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_builder(id2label),
        )

        train_result = trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        label_map_path = os.path.join(args.output_dir, "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "labels": label_list,
                    "label2id": label2id,
                    "id2label": {str(idx): label for idx, label in id2label.items()},
                },
                f,
                indent=2,
            )

        metrics_by_split = {"train_runtime": train_result.metrics}

        for split_name in ("dev", "test", "generalization_test"):
            if split_name in tokenized_datasets:
                metrics_by_split[split_name] = trainer.evaluate(
                    eval_dataset=tokenized_datasets[split_name],
                    metric_key_prefix=split_name,
                )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_by_split, f, indent=2)

    prediction_outputs = []
    split_dataframes = []
    for split_name, split_path in (
        ("train", args.train_path),
        ("dev", args.dev_path),
        ("test", args.test_path),
        ("generalization_test", args.generalization_path),
    ):
        if split_path and os.path.exists(split_path):
            df_split = load_split(split_path)
            split_dataframes.append((split_name, df_split))
            report_split_metrics[split_name] = compute_split_metrics_from_dataframe(
                df_split=df_split,
                tokenizer=tokenizer,
                model=model if trainer is None else trainer.model,
                id2label=id2label,
                max_length=args.max_length,
            )

    for split_name, df_split in split_dataframes:
        if split_name in {"test", "generalization_test"}:
            prediction_outputs.append(
                export_predictions(
                    split_name=split_name,
                    df_split=df_split,
                    tokenizer=tokenizer,
                    model=model if trainer is None else trainer.model,
                    id2label=id2label,
                    output_dir=args.output_dir,
                    max_length=args.max_length,
                )
            )

    write_metrics_report(report_split_metrics, report_path)

    if args.predict_only:
        print(f"Loaded existing model from: {args.output_dir}")
    else:
        print("Training complete.")
        print(f"Model and tokenizer saved to: {args.output_dir}")
        print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")
    for prediction_path in prediction_outputs:
        print(f"Predictions saved to: {prediction_path}")


if __name__ == "__main__":
    main()
