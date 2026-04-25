# Automated Detection of Gen Z Slang in Social Media Texts

## Course Information
**Course:** CSCI 544 - Applied Natural Language Processing

**Instructor:** Professor Robin Jia, Professor Xuezhe Ma

**Institution:** University of Southern California

**Term:** Spring 2026

## Team Members
* Christopher Ernesto (cernesto@usc.edu)
* Christopher Sumali (sumali@usc.edu)
* Jasmine Lao (laojasmi@usc.edu)
* Ryan Keng (rkeng@usc.edu)
* Romeo Nickel (rjnickel@usc.edu)

## Project Description

This project focuses on identifying and labeling slang expressions in text, particularly those used by Gen Z and Gen Alpha. We formulate Gen Z slang detection under two task settings: (1) sentence-level binary classification, which predicts whether a sentence contains slang, and (2) token-level sequence labeling, which identifies the full span of slang expressions using BIO tags.

We implement and compare four approaches: a rule-based method, a classical machine learning model, a supervised neural model, and large language model (LLM) prompting to evaluate how well different approaches capture contextual and evolving language patterns.

## Environment Setup

Run all commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need to rebuild the train/dev/test/generalization splits from the source data, run:

```bash
python build_dataset.py
```

## Device / System Used

This repository was run and checked locally on:

- macOS 15.7.4
- Apple Silicon (`arm64`)
- Python 3.13.7

The neural models use PyTorch and Hugging Face Transformers. For sentence-level BERTweet experiments, we recommend running [bertweet_binary.ipynb](/Users/ryankeng/Documents/Repos/csci544-final-project/bertweet_binary.ipynb) in Google Colab rather than running `bertweet_binary.py` on a local machine, because Colab provides a faster and more practical training environment for this model.

## Running The Code

First, generate the dataset splits if `output/` does not already contain them:

```bash
python build_dataset.py
```

To generate model results:

```bash
python dictionary_baseline.py
python tfidf_baseline.py
python bert_bio.py
```

For the sentence-level BERTweet model, we recommend running [bertweet_binary.ipynb](/Users/ryankeng/Documents/Repos/csci544-final-project/bertweet_binary.ipynb) in Google Colab instead of running `bertweet_binary.py` locally, since Colab is noticeably better for training speed and overall performance. The local script is still included for reproducibility.

We do not include standalone prompting scripts for the LLM experiments; those results are documented directly in `llm_evaluation/`.

To generate error analysis reports:

```bash
python error_analysis_baseline.py
python error_analysis_BERT_sentence.py
python error_analysis_BERT_bio.py
python error_analysis_dict_bio.py
python error_analysis_llm.py
```

## How Results Are Generated

This project has two output categories: model results and error analysis reports.

For model results:

- `dictionary_baseline.py` runs the rule-based dictionary baseline and writes results to `dictionary_baseline/`.
- `tfidf_baseline.py` runs the TF-IDF baseline and writes results to `tfidf_baseline/`.
- [bertweet_binary.ipynb](/Users/ryankeng/Documents/Repos/csci544-final-project/bertweet_binary.ipynb) is the recommended way to fine-tune BERTweet for sentence-level binary classification, especially in Google Colab for better performance; `bertweet_binary.py` is the local-script version of the same workflow. Both write outputs to `bertweet_binary_results/`.
- `bert_bio.py` fine-tunes BERTweet for token-level BIO tagging and writes outputs to `bert_bio/`.
- LLM prompting results are documented in `llm_evaluation/`, including Qwen and DeepSeek qualitative reports and prompting-based evaluations.

Each model script reads the prepared CSV splits in `output/`, generates predictions for the relevant split(s), computes evaluation metrics, and saves report files plus prediction CSVs in its corresponding results directory.

For error analysis reports:

- `error_analysis_baseline.py` performs sentence-level binary error analysis for the dictionary and TF-IDF baselines.
- `error_analysis_BERT_sentence.py` performs sentence-level binary error analysis for fine-tuned BERTweet.
- `error_analysis_BERT_bio.py` performs token-level sequence-alignment and span-based error analysis for fine-tuned BERTweet.
- `error_analysis_dict_bio.py` performs token-level sequence-alignment error analysis for the dictionary baseline.
- `error_analysis_llm.py` performs sentence-level binary error analysis for Qwen and DeepSeek outputs.

These scripts consume prediction files produced by the model runs, then write human-readable reports to `error_analysis/`.

## Dataset

The dataset combines three sources:

| Source | Description |
|---|---|
| `gen_alpha` | Gen Alpha tweets with slang annotations |
| `gen_z` | MLB Trio Gen Z social media text |
| `wsj` | Wall Street Journal articles (non-slang baseline) |

Each example is labeled `SLANG` or `NOT_SLANG` at the sentence level, with token-level BIO tags.

## Dataset Splits

All splits are balanced 50/50 between `SLANG` and `NOT_SLANG` labels (except `generalization_test`). There is **zero overlap** across all splits.

| Split | File | Total | SLANG | NOT_SLANG |
|---|---|---|---|---|
| Train | `output/train.csv` | 14,000 | 7,000 (50%) | 7,000 (50%) |
| Dev | `output/dev.csv` | 3,000 | 1,500 (50%) | 1,500 (50%) |
| Test | `output/test.csv` | 3,000 | 1,500 (50%) | 1,500 (50%) |
| Generalization Test | `output/generalization_test.csv` | 1,000 | 1,000 (100%) | — |

### Source Distribution (train / dev / test)

| Source | Train | Dev | Test |
|---|---|---|---|
| gen_alpha | 10,993 (78.5%) | 2,355 (78.5%) | 2,356 (78.5%) |
| gen_z | 1,050 (7.5%) | 225 (7.5%) | 225 (7.5%) |
| wsj | 1,957 (14.0%) | 420 (14.0%) | 419 (14.0%) |

The `generalization_test` split contains 1,000 `gen_alpha` slang examples held out to evaluate generalization to unseen slang patterns.

## Files

```
data/
  gen_alpha_tweets_labeled.csv     # Gen Alpha tweets with sentence-level labels
  gen_alpha_tweets_bio_tagged.csv  # Gen Alpha tweets with BIO tags
  mlbtrio_cleaned.csv              # MLB Trio slang lexicon (used by dictionary baseline)
  mlbtrio_labeled.csv              # MLB Trio data with sentence-level labels
  mlbtrio_bio_tagged.csv           # MLB Trio data with BIO tags
  mlbtrio_bio_tagged.conll         # MLB Trio data in CoNLL format
  mlbtrio_bio_tagged.jsonl         # MLB Trio data in JSONL format
  wsj_labeled.csv                  # WSJ data with sentence-level labels
output/
  train.csv                        # Training split
  dev.csv                          # Development/validation split
  test.csv                         # Test split
  generalization_test.csv          # Generalization test split (slang-only)
  dataset_stats.txt                # Full split statistics
dictionary_baseline/
  dictionary_baseline_report.txt   # Evaluation report on dictionary lookup model
  sentence/
    test_predictions.csv                 # Sentence-level predictions on test split
    generalization_test_predictions.csv  # Sentence-level predictions on generalization split
  bio/
    test_predictions.csv                 # BIO tag predictions on test split
    generalization_test_predictions.csv  # BIO tag predictions on generalization split
tfidf_baseline/
  tfidf_baseline_report.txt              # Evaluation report on TF-IDF sentence classifier
  test_predictions.csv                   # Sentence-level predictions on test split
  generalization_test_predictions.csv    # Sentence-level predictions on generalization split
bertweet_binary_results/
  sweep_results.csv                # Hyperparameter sweep results (24 configs)
  ablation_results.csv             # Dataset size ablation results (25/50/75/100%)
  best_config.json                 # Best hyperparameter configuration from sweep
  final_report.txt                 # Final evaluation report vs. baselines
  test_predictions.csv             # Sentence-level predictions on test split
  generalization_test_predictions.csv  # Sentence-level predictions on generalization split
bert_bio/
  bert_bio_report.txt              # Evaluation report on BERT BIO token classification model
  test_predictions.csv             # BIO tag predictions on test split
  generalization_test_predictions.csv  # BIO tag predictions on generalization split
error_analysis/
  dict_sentence_test_report.txt              # Sentence-level error analysis: dictionary baseline on test
  dict_sentence_generalization_report.txt    # Sentence-level error analysis: dictionary baseline on generalization
  tfidf_sentence_test_report.txt             # Sentence-level error analysis: TF-IDF baseline on test
  tfidf_sentence_generalization_report.txt   # Sentence-level error analysis: TF-IDF baseline on generalization
  bert_sentence_test_report.txt              # Sentence-level error analysis: BERTweet on test
  bert_sentence_generalization_report.txt    # Sentence-level error analysis: BERTweet on generalization
  bert_bio_test_report.txt                   # BIO-level error analysis: BERT BIO on test
  bert_bio_generalization_report.txt         # BIO-level error analysis: BERT BIO on generalization
  dict_bio_test_report.txt                   # BIO-level error analysis: dictionary baseline on test
  dict_bio_generalization_report.txt         # BIO-level error analysis: dictionary baseline on generalization
build_dataset.py                   # Generates train/dev/test/generalization splits from labeled data
dictionary_baseline.py             # Dictionary lookup model: evaluation and prediction CSV export
tfidf_baseline.py                  # TF-IDF sentence classifier: training, evaluation, and prediction CSV export
error_analysis_baseline.py         # Sentence-level error analysis for dictionary and TF-IDF baselines
error_analysis_BERT_sentence.py    # Sentence-level error analysis for BERTweet binary classifier
error_analysis_BERT_bio.py         # BIO-level error analysis for BERT BIO model
error_analysis_dict_bio.py         # BIO-level error analysis for dictionary baseline
error_analysis_llm.py              # Sentence-level error analysis for Qwen and DeepSeek outputs
bertweet_binary.py                 # Fine-tunes BERTweet for binary slang classification
bertweet_binary.ipynb              # Colab notebook version of bertweet_binary.py
bert_bio.py                        # Fine-tunes BERT for BIO token-level slang tagging
llm_evaluation/
  outputs/
    qwen/                          # Qwen qualitative reports and prompting outputs on test
    qwen_generalization/           # Qwen prompting outputs on generalization split
    deepseek/                      # DeepSeek qualitative reports and prompting outputs on test
    deepseek_generalization/       # DeepSeek prompting outputs on generalization split
```
