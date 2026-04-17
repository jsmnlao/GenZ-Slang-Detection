# Slang Detection

NLP project for detecting Gen Z / Gen Alpha slang in text using sequence labeling (BIO tagging).

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
bertweet_binary.py                 # Fine-tunes BERTweet for binary slang classification
bertweet_binary.ipynb              # Colab notebook version of bertweet_binary.py
bert_bio.py                        # Fine-tunes BERT for BIO token-level slang tagging
```
