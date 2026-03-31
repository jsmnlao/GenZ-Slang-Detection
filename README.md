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
  mlbtrio_labeled.csv              # MLB Trio data with sentence-level labels
  wsj_labeled.csv                  # WSJ data with sentence-level labels
output/
  train.csv                        # Training split
  dev.csv                          # Development/validation split
  test.csv                         # Test split
  generalization_test.csv          # Generalization test split (slang-only)
  dataset_stats.txt                # Full split statistics
dictionary_baseline/
  dictionary_baseline_report.txt   # Evaluation Report on dictionary lookup model
tfidf_baseline/
  tfidf_baseline_report.txt        # Evaluation report on TF-IDF sentence classifier
build_dataset.py                   # Script to generate splits from labeled data
dictionary_baseline.py             # Script to implement dictionary lookup model and generate outputs
tfidf_baseline.py                  # Script to train/evaluate TF-IDF sentence baseline
```
