# Steam Game Review Summarization

This project compares three weighted summarization approaches for Steam game reviews.

The goal is to summarize large collections of Steam reviews into shorter, more useful summaries while allowing review metadata such as playtime and helpfulness to influence review importance.

## Games Used

The current experiment uses three games with different overall reception patterns:

- Clair Obscur: Expedition 33
- Overwatch
- Spacebase DF-9

Each game is expected to contain roughly 2,000 Steam reviews.

## Project Structure

```text
NLP/
├── .venv/
├── data/
│   ├── raw/
│   │   ├── clair_obscur.csv
│   │   ├── overwatch.csv
│   │   └── spacebase_df9.csv
│   │
│   └── cleaned/
│       ├── clair_obscur.csv
│       ├── overwatch.csv
│       └── spacebase_df9.csv
│
├── output/
│   └── summaries.json
│
├── processing.py
├── execution.py
├── requirements.txt
└── README.md
```

## Expected CSV Format

Each raw CSV should contain the following columns:

```text
review,recommended,helpful_or_not,playtime_at_review
```

Meaning:

- `review` — Steam review text
- `recommended` — whether the reviewer recommends the game
- `helpful_or_not` — whether the review is considered helpful
- `playtime_at_review` — reviewer playtime at the time the review was written

## Environment Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install pandas numpy scikit-learn torch transformers sentencepiece
```

Optional: save the installed versions:

```powershell
pip freeze > requirements.txt
```

## Pipeline

The project is executed in two main stages.

```text
Raw Steam CSV files
        ↓
   processing.py
        ↓
Cleaned CSV files
        ↓
   execution.py
        ↓
output/summaries.json
        ↓
Evaluation stage
```

## Stage 1 — Data Preprocessing

Run:

```powershell
python processing.py
```

The script reads:

```text
data/raw/clair_obscur.csv
data/raw/overwatch.csv
data/raw/spacebase_df9.csv
```

and writes cleaned versions to:

```text
data/cleaned/clair_obscur.csv
data/cleaned/overwatch.csv
data/cleaned/spacebase_df9.csv
```

The preprocessing stage is responsible for removing review noise such as:

- Steam BBCode
- ASCII / Unicode art
- decorative symbol sequences
- excessive repeated characters
- excessive whitespace
- low-information or junk reviews

Normal language structure is intentionally preserved.

Stopwords and punctuation are not removed because they are useful for sentence structure and abstractive summarization.

The cleaned files retain the same four columns:

```text
review,recommended,helpful_or_not,playtime_at_review
```

## Stage 2 — Summarization Execution

Run:

```powershell
python execution.py
```

The script processes all three games automatically.

For every game, a review importance score is calculated before summarization.

### Review Importance Score

The current weighting is:

```text
70% Content Representativeness
20% Reviewer Playtime
10% Helpfulness
```

Conceptually:

```text
importance_score =
0.70 × content_score
+ 0.20 × playtime_score
+ 0.10 × helpfulness_score
```

Content representativeness is calculated using TF-IDF similarity to the review corpus centroid.

Playtime uses logarithmic scaling so extremely large playtime values do not dominate the score.

`recommended` is not treated as an importance score. A positive recommendation is not considered more valuable than a negative recommendation.

Instead, recommendation status is used where appropriate to preserve the original Recommended / Not Recommended distribution.

## Summarization Methods

The program compares three weighted approaches.

### Method 1 — Weighted TF-IDF Extractive

All reviews are used.

The pipeline is:

```text
All reviews
    ↓
Sentence segmentation
    ↓
Sentence TF-IDF relevance
    +
Review importance score
    ↓
Rank sentences
    ↓
Select highest-ranked sentences
    ↓
Extractive summary
```

The final summary contains sentences taken directly from the original reviews.

### Method 2 — Weighted Full BART

All reviews are still used.

Because BART cannot accept thousands of reviews in one input, reviews are divided into token-safe chunks.

The pipeline is:

```text
All reviews
    ↓
Review importance ranking
    ↓
Chunk reviews
    ↓
BART summarizes each chunk
    ↓
Combine intermediate summaries
    ↓
Summarize again when necessary
    ↓
Final abstractive summary
```

This process is hierarchical.

"Full BART" means all reviews are eventually processed by BART, not that all reviews are inserted into one model input at once.

### Method 3 — Weighted TF-IDF + BART

This method reduces the number of reviews processed by BART.

The pipeline is:

```text
All reviews
    ↓
Review importance score
    ↓
Preserve Recommended / Not Recommended ratio
    ↓
Select representative reviews
    ↓
BART chunking
    ↓
Hierarchical BART summarization
    ↓
Final abstractive summary
```

The current implementation selects approximately 400 representative reviews per game.

The purpose of this method is to compare summary quality and runtime against Full BART.

## BART Model

The current abstractive model is:

```text
facebook/bart-large-cnn
```

The model is downloaded automatically by Hugging Face the first time the program runs.

The download is large, but the model is normally cached locally after the first successful download.

Future executions should reuse the cached model.

## Output

The execution script writes the results to:

```text
output/summaries.json
```

The output contains the summaries for all three games and all three methods.

It also stores useful experiment information such as:

- number of reviews analyzed
- recommendation counts
- weighting configuration
- number of reviews used by each method
- runtime per method
- generated summary

Example structure:

```json
{
  "Overwatch": {
    "dataset": {
      "reviews_analyzed": 2000,
      "recommended": 850,
      "not_recommended": 1150
    },
    "weighting": {
      "content_representativeness": 0.7,
      "playtime": 0.2,
      "helpfulness": 0.1
    },
    "weighted_tfidf": {
      "runtime_seconds": 1.2,
      "summary": "..."
    },
    "weighted_full_bart": {
      "runtime_seconds": 900.5,
      "summary": "..."
    },
    "weighted_tfidf_bart": {
      "reviews_used": 400,
      "runtime_seconds": 180.7,
      "summary": "..."
    }
  }
}
```

## Recommended Execution Order

Always run:

```powershell
python processing.py
```

before:

```powershell
python execution.py
```

The complete workflow is:

```text
1. Put raw datasets in data/raw/
2. Activate .venv
3. Run processing.py
4. Confirm cleaned files exist in data/cleaned/
5. Run execution.py
6. Check output/summaries.json
7. Use summaries.json for evaluation
```

## Evaluation

The saved JSON output is intended to be used by a separate evaluation stage.

The summarization script does not need to be rerun just to evaluate previously generated summaries.

Possible evaluation criteria include:

- ROUGE
- BERTScore
- human preference evaluation
- runtime comparison
- qualitative error analysis

Reference-based metrics such as ROUGE and BERTScore require an appropriate reference summary, so the exact evaluation design should be defined separately.

## Notes

Full BART is expected to be the slowest method because all reviews are processed through the transformer model.

Weighted TF-IDF + BART is designed to test whether a smaller representative review subset can produce comparable summaries with lower computational cost.

The weighting percentages are experiment parameters and can be changed later if required.
