import csv
import os
import json
import re
import time

import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "output"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "summaries.json"
)

games = {
    "Clair Obscur": "data/cleaned/clair_obscur.csv",
    "Overwatch": "data/cleaned/overwatch.csv",
    "Spacebase DF-9": "data/cleaned/spacebase_df9.csv"
}


MODEL_NAME = "facebook/bart-large-cnn"

# Number of representative reviews for Method 3
REPRESENTATIVE_REVIEW_COUNT = 400

# Number of sentences returned by extractive method
EXTRACTIVE_SENTENCES = 10

# Maximum tokens per BART chunk
MAX_CHUNK_TOKENS = 850


# ============================================================
# COMMON WEIGHTS
# ============================================================

CONTENT_WEIGHT = 0.70
PLAYTIME_WEIGHT = 0.20
HELPFUL_WEIGHT = 0.10


# ============================================================
# DATA LOADING
# ============================================================

def parse_boolean(value):

    value = str(value).strip().lower()

    return value in [
        "true",
        "1",
        "yes"
    ]


def parse_number(value):

    try:
        return float(value)

    except (ValueError, TypeError):
        return 0.0


def load_reviews(file_path):

    reviews = []

    with open(
        file_path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            review_text = row["review"].strip()

            if not review_text:
                continue

            reviews.append({
                "review": review_text,

                "recommended": parse_boolean(
                    row["recommended"]
                ),

                "helpful_or_not": parse_boolean(
                    row["helpful_or_not"]
                ),

                "playtime_at_review": parse_number(
                    row["playtime_at_review"]
                )
            })

    return reviews


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.zeros_like(values)

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# REVIEW IMPORTANCE
# ============================================================

def calculate_review_importance(reviews):

    review_texts = [
        review["review"]
        for review in reviews
    ]

    # --------------------------------------------------------
    # 1. Content representativeness
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        review_texts
    )

    centroid = np.asarray(
        matrix.mean(axis=0)
    )

    content_scores = cosine_similarity(
        matrix,
        centroid
    ).flatten()

    content_scores = normalize(
        content_scores
    )


    # --------------------------------------------------------
    # 2. Playtime
    # --------------------------------------------------------

    playtimes = np.array([
        review["playtime_at_review"]
        for review in reviews
    ])

    # Logarithm prevents huge playtime values
    # from dominating the score
    playtime_scores = np.log1p(
        playtimes
    )

    playtime_scores = normalize(
        playtime_scores
    )


    # --------------------------------------------------------
    # 3. Helpfulness
    # --------------------------------------------------------

    helpful_scores = np.array([
        1.0
        if review["helpful_or_not"]
        else 0.0
        for review in reviews
    ])


    # --------------------------------------------------------
    # Final Importance Score
    # --------------------------------------------------------

    importance_scores = (
        CONTENT_WEIGHT * content_scores
        +
        PLAYTIME_WEIGHT * playtime_scores
        +
        HELPFUL_WEIGHT * helpful_scores
    )

    return importance_scores


def attach_importance_scores(reviews):

    scores = calculate_review_importance(
        reviews
    )

    for review, score in zip(
        reviews,
        scores
    ):

        review["importance_score"] = float(
            score
        )

    return reviews


# ============================================================
# METHOD 1
# WEIGHTED TF-IDF EXTRACTIVE
# ============================================================

def split_sentences(text):

    return re.split(
        r"(?<=[.!?])\s+",
        text
    )


def weighted_extractive_summary(
    reviews,
    number_of_sentences=10
):

    sentences = []
    sentence_importance = []
    sentence_recommendation = []


    # Each sentence inherits the importance
    # of the review it came from
    for review in reviews:

        review_sentences = split_sentences(
            review["review"]
        )

        for sentence in review_sentences:

            sentence = sentence.strip()

            if len(sentence.split()) < 5:
                continue

            sentences.append(
                sentence
            )

            sentence_importance.append(
                review["importance_score"]
            )

            sentence_recommendation.append(
                review["recommended"]
            )


    if not sentences:
        return ""


    # Sentence-level TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        sentences
    )

    centroid = np.asarray(
        matrix.mean(axis=0)
    )

    sentence_content_scores = (
        cosine_similarity(
            matrix,
            centroid
        ).flatten()
    )

    sentence_content_scores = normalize(
        sentence_content_scores
    )

    sentence_importance = np.array(
        sentence_importance
    )


    # Combine sentence relevance with
    # review-level importance
    final_scores = (
        0.70 * sentence_content_scores
        +
        0.30 * sentence_importance
    )


    # --------------------------------------------------------
    # Preserve recommendation distribution
    # --------------------------------------------------------

    recommended_ratio = (
        sum(
            review["recommended"]
            for review in reviews
        )
        / len(reviews)
    )

    recommended_target = round(
        number_of_sentences
        * recommended_ratio
    )

    not_recommended_target = (
        number_of_sentences
        - recommended_target
    )


    recommended_indices = [
        i
        for i, value in enumerate(
            sentence_recommendation
        )
        if value
    ]

    not_recommended_indices = [
        i
        for i, value in enumerate(
            sentence_recommendation
        )
        if not value
    ]


    recommended_ranked = sorted(
        recommended_indices,
        key=lambda i: final_scores[i],
        reverse=True
    )

    not_recommended_ranked = sorted(
        not_recommended_indices,
        key=lambda i: final_scores[i],
        reverse=True
    )


    selected_indices = (
        recommended_ranked[
            :recommended_target
        ]
        +
        not_recommended_ranked[
            :not_recommended_target
        ]
    )


    # Restore original sentence order
    selected_indices = sorted(
        selected_indices
    )


    return " ".join(
        sentences[i]
        for i in selected_indices
    )


# ============================================================
# BART MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {device}"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()


# ============================================================
# BART FUNCTIONS
# ============================================================

def create_chunks(
    texts,
    max_tokens=MAX_CHUNK_TOKENS
):

    chunks = []

    current = []
    token_count = 0


    for text in texts:

        tokens = tokenizer.encode(
            text,
            add_special_tokens=False
        )


        # If a single review is larger
        # than one chunk
        if len(tokens) > max_tokens:

            if current:

                chunks.append(
                    " ".join(current)
                )

                current = []
                token_count = 0


            for start in range(
                0,
                len(tokens),
                max_tokens
            ):

                token_slice = tokens[
                    start:
                    start + max_tokens
                ]

                chunks.append(
                    tokenizer.decode(
                        token_slice,
                        skip_special_tokens=True
                    )
                )

            continue


        if (
            token_count + len(tokens)
            > max_tokens
        ):

            if current:

                chunks.append(
                    " ".join(current)
                )

            current = [text]
            token_count = len(tokens)

        else:

            current.append(
                text
            )

            token_count += len(tokens)


    if current:

        chunks.append(
            " ".join(current)
        )


    return chunks


def summarize_chunk(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )


    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


def hierarchical_bart_summary(
    texts,
    label
):

    current_texts = texts
    level = 1


    while True:

        chunks = create_chunks(
            current_texts
        )

        print(
            f"{label} - Level {level}: "
            f"{len(chunks)} chunks"
        )


        summaries = []


        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            print(
                f"{label}: "
                f"{i}/{len(chunks)}"
            )

            summaries.append(
                summarize_chunk(
                    chunk
                )
            )


        if len(summaries) == 1:

            return summaries[0]


        current_texts = summaries
        level += 1


# ============================================================
# METHOD 2
# WEIGHTED FULL BART
# ============================================================

def weighted_full_bart(reviews):

    # BART itself cannot accept a numeric review weight.
    #
    # Therefore every review is still used,
    # but reviews with greater importance are processed first.

    ranked_reviews = sorted(
        reviews,
        key=lambda review:
            review["importance_score"],
        reverse=True
    )

    texts = [
        review["review"]
        for review in ranked_reviews
    ]


    return hierarchical_bart_summary(
        texts,
        label="Weighted Full BART"
    )


# ============================================================
# METHOD 3
# WEIGHTED TF-IDF + BART
# ============================================================

def select_weighted_representative_reviews(
    reviews,
    top_n=400
):

    if len(reviews) <= top_n:

        return reviews


    # Preserve original recommendation ratio

    recommended_reviews = [
        review
        for review in reviews
        if review["recommended"]
    ]

    not_recommended_reviews = [
        review
        for review in reviews
        if not review["recommended"]
    ]


    recommended_ratio = (
        len(recommended_reviews)
        / len(reviews)
    )


    recommended_target = round(
        top_n * recommended_ratio
    )

    not_recommended_target = (
        top_n
        - recommended_target
    )


    # Rank using common importance score

    recommended_reviews = sorted(
        recommended_reviews,
        key=lambda review:
            review["importance_score"],
        reverse=True
    )

    not_recommended_reviews = sorted(
        not_recommended_reviews,
        key=lambda review:
            review["importance_score"],
        reverse=True
    )


    selected = (
        recommended_reviews[
            :recommended_target
        ]
        +
        not_recommended_reviews[
            :not_recommended_target
        ]
    )


    return selected


def weighted_tfidf_bart(reviews):

    selected_reviews = (
        select_weighted_representative_reviews(
            reviews,
            top_n=REPRESENTATIVE_REVIEW_COUNT
        )
    )


    # Keep most important selected reviews first

    selected_reviews = sorted(
        selected_reviews,
        key=lambda review:
            review["importance_score"],
        reverse=True
    )


    texts = [
        review["review"]
        for review in selected_reviews
    ]


    summary = hierarchical_bart_summary(
        texts,
        label="Weighted TF-IDF + BART"
    )


    return (
        summary,
        selected_reviews
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results = {}


for game_name, file_path in games.items():


    print(
        "\n===================================="
    )

    print(
        f"GAME: {game_name}"
    )

    print(
        "===================================="
    )


    reviews = load_reviews(
        file_path
    )


    # --------------------------------------------------------
    # Calculate common importance scores once
    # --------------------------------------------------------

    reviews = attach_importance_scores(
        reviews
    )


    total_reviews = len(
        reviews
    )


    recommended_count = sum(
        review["recommended"]
        for review in reviews
    )


    not_recommended_count = (
        total_reviews
        - recommended_count
    )


    print(
        f"Reviews loaded: {total_reviews}"
    )

    print(
        f"Recommended: {recommended_count}"
    )

    print(
        f"Not Recommended: "
        f"{not_recommended_count}"
    )


    # ========================================================
    # METHOD 1
    # ========================================================

    print(
        "\n[1/3] Weighted TF-IDF Extractive"
    )

    start_time = time.perf_counter()


    method1_summary = (
        weighted_extractive_summary(
            reviews,
            number_of_sentences=
                EXTRACTIVE_SENTENCES
        )
    )


    method1_runtime = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # METHOD 2
    # ========================================================

    print(
        "\n[2/3] Weighted Full BART"
    )

    start_time = time.perf_counter()


    method2_summary = (
        weighted_full_bart(
            reviews
        )
    )


    method2_runtime = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # METHOD 3
    # ========================================================

    print(
        "\n[3/3] Weighted TF-IDF + BART"
    )

    start_time = time.perf_counter()


    (
        method3_summary,
        selected_reviews
    ) = weighted_tfidf_bart(
        reviews
    )


    method3_runtime = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results[game_name] = {

        "dataset": {

            "reviews_analyzed":
                total_reviews,

            "recommended":
                recommended_count,

            "not_recommended":
                not_recommended_count
        },


        "weighting": {

            "content_representativeness":
                CONTENT_WEIGHT,

            "playtime":
                PLAYTIME_WEIGHT,

            "helpfulness":
                HELPFUL_WEIGHT
        },


        "weighted_tfidf": {

            "method":
                "Weighted TF-IDF Extractive",

            "reviews_used":
                total_reviews,

            "runtime_seconds":
                round(
                    method1_runtime,
                    3
                ),

            "summary":
                method1_summary
        },


        "weighted_full_bart": {

            "method":
                "Weighted Full BART",

            "model":
                MODEL_NAME,

            "reviews_used":
                total_reviews,

            "runtime_seconds":
                round(
                    method2_runtime,
                    3
                ),

            "summary":
                method2_summary
        },


        "weighted_tfidf_bart": {

            "method":
                "Weighted TF-IDF + BART",

            "model":
                MODEL_NAME,

            "reviews_used":
                len(selected_reviews),

            "runtime_seconds":
                round(
                    method3_runtime,
                    3
                ),

            "summary":
                method3_summary
        }
    }


    # ========================================================
    # SAVE AFTER EACH GAME
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )


    print(
        f"\nCompleted: {game_name}"
    )

    print(
        f"Method 1 runtime: "
        f"{method1_runtime:.2f}s"
    )

    print(
        f"Method 2 runtime: "
        f"{method2_runtime:.2f}s"
    )

    print(
        f"Method 3 runtime: "
        f"{method3_runtime:.2f}s"
    )


print(
    "\n===================================="
)

print(
    "ALL GAMES COMPLETED"
)

print(
    "===================================="
)

print(
    f"Results saved to: {OUTPUT_FILE}"
)