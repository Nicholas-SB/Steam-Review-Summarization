import os
import pandas as pd
import re

RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

DATASETS = [
    "clair_obscur.csv",
    "overwatch.csv",
    "spacebase_df9.csv"
]

def clean_steam_ascii(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Strip Steam BBCode tags
    text = re.sub(r'\[/?(?:b|i|u|h1|h2|h3|spoiler|url|list|olist|quote|code).*?\]', '', text, flags=re.IGNORECASE)
    
    # 2. Eradicate Steam ASCII Art
    text = re.sub(r'[\u2800-\u28FF\u2500-\u259F\u25A0-\u25FF]', '', text)
    
    # 3. Remove standard decorative ASCII borders/strings
    text = re.sub(r'[-=\_\*\~+]{4,}', ' ', text)
    
    # 4. Reduce excessive repeating characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 5. Strip Mojibake and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # 6. Collapse excessive whitespace and newlines
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

# 7. Filter out junk reviews
def is_meaningful_text(text):
    if len(text) < 5: return False 
    alphanum_count = sum(c.isalnum() for c in text)
    return (alphanum_count / len(text)) > 0.65

def process_dataset(filename):

    # Example:
    # data/raw/clair_obscur.csv
    input_file = os.path.join(
        RAW_DIR,
        filename
    )

    # Example:
    # data/cleaned/clair_obscur.csv
    output_file = os.path.join(
        CLEANED_DIR,
        filename
    )

    print(f"\nProcessing: {filename}")

    # Load raw data
    df = pd.read_csv(
        input_file,
        encoding="utf-8-sig"
    )

    # Apply existing cleaning function
    df["clean_review"] = df["review"].apply(
        clean_steam_ascii
    )

    # Filter out junk reviews
    df_summarization_ready = df[
        df["clean_review"].apply(
            is_meaningful_text
        )
    ].copy()

# 8. Select only the four requested columns
    columns_to_keep = ['clean_review', 'recommended', 'helpful_or_not', 'playtime_at_review']
    df_final = df_summarization_ready[columns_to_keep]

    df_final = df_final.rename(
        columns={
                "clean_review": "review"
        }
    )

        # Export cleaned dataset
    df_final.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Successfully exported: {output_file}"
    )

def main():

    # Create data/cleaned if it does not exist
    os.makedirs(
        CLEANED_DIR,
        exist_ok=True
    )

    # Process all three games
    for filename in DATASETS:
        process_dataset(filename)


if __name__ == "__main__":
    main()
