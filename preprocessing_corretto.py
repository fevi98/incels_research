import pandas as pd
import re
import os
from emoji import demojize
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", " p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

def remove_url_rows(input_csv, output_csv, removed_csv):
    df = pd.read_csv(input_csv)
    url_pattern = re.compile(r'^\s*https?://\S+\s*$', re.IGNORECASE)
    mask = df["text"].astype(str).str.match(url_pattern)
    df_filtered = df[~mask]
    df_removed = df[mask]
    df_filtered.to_csv(output_csv, index=False)
    df_removed.to_csv(removed_csv, index=False)
    print(f"Removed {mask.sum()} rows with only URLs")
    print(f"Filtered CSV saved as: {output_csv}")
    print(f"Removed rows CSV saved as: {removed_csv}")

def remove_deleted_rows(input_file, output_file):
    df = pd.read_csv(input_file)
    df_filtered = df[~df.apply(lambda row: row.astype(str).str.contains(r'\[deleted\]|\[removed\]', case=False).any(), axis=1)]
    df_filtered.to_csv(output_file, index=False)
    print(f"Rows containing '[deleted]' or '[removed]' removed; output saved as {output_file}")

def remove_empty_rows(input_file, output_file):
    df = pd.read_csv(input_file)
    df_filtered = df.dropna(subset=["text"])
    df_filtered.to_csv(output_file, index=False)
    print(f"Rows containing no text removed; output saved as: {output_file}")

def normalize_text_column(input_file, output_file):
    df = pd.read_csv(input_file)
    df["text"] = df["text"].astype(str).apply(normalizeTweet)
    df.to_csv(output_file, index=False)
    print(f"Normalized tweets saved to: {output_file}")

def process_dataset(input_csv, output_csv):
    temp_csv_1 = "temp_filtered.csv"
    removed_urls_csv = "removed_urls.csv"

    # Step 1: Remove URL-only rows
    remove_url_rows(input_csv, temp_csv_1, removed_urls_csv)

    # Step 2: Remove '[deleted]' or '[removed]'
    remove_deleted_rows(temp_csv_1, temp_csv_1)

    # Step 3: Remove empty rows
    remove_empty_rows(temp_csv_1, temp_csv_1)

    # Step 4: Normalize tweets
    normalize_text_column(temp_csv_1, output_csv)

    # Cleanup
    os.remove(temp_csv_1)
    print(f"Final cleaned and normalized dataset saved to: {output_csv}")

input_csv = 'dataset.csv'
output_csv = 'filtered_dataset.csv'

process_dataset(input_csv, output_csv)
