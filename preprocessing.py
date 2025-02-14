import pandas as pd
import re

def remove_url_rows(input_csv, output_csv, removed_csv):
    df = pd.read_csv(input_csv)
    
    # Define regex pattern to select rows with only URLs
    url_pattern = re.compile(r'^\s*https?://\S+\s*$', re.IGNORECASE)

    # Identify rows that match the pattern
    mask = df["text"].astype(str).str.match(url_pattern)

    # Create the new filtered dataframe; to check the removed rows also a dataframe with removed rows is created
    df_filtered = df[~mask]
    df_removed = df[mask]

    # Save the filtered dataframe to a new CSV file
    df_filtered.to_csv(output_csv, index=False)
    df_removed.to_csv(removed_csv, index=False)

    print(f"Removed {mask.sum()} rows with only URLs")
    print(f"Filtered CSV saved as: {output_csv}")
    print(f"Removed rows CSV saved as: {removed_csv}")

def remove_deleted_rows(input_file, output_file):
    df = pd.read_csv(input_file)

    # Remove rows with [deleted] or [removed]
    df_filtered = df[~df.apply(lambda row: row.astype(str).str.contains(r'\[deleted\]|\[removed\]', case=False).any(), axis=1)]

    # Save the filtered dataframe to a new CSV file
    df_filtered.to_csv(output_file, index=False)

    print(f"Rows containing '[deleted]' or '[removed]' removed; output saved as {output_file}")

def remove_empty_rows(input_file, output_file):
    df = pd.read_csv(input_file)

    # Remove rows with no text
    df_filtered = df.dropna(subset=["text"])

    # Save the filtered dataframe to a new CSV file
    df_filtered.to_csv(output_file, index=False)

    print(f"Rows containing no text removed; output saved as: {output_file}")

def process_dataset(input_csv, output_csv):
    # Step 1: Remove rows with URLs
    removed_urls_csv = "removed_urls.csv"
    remove_url_rows(input_csv, "temp_filtered.csv", removed_urls_csv)

    # Step 2: Remove rows with '[deleted]' or '[removed]'
    remove_deleted_rows("temp_filtered.csv", "temp_filtered.csv")

    # Step 3: Remove rows with no text
    remove_empty_rows("temp_filtered.csv", output_csv)

    # Clean up temporary file
    import os
    os.remove("temp_filtered.csv")
    
    print(f"Final filtered CSV saved as: {output_csv}")

# Example usage
input_csv = 'dataset.csv'  # Replace with your input file path
output_csv = 'filtered_dataset.csv'  # Replace with your desired output file path

process_dataset(input_csv, output_csv)
