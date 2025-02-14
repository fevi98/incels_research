import pandas as pd
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm

# Load model and tokenizer
model_name = 'sangkm/augmented-go-emotions-plus-other-datasets-fine-tuned-distilroberta-v3'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load CSV file
input_file = "sexism_labeled.csv"  # Replace with the actual file path
df = pd.read_csv(input_file)

# Ensure 'text' column exists
if "text" not in df.columns:
    raise ValueError("CSV file must contain a 'text' column.")

# Get total number of rows and define progress checkpoint
total_rows = len(df)
progress_step = max(1, total_rows // 20)  # 5% of total rows
start_time = time.time()  # Record start time

# Process text with progress tracking
emotions = []
for i, text in enumerate(tqdm(df["text"], desc="Processing", unit="row")):
    # Predict with truncation enabled inside the pipeline
    pred = classifier(text, truncation=True, max_length=512)[0]['label']
    emotions.append(pred)

    # Print progress update every 5%
    if (i + 1) % progress_step == 0 or (i + 1) == total_rows:
        elapsed_time = time.time() - start_time
        print(f"Progress: {((i + 1) / total_rows) * 100:.1f}% - Elapsed time: {elapsed_time:.2f} seconds")

# Add "emotions" column to DataFrame
df["emotions"] = emotions

# Save results to a new CSV file
output_file = "emotions_predictions.csv"
df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
