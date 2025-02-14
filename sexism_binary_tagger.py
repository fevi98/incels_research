import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('NLP-LTU/bertweet-large-sexism-detector')
tokenizer = AutoTokenizer.from_pretrained('NLP-LTU/bertweet-large-sexism-detector') 
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation = True, max_length=512)

# Paths to your input and output CSV files
input_file = 'filtered_dataset.csv'
output_file = 'sexism_labeled.csv'

# Open the input CSV file and create the output CSV file
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile, delimiter='\t')

    # Assuming the first row is header, copy it to the output file
    header = next(reader)
    writer.writerow(header + ['Sexism Prediction'])

    for row in reader:
        text = row[0]  # Assuming the text is in the first column; change as necessary
        prediction = classifier(text)[0]  # Classifier returns a list of dictionaries, take the first result
        
        # Extract label prediction (use 'label' to find "sexist" or "not sexist")
        label_pred = prediction['label']
        
        # Write the original row and the prediction label to the output CSV
        writer.writerow(row + [label_pred])

print(f"Predictions saved to {output_file}")
