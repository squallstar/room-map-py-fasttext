import pandas as pd
import fasttext
import re

# Function to preprocess text
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                       # Lowercase
    text = re.sub(r'[^\w\s]', '', text)       # Remove punctuation
    text = re.sub(r'\s+', ' ', text)          # Remove extra spaces
    return text.strip()

# Load CSV data
input_csv = "room_names.csv"  # Replace with your CSV file path
data = pd.read_csv(input_csv)

# Ensure the necessary columns exist
required_columns = ['nuitee_room_name', 'provider_room_name']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Preprocess room names
data['nuitee_room_name'] = data['nuitee_room_name'].apply(preprocess)
data['provider_room_name'] = data['provider_room_name'].apply(preprocess)

# Prepare the FastText training data
output_file = "training_data.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in data.iterrows():
        nuitee_room = row['nuitee_room_name']
        provider_room = row['provider_room_name']
        status = 1

        if status == 1:  # Assuming 1 means "MATCH" and 0 means "MISMATCH"
            label = "__label__MATCH"
        else:
            label = "__label__MISMATCH"

        # Write the training line
        f.write(f"{label} {nuitee_room} {provider_room}\n")

print(f"Training data saved to {output_file}")

# Train the FastText model
model = fasttext.train_supervised(
    input=output_file,
    epoch=10,                # Number of training epochs
    lr=0.1,                  # Learning rate
    wordNgrams=2,            # Use bigrams for context
    verbose=2,               # Verbosity level
    minCount=1               # Include all words
)

# Save the trained model
model_file = "room_mapping_model.bin"
model.save_model(model_file)
print(f"Model trained and saved as '{model_file}'")
