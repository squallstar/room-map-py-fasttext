import pandas as pd
import random

# Load the CSV file
print("Loading CSV file...")
df = pd.read_csv('room_names.csv')
print(f"CSV file loaded successfully. Total rows: {len(df)}")

# Create positive pairs (matches)
print("Generating positive pairs (matches)...")
positive_samples = []
for index, row in df[df['status'] == 1].iterrows():
    nuitee_room = row['nuitee_room_name']
    provider_room = row['provider_room_name']
    if pd.notna(nuitee_room) and pd.notna(provider_room):
        positive_samples.append(f"__label__1 {nuitee_room} ||| {provider_room}")
    if index % 100 == 0:
        print(f"Processed {index} positive samples...")

print(f"Total positive samples generated: {len(positive_samples)}")

# Create negative pairs (non-matches)
print("Generating negative pairs (non-matches)...")
room_names = df['provider_room_name'].dropna().unique()
negative_samples = []
for index, row in df.iterrows():
    nuitee_room = row['nuitee_room_name']
    if pd.notna(nuitee_room):
        provider_room = random.choice(room_names)
        while provider_room == row['provider_room_name']:
            provider_room = random.choice(room_names)  # Ensure it's a mismatch
        negative_samples.append(f"__label__0 {nuitee_room} ||| {provider_room}")
    if index % 100 == 0:
        print(f"Processed {index} negative samples...")

print(f"Total negative samples generated: {len(negative_samples)}")

# Combine and shuffle the samples
print("Combining and shuffling samples...")
all_samples = positive_samples + negative_samples
random.shuffle(all_samples)
print(f"Total samples (positive + negative): {len(all_samples)}")

# Save to a new training file
print("Saving to training file: fasttext_training_balanced.txt...")
with open('fasttext_training_balanced.txt', 'w') as f:
    for index, sample in enumerate(all_samples):
        f.write(sample + "\n")
        if index % 1000 == 0:
            print(f"Written {index} samples to file...")

print("Balanced training file with matches and non-matches created successfully.")
