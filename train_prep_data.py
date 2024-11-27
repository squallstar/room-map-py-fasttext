import csv
import random
from tqdm import tqdm  # Install tqdm if not already installed: pip install tqdm

# File paths
input_file = "room_names.csv"
output_file = "room_mapping_training_with_labels.txt"

# Load room names
matches = []
rooms = []

print("Loading data from CSV...")
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, quotechar='"', skipinitialspace=True)  # Handles quoted fields
    for row in reader:
        nuitee_room = row["nuitee_room_name"].strip()
        provider_room = row["provider_room_name"].strip()

        # Collect matched pairs
        matches.append((nuitee_room, provider_room))

        # Collect all room names for negative sampling
        rooms.append(nuitee_room)
        rooms.append(provider_room)

print(f"Loaded {len(matches)} positive matches.")
print(f"Collected {len(rooms)} room names for negative sampling.")

# Function to escape and quote room names
def escape_room_name(room_name):
    if "," in room_name or " " in room_name:
        return f'"{room_name}"'
    return room_name

# Create labeled training data
print("Generating labeled dataset...")
with open(output_file, "w", encoding="utf-8") as f:
    # Write positive examples
    print("Writing positive examples...")
    for nuitee_room, provider_room in tqdm(matches, desc="Positive Examples"):
        nuitee_escaped = escape_room_name(nuitee_room)
        provider_escaped = escape_room_name(provider_room)
        f.write(f"__label__MATCH {nuitee_escaped} ||| {provider_escaped}\n")

    # Generate and write negative examples
    print("Writing negative examples...")
    for _ in tqdm(range(len(matches)), desc="Negative Examples"):
        while True:
            random_nuitee = random.choice(rooms)
            random_provider = random.choice(rooms)

            # Ensure the random pair is not in the positive matches
            if (random_nuitee, random_provider) not in matches:
                random_nuitee_escaped = escape_room_name(random_nuitee)
                random_provider_escaped = escape_room_name(random_provider)
                f.write(f"__label__NO_MATCH {random_nuitee_escaped} {random_provider_escaped}\n")
                break  # Exit the loop once a valid negative sample is found

print(f"Labeled dataset saved to {output_file}.")
