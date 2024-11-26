import pandas as pd

# Load the CSV file
csv_file = "room_names.csv"  # Replace with your file name
data = pd.read_csv(csv_file)

# Filter rows where provider_room_name exists and both fields are non-empty
filtered_data = data.dropna(subset=['nuitee_room_name', 'provider_room_name'])
filtered_data = filtered_data[
    (filtered_data['nuitee_room_name'].str.strip() != '') &
    (filtered_data['provider_room_name'].str.strip() != '')
]

# Prepare training data: nuitee_room_name (input) and provider_room_name (output)
output_file = "room_mapping_training.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in filtered_data.iterrows():
        f.write(f"{row['nuitee_room_name']}\t{row['provider_room_name']}\n")

print(f"Training data saved to {output_file}")
