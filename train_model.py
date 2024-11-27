import csv
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Load your training data from a CSV file
training_data = []

with open("room_names.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        room1 = row["nuitee_room_name"]
        room2 = row["provider_room_name"]
        label = 1.0  # Similarity label
        training_data.append(InputExample(texts=[room1, room2], label=label))

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# DataLoader for training
train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)

# Define the loss
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='room_mapping_model'
)

print("Model trained and saved at 'room_mapping_model'")
