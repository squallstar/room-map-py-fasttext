from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# Load your training data
training_data = []
with open("room_mapping_training.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:  # Ensure there are exactly two parts
            nuitee, provider = parts
            training_data.append(InputExample(texts=[nuitee, provider], label=1.0))


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

#model.save("room_mapping_model")