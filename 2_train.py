import fasttext
import os

# File paths
training_file = "fasttext_training_balanced.txt"
output_model_path = "room_mapping_model"

# Check if the training file exists
if not os.path.exists(training_file):
    raise FileNotFoundError(f"The training file '{training_file}' does not exist.")

# Training parameters
print("Starting model training...")
model = fasttext.train_supervised(
    input=training_file,
    lr=0.1,  # Learning rate
    epoch=25,  # Number of training epochs
    wordNgrams=2,  # Use word n-grams
    bucket=200000,  # Number of hash buckets for n-grams
    dim=100,  # Dimension of word vectors
    loss="softmax"  # Loss function
)

# Save the trained model
model.save_model(output_model_path)
print(f"Model trained and saved at '{output_model_path}'.")

# Test the model
def test_model(model, test_text_pairs):
    for pair in test_text_pairs:
        label, confidence = model.predict(pair, k=1)  # Top 1 prediction
        print(f"Test: {pair} | Predicted: {label[0]} | Confidence: {confidence[0]}")

# Example test cases
test_text_pairs = [
    "Superior King Room with Balcony ||| Superior Room, 1 King Bed, Balcony",
    "Superior King Room with Balcony ||| Deluxe Queen Room with Sea View",
]
print("\nTesting the trained model:")
test_model(model, test_text_pairs)
