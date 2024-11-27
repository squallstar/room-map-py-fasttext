import fasttext

# Load the trained model
model = fasttext.load_model("room_mapping_model")

# Predict
def predict_similarity(nuitee_room, provider_room):
    input_text = f"{nuitee_room} {provider_room}"
    prediction = model.predict(input_text)
    return prediction

# Example
nuitee_room = "Superior King Room with Balcony"
provider_room = "Superior Room, 1 King Bed, Balcony"
print(predict_similarity(nuitee_room, provider_room))

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
