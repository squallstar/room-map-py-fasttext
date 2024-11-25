import json
from flask import Flask, request, jsonify
import fasttext
from scipy.spatial.distance import cosine
import re

# Path to the FastText binary file
EMBEDDINGS_PATH = "cc.en.300.bin"

# Load FastText model
print("Loading FastText model...")
ft_model = fasttext.load_model(EMBEDDINGS_PATH)
print("FastText model loaded.")

# Initialize Flask app
app = Flask(__name__)

# Function to get word vectors
def get_vector(word):
    return ft_model.get_word_vector(word)

# Function to calculate similarity between two strings
def calculate_similarity(word1, word2):
    vector1 = get_vector(word1)
    vector2 = get_vector(word2)
    similarity = 1 - cosine(vector1, vector2)
    return similarity

# Function to clean and normalize room names
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

# Room mapping logic
def map_rooms(input_json):
    reference_catalog = input_json["referenceCatalog"]
    input_catalog = input_json["inputCatalog"]

    results = []

    for ref_property in reference_catalog:
        property_name = ref_property["propertyName"]
        property_id = ref_property["propertyId"]

        for ref_room in ref_property["referenceRoomInfo"]:
            ref_room_name = ref_room["roomName"]
            ref_room_id = ref_room["roomId"]
            clean_ref_room_name = clean_text(ref_room_name)

            mapped_rooms = []

            for supplier in input_catalog:
                supplier_id = supplier["supplierId"]

                for sup_room in supplier["supplierRoomInfo"]:
                    sup_room_name = sup_room["supplierRoomName"]
                    sup_room_id = sup_room["supplierRoomId"]
                    clean_sup_room_name = clean_text(sup_room_name)

                    similarity = calculate_similarity(clean_ref_room_name, clean_sup_room_name)

                    if similarity > 0.8:  # Threshold for similarity
                        mapped_rooms.append({
                            "supplierId": supplier_id,
                            "supplierRoomId": sup_room_id,
                            "supplierRoomName": sup_room_name,
                            "cleanSupplierRoomName": clean_sup_room_name
                        })

            results.append({
                "propertyName": property_name,
                "propertyId": property_id,
                "roomId": ref_room_id,
                "roomName": ref_room_name,
                "cleanRoomName": clean_ref_room_name,
                "roomDescription": "",  # Placeholder for room description
                "mappedRooms": mapped_rooms
            })

    return {"Results": results}

# Define the POST endpoint
@app.route("/", methods=["POST"])
def process_request():
    try:
        input_data = request.get_json()  # Parse JSON input
        if not input_data:
            return jsonify({"error": "Invalid or missing JSON input"}), 400

        # Process the mapping
        result = map_rooms(input_data)

        return jsonify(result)  # Return the result as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
