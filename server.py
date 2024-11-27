import json
from flask import Flask, request, jsonify
import fasttext
import re
import numpy as np

# Load the FastText model
model_path = "cc.en.300.bin"
ft_model = fasttext.load_model(model_path)

# Initialize Flask app
app = Flask(__name__)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to clean and normalize room names
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

def compute_embeddings(texts):
    """
    Compute embeddings for a list of texts using the FastText model.
    """
    return {text: ft_model.get_sentence_vector(text) for text in texts}


def map_rooms_trained_fasttext(input_json):
    reference_catalog = input_json["referenceCatalog"]
    input_catalog = input_json["inputCatalog"]

    # Precompute embeddings for reference rooms
    reference_embeddings = {}
    ref_room_mapping = {}
    for ref_property in reference_catalog:
        for ref_room in ref_property["referenceRoomInfo"]:
            clean_ref_room_name = clean_text(ref_room["roomName"])
            if clean_ref_room_name not in reference_embeddings:
                reference_embeddings[clean_ref_room_name] = ft_model.get_sentence_vector(clean_ref_room_name)
                ref_room_mapping[clean_ref_room_name] = {
                    "propertyName": ref_property["propertyName"],
                    "propertyId": ref_property["propertyId"],
                    "roomId": ref_room["roomId"],
                    "roomName": ref_room["roomName"],
                    "cleanRoomName": clean_ref_room_name,
                }

    # Precompute embeddings for supplier rooms
    supplier_embeddings = {}
    for supplier in input_catalog:
        for sup_room in supplier["supplierRoomInfo"]:
            clean_sup_room_name = clean_text(sup_room["supplierRoomName"])
            if clean_sup_room_name not in supplier_embeddings:
                supplier_embeddings[clean_sup_room_name] = ft_model.get_sentence_vector(clean_sup_room_name)

    # Initialize results and unmapped rates
    results = {}
    unmapped_rates = []

    # Map supplier rooms to all reference rooms that meet the threshold
    for supplier in input_catalog:
        supplier_id = supplier["supplierId"]

        for sup_room in supplier["supplierRoomInfo"]:
            sup_room_name = sup_room["supplierRoomName"]
            sup_room_id = sup_room["supplierRoomId"]
            clean_sup_room_name = clean_text(sup_room_name)

            supplier_embedding = supplier_embeddings[clean_sup_room_name]

            # Track whether the supplier room is mapped
            is_mapped = False

            for clean_ref_room_name, ref_embedding in reference_embeddings.items():
                similarity = cosine_similarity(supplier_embedding, ref_embedding)

                if similarity > 0.75:  # Threshold for matching
                    is_mapped = True
                    ref_room = ref_room_mapping[clean_ref_room_name]
                    ref_room_key = (ref_room["propertyId"], ref_room["roomId"])

                    if ref_room_key not in results:
                        results[ref_room_key] = {
                            "propertyName": ref_room["propertyName"],
                            "propertyId": ref_room["propertyId"],
                            "roomId": ref_room["roomId"],
                            "roomName": ref_room["roomName"],
                            "cleanRoomName": ref_room["cleanRoomName"],
                            "roomDescription": "",  # Placeholder for room description
                            "mappedRooms": [],
                        }

                    results[ref_room_key]["mappedRooms"].append({
                        "supplierId": supplier_id,
                        "supplierRoomId": sup_room_id,
                        "supplierRoomName": sup_room_name,
                        "cleanSupplierRoomName": clean_sup_room_name,
                        "similarity": float(similarity),
                    })

            if not is_mapped:
                # Mark as unmapped if no match exceeds the threshold
                unmapped_rates.append({
                    "supplierId": supplier_id,
                    "supplierRoomId": sup_room_id,
                    "supplierRoomName": sup_room_name,
                    "cleanSupplierRoomName": clean_sup_room_name,
                    "similarity": float(similarity) if similarity > 0 else 0,
                    "closestMatch": None,  # Optional: Store the closest match if needed
                })


    # Sort unmapped rates by similarity in descending order
    unmapped_rates.sort(key=lambda x: x["similarity"], reverse=True)

    # Convert results dictionary to list
    final_results = list(results.values())

    return {"Results": final_results, "Unmapped": unmapped_rates}


@app.route("/", methods=["POST"])
def process_request_trained_fasttext():
    try:
        input_data = request.get_json()  # Parse JSON input
        if not input_data:
            return jsonify({"error": "Invalid or missing JSON input"}), 400

        # Process the mapping
        result = map_rooms_trained_fasttext(input_data)

        return jsonify(result)  # Return the result as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
