import json
from flask import Flask, request, jsonify
import fasttext
from scipy.spatial.distance import cosine
import re
from sentence_transformers import SentenceTransformer, util
from torch.nn.functional import normalize

# Load the trained model
model = SentenceTransformer("room_mapping_model")

# Initialize Flask app
app = Flask(__name__)

# Function to clean and normalize room names
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

ref_room_name = "Superior King Room"
sup_room_name = "Style Room - Room Only"

nuitee_embedding = normalize(model.encode(ref_room_name, convert_to_tensor=True), p=2, dim=0)
supplier_embedding = normalize(model.encode(sup_room_name, convert_to_tensor=True), p=2, dim=0)

similarity = util.pytorch_cos_sim(nuitee_embedding, supplier_embedding).item()
print(f"Similarity: {similarity}")

def map_rooms_trained(input_json):
    reference_catalog = input_json["referenceCatalog"]
    input_catalog = input_json["inputCatalog"]

    results = []
    unmapped_rates = []

    # Precompute embeddings for reference rooms
    reference_embeddings = {}
    for ref_property in reference_catalog:
        for ref_room in ref_property["referenceRoomInfo"]:
            ref_room_name = clean_text(ref_room["roomName"])
            reference_embeddings[ref_room["roomId"]] = normalize(model.encode(ref_room_name, convert_to_tensor=True), p=2, dim=0)

    # Precompute embeddings for supplier rooms
    supplier_embeddings = {}
    for input_property in input_catalog:
        for supplier_room in input_property["supplierRoomInfo"]:
            supplier_room_name = clean_text(supplier_room["supplierRoomName"])
            supplier_embeddings[supplier_room["supplierRoomId"]] = normalize(model.encode(supplier_room_name, convert_to_tensor=True), p=2, dim=0)

    # Perform mapping
    for ref_property in reference_catalog:
        property_name = ref_property["propertyName"]
        property_id = ref_property["propertyId"]

        for ref_room in ref_property["referenceRoomInfo"]:
            ref_room_id = ref_room["roomId"]
            ref_room_name = ref_room["roomName"]
            clean_ref_room_name = clean_text(ref_room_name)

            # Get precomputed embedding
            nuitee_embedding = reference_embeddings[ref_room_id]

            mapped_rooms = []

            for input_property in input_catalog:
                supplier_id = input_property["supplierId"]

                for supplier_room in input_property["supplierRoomInfo"]:
                    sup_room_id = supplier_room["supplierRoomId"]
                    sup_room_name = supplier_room["supplierRoomName"]

                    # Get precomputed embedding
                    provider_embedding = supplier_embeddings[sup_room_id]

                    # Calculate similarity
                    similarity = util.pytorch_cos_sim(nuitee_embedding, provider_embedding).item()

                    if similarity > 0.8:  # Threshold for similarity
                        mapped_rooms.append({
                            "supplierId": supplier_id,
                            "supplierRoomId": sup_room_id,
                            "supplierRoomName": sup_room_name,
                            "cleanSupplierRoomName": clean_text(sup_room_name),
                            "similarity": float(similarity)
                        })

            results.append({
                "propertyName": property_name,
                "propertyId": property_id,
                "roomId": ref_room_id,
                "roomName": ref_room_name,
                "cleanRoomName": clean_ref_room_name,
                "roomDescription": "",
                "mappedRooms": mapped_rooms
            })

    # Identify unmapped rates
    mapped_supplier_room_ids = {
        mapped["supplierRoomId"]
        for result in results
        for mapped in result["mappedRooms"]
    }

    for input_property in input_catalog:
        for supplier_room in input_property["supplierRoomInfo"]:
            if supplier_room["supplierRoomId"] not in mapped_supplier_room_ids:
                unmapped_rates.append({
                    "supplierId": input_property["supplierId"],
                    "supplierRoomId": supplier_room["supplierRoomId"],
                    "supplierRoomName": supplier_room["supplierRoomName"]
                })

    return {
        "Results": results,
        "Unmapped": unmapped_rates
    }


@app.route("/", methods=["POST"])
def process_request_trained():
    try:
        input_data = request.get_json()  # Parse JSON input
        if not input_data:
            return jsonify({"error": "Invalid or missing JSON input"}), 400

        # Process the mapping
        result = map_rooms_trained(input_data)

        return jsonify(result)  # Return the result as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=False)
