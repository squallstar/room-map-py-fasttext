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

import re
from collections import defaultdict

# Function to clean and normalize room names
def clean_text(room_name):
    if not room_name:
        return "unknown"

    # Helper dictionary for synonyms and normalization
    abbreviation_map = {
        'bdr': 'room', 'bedroom': 'room', 'dbl': 'double', 'dbl room': 'double room',
        'sgl': 'single', 'sgl room': 'single room', 'tpl': 'triple', 'tpl room': 'triple room',
        'trpl': 'triple', 'qd': 'quad', 'quad': 'quad room', 'quadruple': 'quad room',
        'exec': 'executive', 'pres': 'presidential', 'std': 'standard', 'fam': 'family-friendly',
        'rom': 'romantic', 'honeymn': 'honeymoon', 'biz': 'business class', 'prm': 'premium',
        'btq': 'boutique', 'hist': 'historic', 'mod': 'modern', 'high-rise': 'high floor',
        'low-rise': 'low floor', 'ground floor': 'low floor', 'with a view': 'balcony',
        'disability access': 'accessible',
        'breakfast buffet': 'breakfast included'
    }

    # Synonyms normalization
    synonyms = {
        'double-double': 'double room', 'accessible': 'accessible room',
        'family': 'family room', 'connected': 'connected rooms',
        'communicating rooms': 'connected rooms'
    }

    # Predefined terms
    predefined_types = [
        'suite', 'single room', 'double room', 'triple room', 'quad room', 'family room',
        'studio room', 'apartment', 'villa', 'bungalow', 'king room', 'queen room', 'cottage',
        'penthouse', 'loft', 'cabin', 'chalet', 'duplex', 'guesthouse', 'hostel', 'accessible room',
        'connected rooms', 'studio', 'appartment'
    ]

    # Normalize number words to digits
    number_words_to_digits = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }

    # Normalize room name
    room_name = room_name.lower()

    # Replace parentheses with placeholders to preserve inner content
    placeholders = []
    room_name = re.sub(r"\((.*?)\)", lambda m: f"<<{len(placeholders)}>>", room_name)
    placeholders.append(m.group(1) for m in re.finditer(r"\((.*?)\)", room_name))

    # Apply replacements and remove special characters
    room_name = re.sub(r"[^a-zA-Z0-9\s]", " ", room_name)
    room_name = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
                       lambda m: number_words_to_digits[m.group(0)], room_name)
    room_name = re.sub(r"\s+", " ", room_name).strip()

    # Restore parentheses content from placeholders
    room_name = re.sub(r"<<(\d+)>>", lambda m: placeholders[int(m.group(1))], room_name)

    # Apply abbreviation normalization
    for abbr, full_form in abbreviation_map.items():
        room_name = re.sub(fr"\b{abbr}\b", full_form, room_name)

    # Apply synonyms normalization
    for key, value in synonyms.items():
        room_name = re.sub(fr"\b{key}\b", value, room_name)

    # Normalize predefined room types
    for room_type in predefined_types:
        if room_type in room_name:
            return room_type

    # Improved fallback logic
    match = re.search(r"\b(single|double|triple|quad)\b", room_name)
    if match:
        return f"{match.group(0)} room"

    # If no match found, return normalized name
    return room_name.strip()

# Extract details from room names
def extract_room_type(clean_name):
    if not isinstance(clean_name, str):
        clean_name = " ".join(clean_name) if isinstance(clean_name, list) else str(clean_name)
    room_types = [
        'suite', 'single room', 'double room', 'triple room', 'quad room', 'family room',
        'studio room', 'apartment', 'villa', 'bungalow', 'king room', 'queen room', 'cottage', 'singular',
        'penthouse', 'loft', 'cabin', 'chalet', 'duplex', 'guesthouse', 'hostel', 'accessible room',
        'connected rooms', 'studio', 'appartment', 'deluxe suite', 'standard suite'
    ]
    for room_type in room_types:
        if room_type in clean_name:
            return room_type
    return "unknown"

def extract_room_category(clean_name):
    if not isinstance(clean_name, str):
        clean_name = " ".join(clean_name) if isinstance(clean_name, list) else str(clean_name)
    categories = [
        'deluxe', 'superior', 'executive', 'club', 'presidential', 'classic',
        'junior', 'luxury', 'economy', 'standard', 'budget', 'family-friendly',
        'romantic', 'honeymoon', 'business class', 'premium', 'boutique', 'historic', 'modern',
        'high floor', 'low floor', 'prestige', 'style', 'privilege', 'design'
    ]
    matched_categories = [category for category in categories if category in clean_name]
    return matched_categories if matched_categories else "unknown"

def extract_board_type(clean_name):
    if not isinstance(clean_name, str):
        clean_name = " ".join(clean_name) if isinstance(clean_name, list) else str(clean_name)
    board_types = [
        'room only', 'bed and breakfast', 'half board', 'full board',
        'all inclusive', 'self catering', 'board basis', 'breakfast included',
        'dinner included', 'lunch included', 'breakfast & dinner', 'full pension',
        'breakfast for 2', 'free breakfast', 'complimentary breakfast', 'no meals',
        'meal plan available', 'kitchenette', 'full kitchen'
    ]
    for board_type in board_types:
        if board_type in clean_name:
            return board_type
    return "unknown"

def extract_view(clean_name):
    if not isinstance(clean_name, str):
        clean_name = " ".join(clean_name) if isinstance(clean_name, list) else str(clean_name)
    views = [
        'city view', 'sea view', 'garden view', 'courtyard view', 'mountain view',
        'beachfront', 'pool view', 'lake view', 'river view', 'panoramic view',
        'ocean view', 'forest view', 'park view', 'street view', 'skyline view',
        'terrace view', 'courtyard area', 'empire state view', 'fifth avenue', 'seine river view'
    ]
    for view in views:
        if view in clean_name:
            return view
    words = clean_name.split()
    if "view" in words:
        index = words.index("view")
        if index > 0:
            return f"{words[index - 1]} view"
    return "unknown"

def extract_bed_types(clean_name):
    if not isinstance(clean_name, str):
        clean_name = " ".join(clean_name) if isinstance(clean_name, list) else str(clean_name)
    bed_types = [
        'single bed', 'double bed', 'queen bed', 'king bed', 'twin bed', 'twin beds', 'double beds',
        'bunk bed', 'double sofa bed', 'sofa bed', 'futon', 'murphy bed', 'queen',
        'full bed', 'california king bed', 'kingsize', 'queensize', 'twin sofa bed', 'twin sofabed',
        'day bed', 'trundle bed', 'extra bed', 'cot', 'rollaway bed', 'single sofa bed', 'sofabed',
        'queen beds', 'king beds', 'kingsize bed', 'kingsize beds', 'queen size bed', 'queensize bed',
        'queen size beds', 'queensize beds', 'king size bed', 'king size beds'
    ]
    found_beds = [bed_type for bed_type in bed_types if bed_type in clean_name]
    return found_beds if found_beds else ["unknown"]

def extract_amenities(clean_name):
    if not isinstance(clean_name, str):
        clean_name = " ".join(clean_name) if isinstance(clean_name, list) else str(clean_name)
    amenities = [
        'wifi', 'air conditioning', 'heating', 'kitchen', 'workspace', 'gym', 'pool',
        'free parking', 'pet-friendly', 'washer', 'dryer', 'balcony', 'fireplace',
        'accessible', 'elevator', 'security', 'private entrance', 'smoke alarm',
        'carbon monoxide alarm', 'first aid kit', 'safety card', 'fire extinguisher',
        'no smoking', 'beach access', 'ski-in/ski-out', 'spa', 'hot tub', 'waterfront',
        'terrace', 'smart TV', 'streaming services', 'mini-bar',
        'coffee maker', 'soundproofing', 'private pool', 'plunge pool', 'bidet',
        'jacuzzi', 'ensuite bathroom', 'patio', 'garden access', 'roof access',
        'private dock', 'hammock', 'game console', 'board games', 'book collection', 'club access'
    ]
    found_amenities = [amenity for amenity in amenities if amenity in clean_name]
    return found_amenities if found_amenities else ["unknown"]

def normalize_room_name(room_name):
    clean_name = clean_text(room_name)
    return {
        "roomType": extract_room_type(clean_name),
        "roomCategory": extract_room_category(clean_name),
        "board": extract_board_type(clean_name),
        "view": extract_view(clean_name),
        "bedType": extract_bed_types(clean_name),
        "amenities": extract_amenities(clean_name)
    }

def compute_embeddings(texts):
    """
    Compute embeddings for a list of texts using the FastText model.
    """
    return {text: ft_model.get_sentence_vector(text) for text in texts}

def calculate_match_outcome(ref_room, sup_room):
    """
    Calculate the match outcome between a reference room and a supplier room.
    Returns a dictionary of matching attributes.
    """

    def is_partial_match(ref_value, sup_value):
        """
        Determine if there is a partial match between two lists or strings.
        """
        if isinstance(ref_value, list) and isinstance(sup_value, list):
            return bool(set(ref_value).intersection(sup_value))  # Check for overlap
        if isinstance(ref_value, str) and isinstance(sup_value, str):
            return ref_value in sup_value or sup_value in ref_value
        return False

    # Match outcomes
    outcome = {
        "matchedRoomType": ref_room["normalizedRoom"]["roomType"] == sup_room["normalizedRoom"]["roomType"],
        "matchedRoomCategory": ref_room["normalizedRoom"]["roomCategory"] == sup_room["normalizedRoom"]["roomCategory"],
        "matchedView": ref_room["normalizedRoom"]["view"] == sup_room["normalizedRoom"]["view"],
        "matchedAmenities": ref_room["normalizedRoom"]["amenities"] == sup_room["normalizedRoom"]["amenities"],
        "bedTypes": ref_room["normalizedRoom"]["bedType"] == sup_room["normalizedRoom"]["bedType"]
    }

    # Handle partial matches
    if not outcome["matchedRoomCategory"] and is_partial_match(
        ref_room["normalizedRoom"]["roomCategory"], sup_room["normalizedRoom"]["roomCategory"]
    ):
        outcome["matchedRoomCategory"] = "partial"

    if not outcome["matchedView"] and is_partial_match(
        ref_room["normalizedRoom"]["view"], sup_room["normalizedRoom"]["view"]
    ):
        outcome["matchedView"] = "partial"

    if not outcome["matchedAmenities"] and is_partial_match(
        ref_room["normalizedRoom"]["amenities"], sup_room["normalizedRoom"]["amenities"]
    ):
        outcome["matchedAmenities"] = "partial"

    if not outcome["bedTypes"] and is_partial_match(
        ref_room["normalizedRoom"]["bedType"], sup_room["normalizedRoom"]["bedType"]
    ):
        outcome["bedTypes"] = "partial"

    return outcome

def add_match_to_results(ref_room, match, results):
    """
    Add a matched room to the results.
    If the reference room already exists in the results, append the match;
    otherwise, create a new entry for the reference room.
    """
    # Check if the reference room already exists in the results
    for result in results:
        if result["roomId"] == ref_room["roomId"]:
            # Append the new match to the existing reference room
            result["mappedRooms"].append(match)
            return

    # If the reference room is not found, create a new result entry
    results.append({
        "propertyName": ref_room["propertyName"],
        "propertyId": ref_room["propertyId"],
        "roomId": ref_room["roomId"],
        "roomName": ref_room["roomName"],
        "cleanRoomName": ref_room["cleanRoomName"],
        "roomDescription": ref_room["normalizedRoom"],
        "mappedRooms": [match]
    })


def map_rooms_with_multiple_passes(input_json):
    reference_catalog = input_json["referenceCatalog"]
    input_catalog = input_json["inputCatalog"]

    # Precompute embeddings and normalize reference rooms
    reference_rooms = []
    reference_embeddings = {}
    for ref_property in reference_catalog:
        for ref_room in ref_property["referenceRoomInfo"]:
            clean_ref_room_name = clean_text(ref_room["roomName"])
            reference_rooms.append({
                "propertyName": ref_property["propertyName"],
                "propertyId": ref_property["propertyId"],
                "roomId": ref_room["roomId"],
                "roomName": ref_room["roomName"],
                "cleanRoomName": clean_ref_room_name,
                "normalizedRoom": normalize_room_name(ref_room["roomName"])
            })
            if clean_ref_room_name not in reference_embeddings:
                reference_embeddings[clean_ref_room_name] = ft_model.get_sentence_vector(clean_ref_room_name)

    # Normalize and preprocess supplier rooms
    supplier_rooms = []
    supplier_embeddings = {}
    for supplier in input_catalog:
        for sup_room in supplier["supplierRoomInfo"]:
            clean_sup_room_name = clean_text(sup_room["supplierRoomName"])
            supplier_rooms.append({
                "supplierId": supplier["supplierId"],
                "supplierRoomId": sup_room["supplierRoomId"],
                "supplierRoomName": sup_room["supplierRoomName"],
                "cleanRoomName": clean_sup_room_name,
                "normalizedRoom": normalize_room_name(sup_room["supplierRoomName"])
            })
            if clean_sup_room_name not in supplier_embeddings:
                supplier_embeddings[clean_sup_room_name] = ft_model.get_sentence_vector(clean_sup_room_name)

    # Track matched supplier room IDs and results
    mapped_supplier_room_ids = set()
    results = []
    unmapped_rooms = []

    # Helper function to determine match based on outcome and pass criteria
    def is_match(outcome, pass_type):
        if pass_type == "First Pass":
            return (
                outcome.get("matchedRoomType") is True and
                outcome.get("matchedRoomCategory") is True and
                (outcome.get("matchedView") in [True, None]) and
                (outcome.get("matchedAmenities") in [True, None]) and
                (outcome.get("bedTypes") in [True, None])
            )
        elif pass_type == "Second Pass":
            return (
                outcome.get("matchedRoomType") is True and
                (outcome.get("matchedRoomCategory") in [True, "partial", None]) and
                (outcome.get("matchedView") in [True, None]) and
                (outcome.get("matchedAmenities") in [True, "partial", None]) and
                (outcome.get("bedTypes") in [True, "partial", None])
            )
        elif pass_type == "Third Pass":
            return (
                outcome.get("matchedRoomType") in [True, "partial"] and
                outcome.get("matchedRoomCategory") in [True, "partial", None, "supplierInfo", "refInfo"] and
                outcome.get("matchedView") in [True, "partial", None, "supplierInfo", "refInfo"] and
                outcome.get("matchedAmenities") in [True, "partial", None, "supplierInfo", "refInfo"] and
                outcome.get("bedTypes") in [True, "partial", None, "supplierInfo", "refInfo"]
            )
        return False

    # Helper function to perform matching pass
    def perform_matching(reference_rooms, supplier_rooms, pass_type):
        match_count = 0
        for ref_room in reference_rooms:
            for sup_room in supplier_rooms:
                if sup_room["supplierRoomId"] in mapped_supplier_room_ids:
                    continue  # Skip already matched supplier rooms

                # Calculate match outcome
                outcome = calculate_match_outcome(ref_room, sup_room)

                if is_match(outcome, pass_type):
                    match_count += 1
                    match = {
                        "pass": pass_type,
                        "supplierRoomId": sup_room["supplierRoomId"],
                        "supplierRoomName": sup_room["supplierRoomName"],
                        "cleanRoomName": sup_room["cleanRoomName"],
                        "matchAttributes": outcome,
                        "roomDescription": sup_room["normalizedRoom"]
                    }
                    add_match_to_results(ref_room, match, results)
                    mapped_supplier_room_ids.add(sup_room["supplierRoomId"])

        return match_count

    # First Pass: Strict Matching
    first_pass_matches = perform_matching(reference_rooms, supplier_rooms, "First Pass")

    # Second Pass: Relaxed Matching
    remaining_supplier_rooms = [room for room in supplier_rooms if room["supplierRoomId"] not in mapped_supplier_room_ids]
    second_pass_matches = perform_matching(reference_rooms, remaining_supplier_rooms, "Second Pass")

    # Third Pass: Broad Matching
    remaining_supplier_rooms = [room for room in supplier_rooms if room["supplierRoomId"] not in mapped_supplier_room_ids]
    third_pass_matches = perform_matching(reference_rooms, remaining_supplier_rooms, "Third Pass")

    # Fourth Pass: Cosine Similarity Matching
    remaining_supplier_rooms = [room for room in supplier_rooms if room["supplierRoomId"] not in mapped_supplier_room_ids]
    fourth_pass_matches = 0
    for sup_room in remaining_supplier_rooms:
        best_match = None
        best_similarity = -1
        supplier_embedding = supplier_embeddings[sup_room["cleanRoomName"]]

        for ref_room in reference_rooms:
            ref_embedding = reference_embeddings[ref_room["cleanRoomName"]]
            similarity = cosine_similarity(supplier_embedding, ref_embedding)
            similarity = float(similarity)

            if similarity > best_similarity and similarity > 0.9:  # Threshold for cosine similarity
                best_similarity = similarity
                best_match = ref_room

        if best_match:
            fourth_pass_matches += 1
            match = {
                "pass": "Fourth Pass (Cosine Similarity)",
                "supplierRoomId": sup_room["supplierRoomId"],
                "supplierRoomName": sup_room["supplierRoomName"],
                "cleanRoomName": sup_room["cleanRoomName"],
                "similarity": best_similarity,
                "roomDescription": sup_room["normalizedRoom"]
            }
            add_match_to_results(best_match, match, results)
            mapped_supplier_room_ids.add(sup_room["supplierRoomId"])

    # Collect unmapped rooms
    unmapped_rooms = [room for room in supplier_rooms if room["supplierRoomId"] not in mapped_supplier_room_ids]

    return {
        "Results": results,
        "UnmappedRooms": unmapped_rooms,
        "Counts": {
            "TotalSupplierRooms": len(supplier_rooms),
            "FirstPassMatches": first_pass_matches,
            "SecondPassMatches": second_pass_matches,
            "ThirdPassMatches": third_pass_matches,
            "FourthPassMatches": fourth_pass_matches,
            "MappedSupplierRooms": len(mapped_supplier_room_ids),
            "UnmappedSupplierRooms": len(unmapped_rooms)
        }
    }



@app.route("/", methods=["POST"])
def process_request_trained_fasttext():
    try:
        input_data = request.get_json()  # Parse JSON input
        if not input_data:
            return jsonify({"error": "Invalid or missing JSON input"}), 400

        # Process the mapping
        result = map_rooms_with_multiple_passes(input_data)

        return jsonify(result)  # Return the result as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
