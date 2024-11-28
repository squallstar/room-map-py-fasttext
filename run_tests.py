import os
import subprocess
import json
from collections import defaultdict

# Directories
input_directory = 'tests/'
output_directory = 'tests/results'
compare_directory = 'tests/previous'
endpoint_url = "http://127.0.0.1:5555/"  # Replace with your endpoint URL

# Ensure the output and compare directories exist
os.makedirs(output_directory, exist_ok=True)

# Initialize counters for improvements and regressions
total_improvements = 0
total_regressions = 0

# Function to load a JSON file
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

# Compare mappings and output differences
def compare_results(current, previous):
    current_mapped = {
        result["roomId"]: len(result.get("mappedRooms", []))
        for result in current.get("Results", [])
    }
    previous_mapped = {
        result["roomId"]: len(result.get("mappedRooms", []))
        for result in previous.get("Results", [])
    }

    differences = defaultdict(dict)
    for room_id in set(current_mapped.keys()).union(previous_mapped.keys()):
        current_count = current_mapped.get(room_id, 0)
        previous_count = previous_mapped.get(room_id, 0)
        if current_count != previous_count:
            differences[room_id]["current"] = current_count
            differences[room_id]["previous"] = previous_count
            differences[room_id]["difference"] = current_count - previous_count

    return differences

# Process each JSON file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.json'):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename.replace('.json', '-response.json'))
        compare_file_path = os.path.join(compare_directory, filename.replace('.json', '-response.json'))

        print(f"Processing file: {filename}")

        # Read the content of the input JSON file
        with open(input_file_path, 'r') as infile:
            json_data = json.load(infile)

        # Make the curl request
        try:
            response = subprocess.run(
                [
                    'curl', '-X', 'POST',
                    endpoint_url,
                    '-H', 'Content-Type: application/json',
                    '-d', json.dumps(json_data)
                ],
                capture_output=True,
                text=True
            )

            if response.returncode == 0:
                # Save the response to a new file
                with open(output_file_path, 'w') as outfile:
                    outfile.write(response.stdout)
                print(f"Response saved to: {output_file_path}")

                # Load the response JSON
                current_response = json.loads(response.stdout)

                # Compare with previous results if available
                if os.path.exists(compare_file_path):
                    previous_response = load_json(compare_file_path)
                    if previous_response:
                        differences = compare_results(current_response, previous_response)

                        # Output differences
                        print(f"Differences for {filename}:")
                        for room_id, diff in differences.items():
                            diff_value = diff["difference"]
                            if diff_value > 0:
                                total_improvements += diff_value
                            elif diff_value < 0:
                                total_regressions += abs(diff_value)

                            print(f"  Room ID: {room_id}")
                            print(f"    > Current model: {diff['current']} - Previous model: {diff['previous']} - Difference: {diff['difference']}")
                else:
                    print(f"No previous results found for {filename}. Skipping comparison.\r")

                print(f"\r")

            else:
                print(f"Error making request for file {filename}: {response.stderr}")

        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")

# Calculate overall metrics
if total_improvements + total_regressions > 0:
    net_improvement = total_improvements - total_regressions
    total_changes = total_improvements + total_regressions
    improvement_ratio = (net_improvement / total_changes) * 100

    # Output the overall results
    print("\nOverall Results:")
    print(f"  Total Improvements: {total_improvements}")
    print(f"  Total Regressions: {total_regressions}")
    print(f"  Net Improvement: {net_improvement}")
    print(f"  Overall Improvement Ratio: {improvement_ratio:.2f}%")
else:
    print("\nNo changes detected across all files.")


print("Processing complete.")
