import os
import subprocess
import json

# Directory containing JSON files
input_directory = 'tests/'
output_directory = 'tests/results'  # Save responses in a separate directory
endpoint_url = "http://127.0.0.1:5555/"  # Replace with your endpoint URL

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each JSON file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.json'):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename.replace('.json', '-response.json'))

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

                # Load the response JSON and extract metrics
                response_data = json.loads(response.stdout)

                # Count mappedRooms across all Results
                mapped_count = sum(len(result.get('mappedRooms', [])) for result in response_data.get('Results', []))

                # Count Unmapped
                unmapped_count = len(response_data.get('Unmapped', []))

                # Calculate total and percentage
                total_count = mapped_count + unmapped_count
                mapped_percentage = (mapped_count / total_count * 100) if total_count > 0 else 0

                # Print the metrics
                print(f"Metrics for {filename}:")
                print(f"  Mapped Rooms: {mapped_count}")
                print(f"  Unmapped Rooms: {unmapped_count}")
                print(f"  Percentage Mapped: {mapped_percentage:.2f}%\n")
            else:
                print(f"Error making request for file {filename}: {response.stderr}")

        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")

print("Processing complete.")
