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
            else:
                print(f"Error making request for file {filename}: {response.stderr}")

        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")

print("Processing complete.")
