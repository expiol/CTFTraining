import os
import json
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import re

# Set API key
api_key = 'sk-**********'

# API URL
api_url = '************'

# Set folder path to current directory
folder_path = Path('.')

# Models to compare
models = ["gpt-4", "gpt-3.5-turbo"]

# Initialize results dictionary
results = {model: {'tokens_used': 0, 'correct_count': 0, 'total_count': 0} for model in models}

# Initialize list for failed requests
failed_requests = []

def extract_flag(text):
    """Extract flag from the text."""
    match = re.search(r'flag\{.*?\}', text)
    return match.group(0) if match else None

def call_api(model, message):
    """Call the API and return the response."""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.7
    }
    response = requests.post(f"{api_url}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def clean_invalid_chars(json_string):
    """Remove invalid control characters from JSON string."""
    return re.sub(r'[\x00-\x1f\x7f]', '', json_string)

# Read all JSON files in the folder
for json_file in folder_path.glob('*.json'):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            raw_data = file.read()
            clean_data = clean_invalid_chars(raw_data)
            data = json.loads(clean_data)
    except (json.JSONDecodeError, OSError) as e:
        error_info = {
            "file": str(json_file),
            "error": str(e)
        }
        failed_requests.append(error_info)
        print(f"Failed to read file {json_file}: {e}")
        continue

    description = data.get('description', '')
    correct_flag = data.get('flag', '')

    for model in models:
        gpt_answer = None
        tokens_used = 0

        while not gpt_answer:
            try:
                response_data = call_api(model, description)
                response_text = response_data['choices'][0]['message']['content'].strip()
                tokens_used += response_data['usage']['total_tokens']

                # Try to extract flag
                gpt_answer = extract_flag(response_text)

                if not gpt_answer:
                    # If no flag found, ask again
                    description += "\nI couldn't find the flag in your previous response. Could you please provide the flag again?"

            except requests.exceptions.RequestException as e:
                error_info = {
                    "model": model,
                    "description": description,
                    "error": str(e)
                }
                failed_requests.append(error_info)
                print(f"Error calling API for model {model}: {e}")
                break

        # Update statistics
        results[model]['tokens_used'] += tokens_used
        results[model]['total_count'] += 1
        if gpt_answer == correct_flag:
            results[model]['correct_count'] += 1

# Print results
for model in models:
    correct_count = results[model]['correct_count']
    total_count = results[model]['total_count']
    tokens_used = results[model]['tokens_used']
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Model: {model}")
    print(f"Total Tokens Used: {tokens_used}")
    print(f"Correct Answers: {correct_count}/{total_count}")
    print(f"Accuracy: {accuracy:.2f}")
    print('-' * 40)

# Save failed requests to a JSON file
with open('failed_requests.json', 'w', encoding='utf-8') as f:
    json.dump(failed_requests, f, ensure_ascii=False, indent=4)

# Plot results
fig, ax1 = plt.subplots()

# Plot total tokens used as a bar chart
ax1.set_xlabel('Model')
ax1.set_ylabel('Total Tokens Used', color='tab:blue')
ax1.bar(models, [results[model]['tokens_used'] for model in models], color='tab:blue', alpha=0.6)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:red')
ax2.plot(models, [results[model]['correct_count'] / results[model]['total_count'] if results[model]['total_count'] > 0 else 0 for model in models], color='tab:red', marker='o')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add title to the plot
plt.title('Comparison of Different GPT Models')

# Show plot
plt.show()
