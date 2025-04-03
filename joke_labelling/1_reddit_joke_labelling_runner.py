import pandas as pd
from typing import Dict
import time
import requests
from tqdm import tqdm
import os


data = pd.read_csv("../data/rJokesData/data/preprocessed.csv.gz")
data = data[["date", "joke", "score"]]
data['date'] = pd.to_datetime(data['date'], unit='s')


def gen_joke_metrics_llm(joke, url="http://localhost:11434/api/generate", model_name="gemma3:12b", max_joke_len_thresh=5800):
    if len(joke) > max_joke_len_thresh:
        print(f"Joke longer than max character threshold of: {max_joke_len_thresh}")
        return [-1] * 8

    system_message = (
        "You are a joke analysis expert. "
        "Your task is to evaluate the following joke based on several key metrics. "
        "For each metric, provide only the numerical value (a continuous scale) on a separate line, without any text, labels, bullet points, or explanations. "
        ""
        "Strictly follow these guidelines:"
        "- Output must contain exactly **8 numerical values**, one per line."
        "- For each metric, feel free to use any number on a continuous scale within the specified range. "
        "- Ensure that each number is a single value, not prefixed with labels or any other text. "
        "- **Do NOT** include any labels, bullet points, explanations, or additional text."
        ""
        "The metrics you need to evaluate are as follows: "
        "1. Humor: Rate the joke's funniness from 0.0 (not funny) to 100.0 (extremely funny). Considering cleverness, surprise, timing, relevance, originality, and engagement."
        "2. Offensiveness: Rate how offensive the joke is from 0.0 (not offensive) to 100.0 (highly offensive). "
        "3. Clarity: Rate how easy the joke is to understand, considering sentence structure and any cultural/contextual knowledge required. Rate from 0.0 (hard to understand) to 100.0 (easy to understand). "
        "4. Surprise Factor: Rate how unexpected the punchline is from 0.0 (completely predictable) to 100.0 (highly surprising), based on how well it defies audience expectations. "
        "5. Relatability: Rate how well a general audience can connect with the joke based on personal experience or cultural context. Rate from 0.0 (not relatable) to 100.0 (highly relatable). "
        "6. Novelty: Rate how original the joke is from 0.0 (common or overused) to 100.0 (highly original), considering whether the joke brings a fresh perspective or new approach. "
        "7. Conciseness: Rate how efficiently the joke delivers its punchline from 0.0 (too long-winded) to 100.0 (perfectly concise). "
        "8. Sentiment: Rate the overall emotional tone of the joke from -100.0 (very negative) to 100.0 (very positive), with 0.0 being neutral. "
        ""
        "Return **only the numbers, one per line**, without any extra text."
    )

    payload = {
        "model": model_name,
        "system": system_message,
        "prompt": f"Joke:\n{joke}\n\nMetrics:",
        "stream": False,
        "max_tokens": 100,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return [-1] * 8  # Return a list with placeholders if request fails
    
    if response.status_code == 200:
        try:
            # Parse and clean the response
            metrics = response.json().get("response", "").strip().split("\n")
            metrics = [i.strip() for i in metrics if i.strip()]  # Clean up any empty lines

            # Ensure there are exactly 8 metrics returned
            if len(metrics) != 8:
                print("Error: Incorrect number of metrics returned.")
                return [-1] * 8  # Return placeholders in case of incorrect metrics count

            # Attempt to convert all metrics to floats
            metrics = [float(i) for i in metrics]
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error processing response: {e}")
            return [-1] * 8  # Return placeholders if error in processing the response
    else:
        print(f"Error: Received status code {response.status_code}")
        return [-1] * 8  # Return placeholders in case of non-200 response

    return metrics
    

def generate_joke_metrics(data, output_folder="./data/joke_metrics_dataset_1/", sample_frac=0.1, batch_size=50):
    """
    Generate joke metrics and save them to multiple Parquet files in a folder.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the jokes.
    - output_folder (str): The folder where the Parquet files will be saved.
    - sample_frac (float): The fraction of data to sample for metric generation.
    - batch_size (int): Number of rows to process before saving intermediate results.

    Returns:
    - pd.DataFrame: The DataFrame containing the jokes and their generated metrics.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Sample the data
    sampled_data = data.sample(frac=sample_frac, random_state=42)
    sampled_data["joke"] = sampled_data["joke"].fillna("")

    # Load existing data if any Parquet files exist
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.parquet')]
    if existing_files:
        processed_indices = set()
        for file in existing_files:
            file_path = os.path.join(output_folder, file)
            existing_data = pd.read_parquet(file_path)
            processed_indices.update(existing_data.index)
    else:
        processed_indices = set()

    # Get the remaining rows that have not been processed
    remaining_rows = sampled_data.loc[~sampled_data.index.isin(processed_indices)]

    # If there are no remaining rows, return the existing data
    if remaining_rows.empty:
        return pd.concat([pd.read_parquet(os.path.join(output_folder, f)) for f in existing_files])

    # Process and save in batches
    new_data_list = []
    for start in tqdm(range(0, len(remaining_rows), batch_size), desc="Processing batches"):
        batch = remaining_rows.iloc[start:start + batch_size]

        # Generate metrics for the batch
        metrics_list = [gen_joke_metrics_llm(joke) for joke in batch['joke']]

        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_list, columns=[
            'humor', 'offensiveness', 'clarity', 'surprise_factor', 'relatability', 'novelty', 'conciseness', 'sentiment'
        ])
        metrics_df.index = batch.index

        # Append metrics to batch
        batch = pd.concat([batch, metrics_df], axis=1)
        new_data_list.append(batch)

        # Save each batch to a separate Parquet file
        batch_file_name = f"batch_{int(time.time() * 1000)}.parquet"
        batch_file_path = os.path.join(output_folder, batch_file_name)
        batch.to_parquet(batch_file_path)

    # After processing all batches, read all Parquet files and combine them
    final_data = pd.concat([pd.read_parquet(os.path.join(output_folder, f)) for f in os.listdir(output_folder) if f.endswith('.parquet')])

    return final_data


# Assuming 'data' is your original DataFrame
final_result = generate_joke_metrics(data)
final_result = final_result[final_result["humor"] > -1]
final_result.to_parquet("../data/labeled_jokes.parquet")