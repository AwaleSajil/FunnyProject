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
        return [-1, -1, -1]

    system_message = (
        "You are a joke analysis expert. Your task is to evaluate the following joke, collected from Reddit, based on three classification metrics. "
        "For each metric, provide only a single integer value on a separate line, and nothing else—no labels, bullet points, or extra text. "
        "\n\nStrictly follow these guidelines:"
        "\n- Output must contain exactly 3 integer values, one per line."
        "\n- Do NOT include any labels, punctuation, or additional text other than the numbers."
        "\n\nEvaluate the following metrics as follows:"
        "\n1. Humor: Output 1 if the joke is funny, and 0 otherwise."
        "\n2. Offensiveness: Output 1 if the joke is offensive, and 0 otherwise."
        "\n3. Sentiment: Evaluate the joke's sentiment. Output 0 if the joke expresses negative sentiment, and 1 if it expresses positive sentiment."
        "\n\nReturn only the integer values, one per line, exactly as specified."
    )

    payload = {
        "model": model_name,
        "system": system_message,
        "prompt": f"Joke:\n{joke}\n\nMetrics:",
        "stream": False,
        "max_tokens": 20,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return [-1, -1, -1]  # Return a list with placeholders if request fails
    
    if response.status_code == 200:
        try:
            # Parse and clean the response
            metrics = response.json().get("response", "").strip().split("\n")
            metrics = [i.strip() for i in metrics if i.strip()]  # Clean up any empty lines

            # Ensure there are exactly 3 metrics returned
            if len(metrics) != 3:
                print("Error: Incorrect number of metrics returned.")
                return [-1, -1, -1]  # Return placeholders in case of incorrect metrics count

            # Convert the metrics to integers
            metrics = [int(float(i)) for i in metrics]
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error processing response: {e}")
            return [-1, -1, -1]  # Return placeholders if error in processing the response
    else:
        print(f"Error: Received status code {response.status_code}")
        return [-1, -1, -1]  # Return placeholders in case of non-200 response

    return metrics
    

def generate_joke_metrics(data, output_folder="../data/joke_metrics_dataset_classification/", sample_frac=0.1, batch_size=50):
    """
    Generate joke metrics for classification and save them to multiple Parquet files in a folder.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the jokes.
    - output_folder (str): The folder where the Parquet files will be saved.
    - sample_frac (float): The fraction of data to sample for metric generation.
    - batch_size (int): Number of rows to process before saving intermediate results.

    Returns:
    - pd.DataFrame: The DataFrame containing the jokes and their generated classification metrics.
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

        # Generate classification metrics for the batch
        metrics_list = [gen_joke_metrics_llm(joke) for joke in batch['joke']]

        # Convert to DataFrame with appropriate column names
        metrics_df = pd.DataFrame(metrics_list, columns=['humor', 'offensiveness', 'sentiment'])
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
final_result = generate_joke_metrics(data, sample_frac=0.1, batch_size=50)
final_result = final_result[final_result["humor"] > -1]  # Filter out rows where metrics couldn't be generated successfully
final_result.to_parquet("../data/labeled_jokes_classification.parquet")
