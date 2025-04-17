import pandas as pd
from typing import Dict
import time
import requests
from tqdm import tqdm
import os


data = pd.read_csv("../data/rJokesData/data/preprocessed.csv.gz")
data = data[["date", "joke", "score"]]
data['date'] = pd.to_datetime(data['date'], unit='s')


def gen_joke_metrics_llm(joke, metric_dict, url="http://localhost:11434/api/generate", model_name="gemma3:12b",max_joke_len_thresh=5800):
    if len(joke) > max_joke_len_thresh:
        print(f"Joke longer than max character threshold of: {max_joke_len_thresh}")
        return [-1] * len(metric_dict)
    
    metric_defination = "\n".join([f"\n{i}. {k.title()}: {v}" for i, (k, v) in enumerate(metric_dict.items())])

    system_message = (
        "You are a joke analysis expert. Your task is to evaluate the following joke, collected from Reddit, based on several key metrics. "
        "For each metric, provide only a single numerical value on a continuous scale on a separate line, and nothing else—no labels, bullet points, or extra text. "
        "\n\nStrictly follow these guidelines:"
        f"\n- Output must contain exactly {len(metric_dict)} numerical values, one per line."
        "\n- Each value must be a continuous number; if applicable, include at least one decimal point to reflect nuanced differences (avoid rounding to whole numbers unless unavoidable)."
        "\n- Do NOT include any labels, punctuation, or additional text other than the numbers."
        "\n\nEvaluate the following metrics as follows:"
        f"{metric_defination}"
        "\n\nReturn only the numerical values, one per line, exactly as specified."
    )

    payload = {
        "model": model_name,
        "system": system_message,
        "prompt": f"Joke:\n{joke}\n\nMetrics:",
        "stream": False,
        "max_tokens": 20,
        "temperature": 0.8
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return [-1] * len(metric_dict)  # Return a list with placeholders if request fails
    
    if response.status_code == 200:
        try:
            # Parse and clean the response
            metrics = response.json().get("response", "").strip().split("\n")
            metrics = [i.strip() for i in metrics if i.strip()]  # Clean up any empty lines

            # Ensure there are exactly 8 metrics returned
            if len(metrics) != len(metric_dict):
                print("Error: Incorrect number of metrics returned.")
                return [-1] * len(metric_dict)   # Return placeholders in case of incorrect metrics count

            # Attempt to convert all metrics to floats
            metrics = [float(i) for i in metrics]
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error processing response: {e}")
            return [-1] * len(metric_dict)   # Return placeholders if error in processing the response
    else:
        print(f"Error: Received status code {response.status_code}")
        return [-1] * len(metric_dict)   # Return placeholders in case of non-200 response

    return metrics
    

def generate_joke_metrics(data, model_name="gemma3:12b", output_folder="../data/joke_metrics_dataset_regression/", sample_frac=0.1, batch_size=50):
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
    output_folder = f"{output_folder}{model_name}"
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
    metrics_dict = {'humor': "Rate the joke's funniness on a continuous scale from 0 (not funny) to 100 (extraordinarily funny-that it triggers immediate, uncontrollable, laugh-out-loud amusement ). Assess the joke solely on its comedic impact, wit, and delivery. Use the full spectrum to capture even subtle variations in humor. Do not allow any potentially offensive or negative elements to influence your humor evaluation.", 
                    'offensiveness': "Rate how offensive the joke is from 0.0 (not offensive) to 100.0 (extremely offensive). Evaluate whether the joke’s subject matter or phrasing is likely to be perceived as insensitive or harmful by a broad audience.", 
                    'sentiment': "Rate the overall emotional tone of the joke on a continuous scale from -100.0 (very negative) to 100.0 (very positive), with 0.0 being neutral. Consider whether the joke conveys a light-hearted, uplifting, or positive mood versus a negative or harmful one."}
    for start in tqdm(range(0, len(remaining_rows), batch_size), desc="Processing batches"):
        batch = remaining_rows.iloc[start:start + batch_size]

        # Generate metrics for the batch
        metrics_list = [gen_joke_metrics_llm(joke, metrics_dict, model_name=model_name) for joke in batch['joke']]

        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_list, columns=list(metrics_dict.keys()))
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
model_name="mistral:latest"
final_result = generate_joke_metrics(data, model_name=model_name, sample_frac=0.1, batch_size=50)
final_result = final_result[final_result["humor"] > -1]  # Filter out rows where metrics couldn't be generated successfully
final_result.to_parquet(f"../data/labeled_jokes_regression_{model_name}.parquet")