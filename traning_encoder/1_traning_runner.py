import argparse
from typing import Any, Dict, List, Optional, TextIO, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    RobertaConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
)
# from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import evaluate
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    multilabel_confusion_matrix
)

from transformers import TrainerCallback, DataCollatorWithPadding
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser(description="Joke classification training")
parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization"
    )
parser.add_argument(
        "--target_columns",
        nargs="+",
        default=["humor", "offensiveness", "sentiment"],
        help="List of target column names"
    )
parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="FacebookAI/roberta-base",
        help="HuggingFace model checkpoint to use"
    )
parser.add_argument(
        "--data_path",
        type=str,
        default="../data/labeled_jokes_classification_mistral:latest.parquet",
        help="Path to the labeled jokes Parquet file"
    )
parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to use for traning"
    )

args = parser.parse_args()


MAX_LEN = args.max_len
TARGET_COLUMNS = args.target_columns
NUM_TARGETS = len(TARGET_COLUMNS)
MODEL_CHECKPOINT = args.model_checkpoint
data_path = args.data_path
nrows = args.nrows

output_dir = f"./classification_trained_models/{MODEL_CHECKPOINT.split('/')[-1]}/maxlen{MAX_LEN}/{data_path.split('/')[-1].split('.')[0]}/"
os.makedirs(output_dir, exist_ok=True)

# make a log file
log_file = open(os.path.join(output_dir, "training_log.txt"), "w")



def printd(*args: Any, **kwargs: Any) -> None:
    """
    Prints the provided arguments. If the 'file' keyword argument is provided,
    it prints to the file and then to the default standard output.

    Args:
        *args: Positional arguments to be printed.
        **kwargs: Keyword arguments for the `print` function.
    """
    file: Optional[TextIO] = kwargs.get("file", None)

    # Print to the specified file if provided
    print(*args, **kwargs)

    # If 'file' is provided, remove it and print to default stdout
    if file is not None:
        del kwargs["file"]
        print(*args, **kwargs)

class CustomTrainer(Trainer):
    def __init__(self, *args, focal_loss_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_gamma = focal_loss_gamma

    def focal_binary_cross_entropy(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        p = torch.sigmoid(logits)
        # Calculate focal loss
        p_t = torch.where(targets == 1, p, 1 - p)
        log_p_t = torch.log(torch.clamp(p_t, min=1e-4, max=1 - 1e-4))  # Cross Entropy
        loss = - (1 - p_t) ** self.focal_loss_gamma * log_p_t
        return loss.mean()

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute focal loss
        loss = self.focal_binary_cross_entropy(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

# Custom callback to record loss history
class LossHistory(TrainerCallback):
    def __init__(self):
        self.train_losses = []  # to store (global_step, training loss)
        self.eval_losses = []   # to store (global_step, evaluation loss)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log training loss if available
        if logs is not None and "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Log evaluation loss if available
        if metrics is not None and "eval_loss" in metrics:
            self.eval_losses.append((state.global_step, metrics["eval_loss"]))

def plot_loss_history(loss_history, save_path="loss_plot.png"):
    """
    Plots the training and evaluation loss curves stored in the loss_history callback.
    """
    if loss_history.train_losses:
        train_steps, train_loss_values = zip(*loss_history.train_losses)
    else:
        train_steps, train_loss_values = [], []
    
    if loss_history.eval_losses:
        eval_steps, eval_loss_values = zip(*loss_history.eval_losses)
    else:
        eval_steps, eval_loss_values = [], []
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss_values, label="Training Loss", marker='o')
    plt.plot(eval_steps, eval_loss_values, label="Evaluation Loss", marker='o')
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Loss plot saved as '{save_path}'.")

# Function to load data from a parquet file and process targets
def load_data(file_path, nrows=None):
    # Load dataset from a Parquet file
    df = pd.read_parquet(file_path)
    if nrows:
        df = df.head(nrows)
    
    # Cast the target columns to int for classification purposes.
    df[TARGET_COLUMNS] = df[TARGET_COLUMNS].astype(int)
    
    # Drop rows where any value in the target columns isn't 0 or 1.
    # This creates a boolean mask that checks for binary values.
    df = df[df[TARGET_COLUMNS].isin([0, 1]).all(axis=1)]

    df[TARGET_COLUMNS] = df[TARGET_COLUMNS].astype(float)
    
    # Ensure that the 'joke' column is of type string.
    df['joke'] = df['joke'].astype(str)

    # drop duplicates
    df = df.drop_duplicates(subset=['joke'])
    # drop empty jokes
    df = df[df['joke'].str.strip() != '']
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    
    return df


# Custom dataset class for classification
class JokeDataset(Dataset):
    def __init__(self, jokes, targets, tokenizer, max_len):
        self.jokes = jokes
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.jokes)
    
    def __getitem__(self, idx):
        joke = str(self.jokes[idx])
        # Convert target values to float for BCEWithLogitsLoss (they are binary: 0 or 1)
        targets = np.array(self.targets[idx]).astype(np.float32)
        
        encoding = self.tokenizer.encode_plus(
            joke,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(targets, dtype=torch.float)
        }
    

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# Define the compute_metrics function for multi-label classification
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Apply sigmoid to get probabilities
    sigmoid_preds = 1 / (1 + np.exp(-predictions))
    # Threshold probabilities at 0.5 for binary predictions
    binary_preds = (sigmoid_preds > 0.5).astype(int)
    
    # Compute precision, recall, and f1 scores for each target
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(NUM_TARGETS):
        precision = precision_score(labels[:, i], binary_preds[:, i], zero_division=0)
        recall = recall_score(labels[:, i], binary_preds[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], binary_preds[:, i], zero_division=0)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    results = {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list)
    }
    for i, target in enumerate(TARGET_COLUMNS):
        results[f"precision_{target}"] = precision_list[i]
        results[f"recall_{target}"] = recall_list[i]
        results[f"f1_{target}"] = f1_list[i]
    
    return results


def iterative_split_dataframe(df, target_columns, input_columns=None, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Splits a multi-label DataFrame into train, validation, and test sets using iterative stratification.

    Args:
        df (pd.DataFrame): Full dataframe with both input features and target labels.
        target_columns (list): List of column names that are the target labels.
        input_columns (list, optional): List of input feature columns. If None, inferred from df.
        train_size (float): Proportion of data for training.
        val_size (float): Proportion of data for validation.
        test_size (float): Proportion of data for testing.

    Returns:
        (train_df, val_df, test_df): Tuple of DataFrames.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1"

    if input_columns is None:
        input_columns = [col for col in df.columns if col not in target_columns]

    X = df[input_columns]
    y = df[target_columns]

    # First split: train and temp
    temp_ratio = 1 - train_size
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(X.values, y.values, test_size=temp_ratio)

    # Second split: val and test from temp
    val_ratio = val_size / (val_size + test_size)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=1 - val_ratio)

    # Reconstruct DataFrames
    train_df = pd.concat([pd.DataFrame(X_train, columns=input_columns),
                          pd.DataFrame(y_train, columns=target_columns)], axis=1)
    
    val_df = pd.concat([pd.DataFrame(X_val, columns=input_columns),
                        pd.DataFrame(y_val, columns=target_columns)], axis=1)

    test_df = pd.concat([pd.DataFrame(X_test, columns=input_columns),
                         pd.DataFrame(y_test, columns=target_columns)], axis=1)

    return train_df, val_df, test_df


def plot_probability_distributions(probabilities, labels, split_name, target_columns):
    """
    Plots the distribution of predicted probabilities for each target.
    
    Args:
        probabilities (np.ndarray): Array of shape (num_samples, num_targets) with probabilities.
        labels (np.ndarray): Actual labels (not used in plot but could be overlaid).
        split_name (str): Either "Train" or "Test" to label the plot.
        target_columns (list): List of target column names.
    """
    num_targets = len(target_columns)
    plt.figure(figsize=(5 * num_targets, 5))

    for i in range(num_targets):
        plt.subplot(1, num_targets, i + 1)
        sns.histplot(probabilities[:, i], bins=50, kde=True, color='skyblue')
        plt.title(f"{split_name} Set - {target_columns[i]}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}{split_name.lower()}_probability_distributions.png")
    plt.show()


def gen_evalautions(trainer, dataset, name="Dataset", log_file=None):
    """
    Runs prediction + evaluation on `dataset` using `trainer`.
    Logs to `log_file` via printd(..., file=log_file).
    Prints:
      - Classification report (precision/recall/F1 + micro/macro)
      - Subset accuracy (exact‑match)
      - Per‑label accuracy
      - Per‑label confusion matrices
    """
    # 1. Predict
    out = trainer.predict(dataset)
    logits = out.predictions
    labels = out.label_ids

    # 2. Binarize
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    # 3. Classification report
    printd(f"\n=== {name} Classification Report ===", file=log_file)
    printd(classification_report(
        labels,
        preds,
        target_names=TARGET_COLUMNS,
        zero_division=0
    ), file=log_file)

    # 4. Subset (exact‑match) accuracy
    subset_acc = accuracy_score(labels, preds)
    printd(f"{name} Subset Accuracy: {subset_acc:.4f}", file=log_file)

    # 5. Per‑label accuracy
    printd(f"\n{name} Per‑Label Accuracy:", file=log_file)
    for i, label in enumerate(TARGET_COLUMNS):
        acc_i = accuracy_score(labels[:, i], preds[:, i])
        printd(f"  {label:<15} {acc_i:.4f}", file=log_file)

    # 6. Per‑label confusion matrices
    cms = multilabel_confusion_matrix(labels, preds)
    for i, cm in enumerate(cms):
        tn, fp, fn, tp = cm.ravel()
        printd(f"\nLabel '{TARGET_COLUMNS[i]}' Confusion Matrix:", file=log_file)
        printd("       Pred=0    Pred=1", file=log_file)
        printd(f" True=0   {tn:>4}       {fp:>4}", file=log_file)
        printd(f" True=1   {fn:>4}       {tp:>4}", file=log_file)
        
    return probs, labels


def main(data_path, nrows):
    # Load data
    print(f"Loading data from {data_path}")
    df = load_data(data_path, nrows=nrows)

    # Split data into train, validation, and test sets (80%, 10%, 10%)
    train_df, val_df, test_df = iterative_split_dataframe(df, target_columns=TARGET_COLUMNS, input_columns=["joke"], train_size=0.8, val_size=0.1, test_size=0.1)
    
    printd(f"Train set: {len(train_df)} samples", file=log_file)
    printd(f"Validation set: {len(val_df)} samples", file=log_file)
    printd(f"Test set: {len(test_df)} samples", file=log_file)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create datasets
    train_dataset = JokeDataset(
        jokes=train_df['joke'].values,
        targets=train_df[TARGET_COLUMNS].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = JokeDataset(
        jokes=val_df['joke'].values,
        targets=val_df[TARGET_COLUMNS].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_dataset = JokeDataset(
        jokes=test_df['joke'].values,
        targets=test_df[TARGET_COLUMNS].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Configure the model for multi-label classification
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    config.num_labels = NUM_TARGETS
    config.problem_type = "multi_label_classification"
    
    # Add id to label and label to id mappings to the model config
    id2label = {i: label for i, label in enumerate(TARGET_COLUMNS)}
    label2id = {label: i for i, label in enumerate(TARGET_COLUMNS)}
    config.id2label = id2label
    config.label2id = label2id
    print("Mapping id to label:", config.id2label)
    print("Mapping label to id:", config.label2id)
    
    # Initialize model; using AutoModelForSequenceClassification sets up BCEWithLogitsLoss internally.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
        torch_dtype="auto",
        # attn_implementation="eager",
        # reference_compile=False,
    )
    # model = torch.compile(model)
    
    # Optionally, freeze the base model layers if you want to fine-tune only the classification head
    for param in model.parameters():
        param.requires_grad = True

    # Instantiate loss history callback
    loss_history = LossHistory()
    
    # Set up training arguments (note that we now use "f1" as our metric for best model)
    training_args = TrainingArguments(
        output_dir=f'{output_dir}results',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=300,
        weight_decay=0.1,
        logging_dir='./logs',
        logging_steps=150,
        eval_steps=150,
        save_steps=150,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        learning_rate=5.0e-6
        # fp16=True,  # Uncomment if using mixed precision
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), loss_history],
        data_collator=data_collator,
        focal_loss_gamma=1.5,
    )
    

    # Train the model
    printd("Starting training...", file=log_file)
    trainer.train()
    plot_loss_history(loss_history, save_path=f"{output_dir}loss_plot.png")
    
    # Evaluate on test set
    printd("Evaluating on test set...", file=log_file)
    test_results = trainer.evaluate(test_dataset)
    printd("Test results:", test_results, file=log_file)
    
    # Save model
    printd("Saving model...", file=log_file)
    trainer.save_model(f"{output_dir}final_model")
    
    train_sigmoid, train_labels = gen_evalautions(trainer, train_dataset, name="Train", log_file=log_file)
    test_sigmoid, test_labels = gen_evalautions(trainer, test_dataset,  name="Test",  log_file=log_file)


    plot_probability_distributions(train_sigmoid, train_labels, "Train", TARGET_COLUMNS)
    plot_probability_distributions(test_sigmoid, test_labels, "Test", TARGET_COLUMNS)


    
    # Return results for further use if needed
    return {
        "model": model,
        "test_results": test_results
    }


def predict_joke_ratings(joke_text, model_path="./joke_classification_model"):
    """
    Use the trained classification model to predict ratings for a new joke.
    
    Args:
        joke_text (str): The text of the joke to rate.
        model_path (str): Path to the saved model.
    
    Returns:
        dict: Predicted ratings (0/1) for each metric, with labels from the model configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    # Load the model configuration and extract the id2label mapping
    config = AutoConfig.from_pretrained(model_path)
    id2label = config.id2label  # id2label should be a dict with integer keys mapping to target labels
    
    # Load the model using the updated configuration
    model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            config=config,
            # reference_compile=False,
            # attn_implementation="eager",  # optional but often helpful
        )
    model.eval()
    
    # Tokenize the joke text
    encoding = tokenizer.encode_plus(
        joke_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Perform inference
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        logits = outputs.logits.cpu().numpy()[0]
    
    # Convert logits to probabilities and then to binary predictions
    sigmoid_probs = 1 / (1 + np.exp(-logits))
    binary_preds = (sigmoid_probs > 0.5).astype(int)
    
    # Use the id2label mapping from the model configuration to form the results
    results = {id2label[i]: int(pred) for i, pred in enumerate(binary_preds)}
    return results


if __name__ == "__main__":
    # Replace with your actual file path
    results = main(data_path, nrows=nrows)