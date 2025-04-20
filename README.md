# CS 588 - Intro To Big Data Computing: Funny Project

**Author:** Sajil Awale

## Project Overview
This project addresses the challenge of automatically assessing humor, offensiveness, and sentiment in jokes from Reddit’s [r/Jokes](https://www.reddit.com/r/Jokes/) community. We leverage a hybrid approach combining powerful large language models (LLMs) for high-quality weak supervision and fine-tuned encoder models for efficient, scalable classification.


## Data
The dataset consists of over 570,000 jokes sourced from Reddit's r/Jokes community.

### Data Labeling
- **10% Subset:** Labeled using decoder-based LLMs (Gemma3:12b and Mistral:7b).
- **Remaining 90%:** Annotated by fine-tuned encoder models (BERT, RoBERTa, MiniLM).

## Project Workflow
![Project Workflow](fig/project_workflow.png)


## Model Experiments
We fine-tuned three transformer-based encoder models:
- **BERT (bert-base-uncased)**
- **RoBERTa (roberta-base)**
- **MiniLM (all-MiniLM-L6-v2)**

## Results
| Model                 | Test Weighted F1-Score |
|-----------------------|------------------------|
| **BERT**              | 0.86                   |
| **RoBERTa**           | 0.86                   |
| **MiniLM**            | 0.85                   |


## Getting Started
### Clone Repository
```bash
git clone --recurse-submodules https://github.com/AwaleSajil/FunnyProject.git
cd FunnyProject/
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Generate Joke Metrics
```bash
cd joke_labelling/
python 2_reddit_joke_discrete_labelling.py
```

### Perform EDA
```bash
cd ../analysing\ data
jupyter nbconvert --to notebook --execute 2_eda_discrete.ipynb
```

### Train Model
```bash
cd ../traning_encoder
python 1_traning_runner.py
```

### Scaled Inference
```bash
cd ../scaling
jupyter nbconvert --to notebook --execute 1_infernce_remaining.ipynb
```

## Hugging Face Integration
### Load Dataset
```python
from datasets import load_dataset
ds = load_dataset("SajilAwale/FunnyData")
```

### Model Inference
```python
from transformers import pipeline

pipe = pipeline("text-classification", model="SajilAwale/FunnyModel", return_all_scores=True)

joke = "I asked my dog what's two minus two. He said nothing."
results = pipe(joke)
print(results)
```


## References
- [The r/Jokes Dataset](https://github.com/orionw/rJokesData)
- [Transformers Library](https://huggingface.co/transformers/)

---

**Contact:** [Sajil Awale](https://github.com/AwaleSajil)

