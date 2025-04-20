# CS 588 - Intro To Big Data Computing: Funny Project

**Author:** Sajil Awale

## Project Overview
This project addresses the challenge of automatically assessing humor, offensiveness, and sentiment in jokes from Reddit’s [r/Jokes](https://www.reddit.com/r/Jokes/) community. We leverage a hybrid approach combining powerful large language models (LLMs) for high-quality weak supervision and fine-tuned encoder models for efficient, scalable classification.


## Data
The dataset consists of over 570,000 jokes sourced from Reddit's r/Jokes community.

### Data Labeling
- **10% Subset:** Labeled using decoder-based LLMs (Gemma3:12b and Mistral:7b).
- **Remaining 90%:** Annotated by fine-tuned encoder models (BERT, RoBERTa, MiniLM).
- 
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/76713b94-98e8-45a6-83c8-c182b2555e48" width="400">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/b38651e9-cc06-4459-8778-eef43736ea1e" width="400">
    </td>
  </tr>
  <tr>
    <td align="center">Venn Diagram – Gemma3:12b</td>
    <td align="center">Venn Diagram – Mistral:7b</td>
  </tr>
</table>

## Project Workflow
![project_workflow (1)](https://github.com/user-attachments/assets/d6ba4b31-1bcd-4c3a-a53e-8eeb9b70d2f1)


## Model Experiments
We fine-tuned three transformer-based encoder models:
- **BERT (bert-base-uncased)**
- **RoBERTa (roberta-base)**
- **MiniLM (all-MiniLM-L6-v2)**

## Results
| Metric           | Support | BERT_P | BERT_R | BERT_F1 | MiniLM_P | MiniLM_R | MiniLM_F1 | RoBERTa_P | RoBERTa_R | RoBERTa_F1 |
|------------------|---------|--------|--------|---------|----------|----------|-----------|-----------|-----------|------------|
| Humor            | 3,809   | 0.83   | 0.88   | 0.85    | 0.82     | 0.87     | 0.84      | 0.82      | 0.91      | 0.86       |
| Offensiveness    | 1,406   | 0.69   | 0.63   | 0.66    | 0.67     | 0.60     | 0.63      | 0.74      | 0.57      | 0.64       |
| Sentiment        | 4,706   | 0.92   | 0.94   | 0.93    | 0.91     | 0.94     | 0.92      | 0.90      | 0.97      | 0.93       |
| Micro Avg        | 9,921   | 0.85   | 0.87   | 0.86    | 0.84     | 0.86     | 0.85      | 0.85      | 0.89      | 0.87       |
| Macro Avg        | 9,921   | 0.81   | 0.82   | 0.81    | 0.80     | 0.80     | 0.80      | 0.82      | 0.81      | 0.81       |
| Weighted Avg     | 9,921   | 0.85   | 0.87   | 0.86    | 0.84     | 0.86     | 0.85      | 0.85      | 0.89      | 0.86       |
| Samples Avg      | 9,921   | 0.85   | 0.87   | 0.85    | 0.84     | 0.86     | 0.83      | 0.85      | 0.88      | 0.85       |
| Overall Accuracy | 9,921   | 0.70   |   –    |   –     | 0.68     |    –     |     –     | 0.70      |    –      |    –       |


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

