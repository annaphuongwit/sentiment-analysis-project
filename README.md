# Sentiment Analysis Pipeline

## Setup

### Option 1: Python venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Option 2: Conda
conda create -n sentiment-env python=3.11 -y
conda activate sentiment-env
pip install -r requirements.txt

## Train
python src/train.py --data data/train.csv --out models/sentiment.joblib

# Predict
python src/predict.py "I absolutely loved it" "That was awful"
# Output format: label  probability  text
# Example:
# 1    0.982    I absolutely loved it
# 0    0.015    That was awful


# sentiment-analysis-Pipeline
## Predict
Run the predictions in the command line. Use any sentence you like.
python src/predict.py " That was the best"  " I'm happy "Whoa! That's wonderfull"  " That's too excellent"

## 🧠 How to Run the Predictor
To make a quick sentiment prediction:
```bash
python src/predict.py --model models/sentiment.joblib "I love this!" "This is bad."


### In Code
Add docstrings to functions like `format_prediction_lines`:
```python
def format_prediction_lines(...):
    """
    Formats the model predictions for CLI output.
    Example line:
        1    0.932    I love this changing and this product!
    """

