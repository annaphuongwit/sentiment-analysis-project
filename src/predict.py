import argparse
import logging
import sys
from typing import Any

import numpy as np
from joblib import load
from numpy.typing import NDArray

# --- Logging config ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger(__name__)


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


# Other functions will go here
def predict_texts(
    classifier: Any, input_texts: list[str]
) -> tuple[list[int], list[float | None]]:
    """Return labels and probability-of-positive for each text."""
    preds: NDArray[Any] = classifier.predict(input_texts)
    if hasattr(classifier, "predict_proba"):
        probs_arr: NDArray[np.float64] = classifier.predict_proba(input_texts)[:, 1]
        probs = [float(p) for p in probs_arr.tolist()]
    else:
        probs = [None] * len(input_texts)
    return preds.astype(int).tolist(), probs


def format_prediction_lines(
    texts: list[str], preds: list[int], probs: list[float | None]
) -> list[str]:
    """Return tab-separated CLI output lines for each input text."""
    lines: list[str] = []
    for text, pred, prob in zip(texts, preds, probs):
        if prob is None:
            lines.append(f"{pred}\t{text}")
        else:
            lines.append(f"{pred}\t{prob:.3f}\t{text}")
    return lines


# --- Safe wrapper ---
def safe_predict_texts(
    classifier: Any, input_texts: list[str]
) -> tuple[list[int], list[float | None]]:
    """Return predictions safely; handle empty or invalid input."""
    if not input_texts:
        # print("⚠️ No input text provided.")
        log.warning("⚠️ No input text provided.")
        return [], []
    try:
        return predict_texts(classifier, input_texts)
    except Exception:
        # print(f"⚠️ Prediction error: {e}")
        log.warning("⚠️ Prediction error: {e}")
        return [], []


def summarize_predictions(preds: list[int]) -> None:
    """Show 📊 summary of 👍 positive vs 👎 negative predictions."""
    summary = (
    f"\n📊 {len(preds)} texts | 👍 {sum(preds)} pos | "
    f"👎 {len(preds) - sum(preds)} neg\n")
    log.info(summary)



def main(model_path: str, input_texts: list[str]) -> None:
    classifier = load_model(model_path)
    preds, probs = predict_texts(classifier, input_texts)
    for line in format_prediction_lines(input_texts, preds, probs):
        log.info(line)

    preds, probs = safe_predict_texts(classifier, input_texts)

    summarize_predictions(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    args = parser.parse_args()

    if not args.text:
        parser.print_usage()
        log.info(
            "\nerror: at least one text is required.\n"
            "example:\n  python -m src.predict --model models/sentiment.joblib "
            '"I love this!" "This is terrible."\n',
            file=sys.stderr,
        )

        sys.exit(2)

    main(model_path=args.model, input_texts=args.text)
