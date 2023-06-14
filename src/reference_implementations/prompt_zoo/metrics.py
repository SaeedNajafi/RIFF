"""This module implements different metrics used to evaluate the predictions
for the downstream tasks."""
from typing import Dict

import numpy as np
import pandas as pd
from absl import flags

FLAGS = flags.FLAGS


def sentiment_metric(prediction_file: str) -> Dict[str, float]:
    """Compute the classification accuracy for sentiment classification."""

    df = pd.read_csv(prediction_file, delimiter=",")

    gold_labels = df["gold_class"].tolist()

    # pick the class with the highest score among the possible class labels!
    num_labels = len(set(gold_labels))

    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [label.strip("<s>").strip("</s>").strip() for label in df["potential_class"].tolist()]
    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))

    return_metrics: Dict[str, float] = {}

    if "prediction_score" in df.columns:
        # accuracy based on the scores of the paraphrases
        scores = df["prediction_score"].tolist()
        prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
        max_predictions = np.argmax(prediction_scores, axis=1)
        max_labels = []
        for index in range(len(predictions) // num_labels):
            labels_row = prediction_labels[index]
            max_labels.append(labels_row[max_predictions[index]])

        corrects = 0.0
        total = 0.0
        for index in range(len(predictions) // num_labels):
            total += 1.0
            if gold_labels[index * num_labels] == max_labels[index]:
                corrects += 1.0

        accuracy = corrects / total
        return_metrics["accuracy"] = accuracy

    if "original_prediction_score" in df.columns:
        # accuracy based on the original input sentence.
        scores = df["original_prediction_score"].tolist()
        prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
        max_predictions = np.argmax(prediction_scores, axis=1)
        max_labels = []
        for index in range(len(predictions) // num_labels):
            labels_row = prediction_labels[index]
            max_labels.append(labels_row[max_predictions[index]])

        corrects = 0.0
        total = 0.0
        for index in range(len(predictions) // num_labels):
            total += 1.0
            if gold_labels[index * num_labels] == max_labels[index]:
                corrects += 1.0

        accuracy = corrects / total
        return_metrics["original_accuracy"] = accuracy

    if "all_prediction_scores" in df.columns:
        # accuracy based on all the scores from the original sentence and its paraphrases.
        scores = df["all_prediction_scores"].tolist()
        num_workers = len(scores[0].split(","))

        score_arrays = [[float(s) for s in score.split(",")] for score in scores]
        prediction_scores = np.array(score_arrays).reshape((len(predictions) // num_labels, num_labels, num_workers))

        average_prediction_scores = np.mean(prediction_scores, axis=2)
        max_predictions = np.argmax(average_prediction_scores, axis=1)

        max_labels = []
        for index in range(len(predictions) // num_labels):
            labels_row = prediction_labels[index]
            max_labels.append(labels_row[max_predictions[index]])

        corrects = 0.0
        total = 0.0
        for index in range(len(predictions) // num_labels):
            total += 1.0
            if gold_labels[index * num_labels] == max_labels[index]:
                corrects += 1.0

        all_accuracy = corrects / total
        return_metrics["all_accuracy"] = all_accuracy

    return return_metrics


def classifier_sentiment_metric(prediction_file: str) -> Dict[str, float]:
    """Compute the classification accuracy for sentiment classification where
    we have classifier on top of the LM compared to generation of the classes
    in the decoder."""

    df = pd.read_csv(prediction_file, delimiter=",")
    prediction_indices = df["predicted_class"].tolist()
    class_indices = df["class_index"].tolist()

    corrects = 0.0
    total = 0.0
    for index, gold in enumerate(class_indices):
        total += 1.0
        if gold == prediction_indices[index]:
            corrects += 1.0
    return {"accuracy": corrects / total}
