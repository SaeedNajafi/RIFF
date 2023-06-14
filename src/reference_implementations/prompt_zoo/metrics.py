"""This module implements different metrics used to evaluate the predictions
for the downstream tasks."""
from typing import Dict

import numpy as np
import pandas as pd
from absl import flags
from scipy.stats import entropy
from sklearn.metrics import balanced_accuracy_score

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


def grips_sentiment_metric(prediction_file: str) -> float:
    """Compute the balanced accuracy + entropy for sentiment classification
    used in grips training."""
    # pick the class with the highest score among the possible class labels!
    df = pd.read_csv(prediction_file, delimiter=",")
    gold_labels = [label.strip() for label in df["gold_class"].tolist()]
    num_labels = len(set(gold_labels))
    golds = np.array(gold_labels).reshape((len(gold_labels) // num_labels, num_labels))[:, 0]

    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [label.strip(" </s>") for label in df["potential_class"].tolist()]
    scores = df["prediction_score"].tolist()

    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))
    prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
    max_predictions = np.argmax(prediction_scores, axis=1)
    max_labels = prediction_labels[:, max_predictions][0]

    per_label_correct = {g_label: 0 for g_label in list(set(gold_labels))}
    total = 0.0
    for index, gold in enumerate(golds):
        total += 1.0
        if gold == max_labels[index]:
            per_label_correct[gold] += 1

    per_label_frequencies = [count / total for count in per_label_correct.values()]
    balanced_acc = balanced_accuracy_score(y_true=golds, y_pred=max_labels)

    # 10 is a factor used in the grips implementation.
    return np.round(100 * balanced_acc, 2) + 10 * entropy(np.array(per_label_frequencies))
