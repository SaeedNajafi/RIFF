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
    """Compute the classification accuracy for sentiment classification. We
    report the following metrics:

    1 - accuracy: is the classification accuracy obtained by averaging the prediction scores across
                  the paraphrases of the input.
    2 - original_accuracy: is the classification accuracy obtained by the prediction score
                  of the original input text.
    3 - all_accuracy: is the classification accuracy obtained by averaging the prediction scores
                  from the paraphrases and the original input text.
    """

    df = pd.read_csv(prediction_file, delimiter=",")

    gold_labels = [str(label) for label in df["gold_class"].tolist()]

    # pick the class with the highest score among the possible class labels!
    num_labels = len(set(gold_labels))

    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [str(label).strip("<s>").strip("</s>").strip() for label in df["potential_class"].tolist()]
    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))

    return_metrics: Dict[str, float] = {}

    metrics = {
        "prediction_score": "accuracy",
        "original_prediction_score": "original_accuracy",
        "all_prediction_score": "all_accuracy",
    }

    for metric_column, metric in metrics.items():
        if metric_column in df.columns:
            scores = df[metric_column].tolist()
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
            return_metrics[metric] = accuracy

    return return_metrics


def grips_sentiment_metric(prediction_file: str) -> float:
    """Compute the balanced accuracy + entropy for sentiment classification
    used in grips training."""
    df = pd.read_csv(prediction_file, delimiter=",")

    gold_labels = [str(label) for label in df["gold_class"].tolist()]

    # pick the class with the highest score among the possible class labels!
    num_labels = len(set(gold_labels))
    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [str(label).strip("<s>").strip("</s>").strip() for label in df["potential_class"].tolist()]
    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))
    scores = df["prediction_score"].tolist()
    prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
    max_predictions = np.argmax(prediction_scores, axis=1)
    max_labels = []
    golds = []
    for index in range(len(predictions) // num_labels):
        labels_row = prediction_labels[index]
        max_labels.append(labels_row[max_predictions[index]])
        golds.append(gold_labels[index * num_labels])

    per_label_correct = {g_label: 0 for g_label in list(set(golds))}
    total = 0.0
    for index in range(len(predictions) // num_labels):
        total += 1.0
        if golds[index] == max_labels[index]:
            per_label_correct[golds[index]] += 1

    per_label_frequencies = [count / total for count in per_label_correct.values()]
    balanced_acc = balanced_accuracy_score(y_true=np.array(golds), y_pred=np.array(max_labels))

    # 10 is a factor used in the grips implementation.
    return np.round(100 * balanced_acc, 2) + 10 * entropy(np.array(per_label_frequencies))
