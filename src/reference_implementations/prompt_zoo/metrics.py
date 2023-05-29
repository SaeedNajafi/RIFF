"""This module implements different metrics used to evaluate the predictions
for the downstream tasks."""
import numpy as np
import pandas as pd
from absl import flags

FLAGS = flags.FLAGS


def sentiment_metric(prediction_file: str) -> float:
    """Compute the classification accuracy for sentiment classification."""

    df = pd.read_csv(prediction_file, delimiter=",")

    gold_labels = df["gold_class"].tolist()

    # pick the class with the highest score among the possible class labels!
    num_labels = len(set(gold_labels))

    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [label.strip("<s>").strip("</s>").strip() for label in df["potential_class"].tolist()]
    scores = df["prediction_score"].tolist()

    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))

    if isinstance(scores[0], str):
        num_workers = len(scores[0].split(","))

        score_arrays = [[float(s) for s in score.split(",")] for score in scores]
        prediction_scores = np.array(score_arrays).reshape((len(predictions) // num_labels, num_labels, num_workers))

        # average_prediction_scores = np.mean(prediction_scores, axis=2)
        # only pick the score for
        average_prediction_scores = prediction_scores[:, :, 0]
        max_predictions = np.argmax(average_prediction_scores, axis=1)

    else:
        prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
        max_predictions = np.argmax(prediction_scores, axis=1)

    max_labels = []
    for index in range(len(predictions) // num_labels):
        labels_row = prediction_labels[index]
        max_labels.append(labels_row[max_predictions[index]])

    corrects = 0.0
    total = 0.0
    wrong_rows = []
    for index in range(len(predictions) // num_labels):
        total += 1.0
        if gold_labels[index * num_labels] == max_labels[index]:
            corrects += 1.0
        else:
            wrong_rows.extend(list(range(index * num_labels, (index + 1) * num_labels, 1)))

    df.iloc[wrong_rows,].to_csv("wrong_rows.csv")
    return corrects / total


def classifier_sentiment_metric(prediction_file: str) -> float:
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
    return corrects / total
