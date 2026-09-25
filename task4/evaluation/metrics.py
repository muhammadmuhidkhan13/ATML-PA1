import numpy as np
from sklearn.metrics import roc_auc_score

from task4.evaluation.thresholds import acceptance_rate
from task4.evaluation.thresholds import rejection_rate


def closed_set_accuracy(predictions, labels):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    return float(
        (predictions == labels).mean()
    )


def known_unknown_auroc(
    known_unknownness,
    unknown_unknownness
):
    known_scores = np.asarray(
        known_unknownness
    )

    unknown_scores = np.asarray(
        unknown_unknownness
    )

    targets = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])

    scores = np.concatenate([
        known_scores,
        unknown_scores
    ])

    return float(
        roc_auc_score(targets, scores)
    )


def operating_point_metrics(
    known_test_unknownness,
    unknown_unknownness,
    threshold
):
    known_acceptance = acceptance_rate(
        known_test_unknownness,
        threshold
    )

    unknown_rejection = rejection_rate(
        unknown_unknownness,
        threshold
    )

    unknown_acceptance = acceptance_rate(
        unknown_unknownness,
        threshold
    )

    return {
        "known_test_acceptance_rate": known_acceptance,
        "unknown_rejection_rate": unknown_rejection,
        "fpr_at_95_tpr": unknown_acceptance
    }