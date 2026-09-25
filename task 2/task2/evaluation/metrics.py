import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


NUMBER_OF_CLASSES = 7


def calculate_classification_metrics(
    true_labels,
    predicted_labels,
    number_of_classes=NUMBER_OF_CLASSES,
):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(
        predicted_labels
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        labels=list(range(number_of_classes)),
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(number_of_classes)),
    )

    class_totals = matrix.sum(axis=1)
    class_correct = np.diag(matrix)

    per_class_accuracy = np.divide(
        class_correct,
        class_totals,
        out=np.zeros(
            number_of_classes,
            dtype=float,
        ),
        where=class_totals != 0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "number_of_examples": int(
            len(true_labels)
        ),
        "per_class_accuracy": (
            per_class_accuracy.tolist()
        ),
        "confusion_matrix": matrix.tolist(),
    }

    return metrics


def collect_model_predictions(
    backbone,
    classifier,
    data_loader,
    device,
):
    backbone_was_training = backbone.training
    classifier_was_training = classifier.training

    backbone.eval()
    classifier.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            features = backbone(images)
            logits = classifier(features)

            predictions = logits.argmax(dim=1)

            all_true_labels.extend(
                labels.cpu().tolist()
            )

            all_predicted_labels.extend(
                predictions.cpu().tolist()
            )

    if backbone_was_training:
        backbone.train()

    if classifier_was_training:
        classifier.train()

    return (
        all_true_labels,
        all_predicted_labels,
    )


def evaluate_model(
    backbone,
    classifier,
    data_loader,
    device,
):
    true_labels, predicted_labels = (
        collect_model_predictions(
            backbone=backbone,
            classifier=classifier,
            data_loader=data_loader,
            device=device,
        )
    )

    metrics = calculate_classification_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )

    return metrics


def evaluate_source_domains(
    backbone,
    classifier,
    validation_loaders,
    device,
):
    domain_metrics = {}

    for domain, data_loader in (
        validation_loaders.items()
    ):
        metrics = evaluate_model(
            backbone=backbone,
            classifier=classifier,
            data_loader=data_loader,
            device=device,
        )

        domain_metrics[domain] = metrics

    source_accuracies = [
        metrics["accuracy"]
        for metrics in domain_metrics.values()
    ]

    source_macro_f1_scores = [
        metrics["macro_f1"]
        for metrics in domain_metrics.values()
    ]

    mean_source_accuracy = float(
        np.mean(source_accuracies)
    )

    mean_source_macro_f1 = float(
        np.mean(source_macro_f1_scores)
    )

    result = {
        "domains": domain_metrics,
        "mean_source_accuracy": (
            mean_source_accuracy
        ),
        "mean_source_macro_f1": (
            mean_source_macro_f1
        ),
    }

    return result


def main():
    true_labels = [
        0,
        0,
        1,
        1,
        2,
        2,
    ]

    predicted_labels = [
        0,
        1,
        1,
        1,
        2,
        0,
    ]

    metrics = calculate_classification_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(
        "Per-class accuracy: "
        f"{metrics['per_class_accuracy']}"
    )
    print("Confusion matrix:")

    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()