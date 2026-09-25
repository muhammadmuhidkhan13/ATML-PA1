import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DEFAULT_SEED = 6304
TRAIN_FRACTION = 0.70
LOGISTIC_REGRESSION_C = 1.0


def to_numpy(features):
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()

    return np.asarray(features, dtype=np.float32)


def validate_features(
    source_features,
    target_features,
):
    if source_features.ndim != 2:
        raise ValueError(
            "source_features must have shape "
            "[number_of_examples, feature_dimension]."
        )

    if target_features.ndim != 2:
        raise ValueError(
            "target_features must have shape "
            "[number_of_examples, feature_dimension]."
        )

    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            "Source and target features must have "
            "the same feature dimension."
        )

    if len(source_features) < 2 or len(target_features) < 2:
        raise ValueError(
            "At least two source and two target "
            "examples are required."
        )


def create_balanced_domain_dataset(
    source_features,
    target_features,
    seed=DEFAULT_SEED,
):
    source_features = to_numpy(source_features)
    target_features = to_numpy(target_features)

    validate_features(
        source_features=source_features,
        target_features=target_features,
    )

    number_per_domain = min(
        len(source_features),
        len(target_features),
    )

    random_generator = np.random.default_rng(seed)

    source_indices = random_generator.choice(
        len(source_features),
        size=number_per_domain,
        replace=False,
    )

    target_indices = random_generator.choice(
        len(target_features),
        size=number_per_domain,
        replace=False,
    )

    balanced_source = source_features[source_indices]
    balanced_target = target_features[target_indices]

    features = np.concatenate(
        [balanced_source, balanced_target],
        axis=0,
    )

    domain_labels = np.concatenate(
        [
            np.zeros(number_per_domain, dtype=np.int64),
            np.ones(number_per_domain, dtype=np.int64),
        ]
    )

    return features, domain_labels, number_per_domain


def measure_domain_separability(
    source_features,
    target_features,
    seed=DEFAULT_SEED,
):
    (
        features,
        domain_labels,
        number_per_domain,
    ) = create_balanced_domain_dataset(
        source_features=source_features,
        target_features=target_features,
        seed=seed,
    )

    (
        training_features,
        testing_features,
        training_labels,
        testing_labels,
    ) = train_test_split(
        features,
        domain_labels,
        train_size=TRAIN_FRACTION,
        random_state=seed,
        stratify=domain_labels,
    )

    classifier = LogisticRegression(
        C=LOGISTIC_REGRESSION_C,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )

    classifier.fit(
        training_features,
        training_labels,
    )

    predicted_domains = classifier.predict(
        testing_features
    )

    held_out_accuracy = accuracy_score(
        testing_labels,
        predicted_domains,
    )

    return {
        "domain_separability_accuracy": float(
            held_out_accuracy
        ),
        "number_per_domain": int(number_per_domain),
        "training_examples": int(
            len(training_labels)
        ),
        "testing_examples": int(len(testing_labels)),
        "train_fraction": TRAIN_FRACTION,
        "logistic_regression_c": (
            LOGISTIC_REGRESSION_C
        ),
        "seed": int(seed),
    }


def main():
    random_generator = np.random.default_rng(
        DEFAULT_SEED
    )

    source_features = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=(100, 512),
    )

    target_features = random_generator.normal(
        loc=0.5,
        scale=1.0,
        size=(150, 512),
    )

    result = measure_domain_separability(
        source_features=source_features,
        target_features=target_features,
    )

    print(result)


if __name__ == "__main__":
    main()
