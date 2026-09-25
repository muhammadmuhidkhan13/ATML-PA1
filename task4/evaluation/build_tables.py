import csv
from pathlib import Path


STANDARD_SCORES = [
    "msp",
    "mls",
    "energy",
    "mahalanobis"
]


def read_csv(path):
    with open(path, "r", newline="") as file:
        return list(
            csv.DictReader(file)
        )


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def load_model_results(model_name):
    path = (
        Path("task4/results")
        / model_name
        / "posthoc_osr_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing evaluation results: {path}"
        )

    return read_csv(path)


def select_row(rows, score_name):
    matches = [
        row for row in rows
        if row["score"] == score_name
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for score "
            f"{score_name}, found {len(matches)}."
        )

    return matches[0]


def main():
    output_directory = Path(
        "task4/results/final_tables"
    )

    vanilla_rows = load_model_results(
        "vanilla"
    )

    table_one_fields = [
        "model",
        "score",
        "closed_set_accuracy",
        "threshold",
        "known_test_acceptance_rate",
        "near_auroc",
        "far_auroc",
        "all_unknown_auroc",
        "near_unknown_rejection_rate",
        "far_unknown_rejection_rate",
        "all_unknown_rejection_rate"
    ]

    table_one = []

    for score_name in STANDARD_SCORES:
        row = select_row(
            vanilla_rows,
            score_name
        )

        table_one.append({
            field: row[field]
            for field in table_one_fields
        })

    write_csv(
        table_one,
        output_directory
        / "table1_vanilla_scores.csv",
        table_one_fields
    )

    gcsc_rows = load_model_results(
        "gcsc"
    )

    proser_rows = load_model_results(
        "proser"
    )

    comparison_rows = [
        select_row(vanilla_rows, "mls"),
        select_row(gcsc_rows, "mls"),
        select_row(proser_rows, "mls"),
        select_row(
            proser_rows,
            "proser_placeholder"
        )
    ]

    table_two_fields = [
        "model",
        "score",
        "closed_set_accuracy",
        "threshold",
        "known_test_acceptance_rate",
        "near_auroc",
        "far_auroc",
        "all_unknown_auroc",
        "near_unknown_rejection_rate",
        "far_unknown_rejection_rate",
        "all_unknown_rejection_rate"
    ]

    table_two = []

    for row in comparison_rows:
        table_two.append({
            field: row[field]
            for field in table_two_fields
        })

    write_csv(
        table_two,
        output_directory
        / "table2_model_comparison.csv",
        table_two_fields
    )

    print(
        "Saved: "
        "task4/results/final_tables/"
        "table1_vanilla_scores.csv"
    )

    print(
        "Saved: "
        "task4/results/final_tables/"
        "table2_model_comparison.csv"
    )


if __name__ == "__main__":
    main()