import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


VANILLA_RESULTS = Path(
    "task4/results/vanilla/posthoc_scores.pt"
)

MODEL_TABLE = Path(
    "task4/results/final_tables/"
    "table2_model_comparison.csv"
)

OUTPUT_DIRECTORY = Path(
    "task4/results/presentation_figures"
)


SCORE_NAMES = [
    "msp",
    "mls",
    "mahalanobis"
]


SCORE_TITLES = {
    "msp": "(a) MSP unknownness",
    "mls": "(b) MLS unknownness",
    "mahalanobis": "(c) Mahalanobis distance"
}


MODEL_LABELS = [
    "Vanilla\nMLS",
    "GCSC\nMLS",
    "PROSER\nMLS",
    "PROSER\nPlaceholder"
]


COLORS = {
    "known": "#4C78A8",
    "near": "#F58518",
    "far": "#54A24B",
    "all": "#B279A2"
}


def to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()

    return np.asarray(values)


def load_model_rows(path):
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    for row in rows:
        row["model"] = row["model"].strip().lower()
        row["score"] = row["score"].strip().lower()

    required_rows = [
        ("vanilla", "mls"),
        ("gcsc", "mls"),
        ("proser", "mls"),
        ("proser", "proser_placeholder")
    ]

    selected_rows = []

    for model_name, score_name in required_rows:
        match = next(
            (
                row for row in rows
                if row["model"] == model_name
                and row["score"] == score_name
            ),
            None
        )

        if match is None:
            raise ValueError(
                "Could not find row for "
                f"{model_name}/{score_name} in {path}"
            )

        selected_rows.append(match)

    return selected_rows


def numeric_column(rows, column):
    return np.asarray(
        [float(row[column]) * 100.0 for row in rows],
        dtype=np.float64
    )


def add_bar_labels(
    axis,
    bars,
    decimals=1,
    fontsize=9
):
    for bar in bars:
        height = bar.get_height()

        axis.annotate(
            f"{height:.{decimals}f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


def style_axis(axis):
    axis.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.8
    )

    axis.set_axisbelow(True)

    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)


def plot_score_distribution(
    axis,
    score_data,
    score_name
):
    known = to_numpy(
        score_data[score_name]["test"]
    )

    near = to_numpy(
        score_data[score_name]["near_unknown"]
    )

    far = to_numpy(
        score_data[score_name]["far_unknown"]
    )

    threshold = float(
        score_data[score_name]["threshold"]
    )

    combined = np.concatenate([
        known,
        near,
        far,
        np.asarray([threshold])
    ])

    lower_limit = float(np.min(combined))
    upper_limit = float(np.max(combined))

    if lower_limit == upper_limit:
        lower_limit -= 0.5
        upper_limit += 0.5

    bins = np.linspace(
        lower_limit,
        upper_limit,
        51
    )

    axis.hist(
        known,
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLORS["known"],
        label="CIFAR-10 known"
    )

    axis.hist(
        near,
        bins=bins,
        density=True,
        alpha=0.48,
        color=COLORS["near"],
        label="Near unknown"
    )

    axis.hist(
        far,
        bins=bins,
        density=True,
        alpha=0.48,
        color=COLORS["far"],
        label="Far unknown"
    )

    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Validation threshold"
    )

    axis.set_title(
        SCORE_TITLES[score_name],
        fontsize=14,
        fontweight="bold"
    )

    axis.set_xlabel(
        "Unknownness score",
        fontsize=11
    )

    axis.set_ylabel(
        "Density",
        fontsize=11
    )

    axis.tick_params(
        axis="both",
        labelsize=9
    )

    style_axis(axis)


def plot_closed_set_accuracy(
    axis,
    rows
):
    model_rows = rows[:3]

    labels = [
        "Vanilla",
        "GCSC",
        "PROSER"
    ]

    values = numeric_column(
        model_rows,
        "closed_set_accuracy"
    )

    bars = axis.bar(
        labels,
        values,
        width=0.62,
        color=[
            COLORS["known"],
            COLORS["near"],
            COLORS["far"]
        ]
    )

    add_bar_labels(
        axis,
        bars,
        decimals=2,
        fontsize=10
    )

    axis.set_title(
        "(d) Closed-set performance",
        fontsize=14,
        fontweight="bold"
    )

    axis.set_ylabel(
        "CIFAR-10 accuracy (%)",
        fontsize=11
    )

    axis.set_ylim(90, 100)
    axis.tick_params(labelsize=10)

    style_axis(axis)


def plot_grouped_metrics(
    axis,
    rows,
    columns,
    legend_labels,
    title,
    ylabel
):
    x_positions = np.arange(
        len(MODEL_LABELS)
    )

    number_of_groups = len(columns)
    total_width = 0.78
    bar_width = total_width / number_of_groups

    offsets = (
        np.arange(number_of_groups)
        - (number_of_groups - 1) / 2
    ) * bar_width

    group_colors = [
        COLORS["known"],
        COLORS["near"],
        COLORS["far"]
    ]

    for index, (
        column,
        legend_label
    ) in enumerate(
        zip(columns, legend_labels)
    ):
        values = numeric_column(
            rows,
            column
        )

        bars = axis.bar(
            x_positions + offsets[index],
            values,
            width=bar_width,
            color=group_colors[index],
            label=legend_label
        )

        add_bar_labels(
            axis,
            bars,
            decimals=1,
            fontsize=8
        )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        MODEL_LABELS,
        fontsize=9
    )

    axis.set_ylim(0, 100)

    axis.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    axis.set_ylabel(
        ylabel,
        fontsize=11
    )

    axis.legend(
        loc="upper right",
        fontsize=9,
        frameon=True
    )

    axis.tick_params(
        axis="y",
        labelsize=9
    )

    style_axis(axis)


def main():
    if not VANILLA_RESULTS.exists():
        raise FileNotFoundError(
            f"Missing score file: {VANILLA_RESULTS}"
        )

    if not MODEL_TABLE.exists():
        raise FileNotFoundError(
            f"Missing comparison table: {MODEL_TABLE}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    score_data = torch.load(
        VANILLA_RESULTS,
        map_location="cpu",
        weights_only=False
    )

    model_rows = load_model_rows(
        MODEL_TABLE
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.0,
        "savefig.facecolor": "white",
        "figure.facecolor": "white"
    })

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    figure.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.10,
        top=0.90,
        wspace=0.27,
        hspace=0.38
    )

    for axis, score_name in zip(
        axes[0],
        SCORE_NAMES
    ):
        plot_score_distribution(
            axis,
            score_data,
            score_name
        )

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
        fontsize=11,
        frameon=True
    )

    plot_closed_set_accuracy(
        axes[1, 0],
        model_rows
    )

    plot_grouped_metrics(
        axis=axes[1, 1],
        rows=model_rows,
        columns=[
            "near_auroc",
            "far_auroc",
            "all_unknown_auroc"
        ],
        legend_labels=[
            "Near",
            "Far",
            "All unknowns"
        ],
        title="(e) Open-set ranking",
        ylabel="AUROC (%)"
    )

    plot_grouped_metrics(
        axis=axes[1, 2],
        rows=model_rows,
        columns=[
            "near_unknown_rejection_rate",
            "far_unknown_rejection_rate",
            "all_unknown_rejection_rate"
        ],
        legend_labels=[
            "Near",
            "Far",
            "All unknowns"
        ],
        title="(f) Calibrated rejection",
        ylabel="Unknown rejection rate (%)"
    )

    png_path = (
        OUTPUT_DIRECTORY
        / "compact_task4_results.png"
    )

    pdf_path = (
        OUTPUT_DIRECTORY
        / "compact_task4_results.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()