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
    "msp": (
        r"(a) MSP: $u(x)=1-\max_k p_k(x)$"
    ),
    "mls": (
        r"(b) MLS: $u(x)=-\max_k z_k(x)$"
    ),
    "mahalanobis": (
        "(c) Mahalanobis: minimum class distance"
    )
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
    "all": "#B279A2",
    "neutral": "#6B7280"
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
                row
                for row in rows
                if row["model"] == model_name
                and row["score"] == score_name
            ),
            None
        )

        if match is None:
            raise ValueError(
                "Could not find row for "
                f"{model_name}/{score_name}"
            )

        selected_rows.append(match)

    return selected_rows


def numeric_column(rows, column):
    return np.asarray(
        [
            float(row[column]) * 100.0
            for row in rows
        ],
        dtype=np.float64
    )


def style_axis(axis):
    axis.grid(
        axis="y",
        alpha=0.22,
        linewidth=0.45
    )

    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_bar_labels(
    axis,
    bars,
    decimals=1
):
    for bar in bars:
        height = bar.get_height()

        axis.annotate(
            f"{height:.{decimals}f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height
            ),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.2
        )


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

    minimum = float(np.min(combined))
    maximum = float(np.max(combined))

    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    bins = np.linspace(
        minimum,
        maximum,
        51
    )

    axis.hist(
        known,
        bins=bins,
        density=True,
        alpha=0.58,
        color=COLORS["known"],
        label="CIFAR-10 known"
    )

    axis.hist(
        near,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLORS["near"],
        label="Near unknown"
    )

    axis.hist(
        far,
        bins=bins,
        density=True,
        alpha=0.50,
        color=COLORS["far"],
        label="Far unknown"
    )

    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Validation threshold"
    )

    axis.set_title(
        SCORE_TITLES[score_name],
        fontsize=7.3,
        fontweight="bold",
        pad=4
    )

    axis.set_xlabel(
        "Unknownness score\n"
        "(larger = more unknown)",
        fontsize=6.2,
        labelpad=2
    )

    axis.set_ylabel(
        "Probability density",
        fontsize=6.2
    )

    axis.tick_params(
        axis="both",
        labelsize=5.6,
        length=2.5
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
        color=COLORS["neutral"]
    )

    add_bar_labels(
        axis,
        bars,
        decimals=2
    )

    axis.set_title(
        "(d) Closed-set accuracy",
        fontsize=7.3,
        fontweight="bold",
        pad=4
    )

    axis.set_ylabel(
        "CIFAR-10 accuracy (%)",
        fontsize=6.2
    )

    axis.set_ylim(90, 100)

    axis.tick_params(
        axis="x",
        labelsize=5.8
    )

    axis.tick_params(
        axis="y",
        labelsize=5.6
    )

    style_axis(axis)


def plot_grouped_metrics(
    axis,
    rows,
    columns,
    legend_labels,
    title,
    ylabel,
    show_legend=False
):
    positions = np.arange(
        len(MODEL_LABELS)
    )

    number_of_groups = len(columns)
    total_width = 0.78
    bar_width = (
        total_width / number_of_groups
    )

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
            positions + offsets[index],
            values,
            width=bar_width,
            color=group_colors[index],
            label=legend_label
        )

        add_bar_labels(
            axis,
            bars,
            decimals=1
        )

    axis.set_xticks(positions)

    axis.set_xticklabels(
        MODEL_LABELS,
        fontsize=5.4,
        linespacing=0.9
    )

    axis.set_ylim(0, 100)

    axis.set_title(
        title,
        fontsize=7.3,
        fontweight="bold",
        pad=4
    )

    axis.set_ylabel(
        ylabel,
        fontsize=6.2
    )

    axis.tick_params(
        axis="y",
        labelsize=5.6
    )

    if show_legend:
        axis.legend(
            loc="upper right",
            fontsize=5.4,
            frameon=True,
            borderpad=0.3,
            handlelength=1.4,
            labelspacing=0.25
        )

    style_axis(axis)


def main():
    if not VANILLA_RESULTS.exists():
        raise FileNotFoundError(
            f"Missing file: {VANILLA_RESULTS}"
        )

    if not MODEL_TABLE.exists():
        raise FileNotFoundError(
            f"Missing file: {MODEL_TABLE}"
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
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white"
    })

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 5.4)
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.82,
        wspace=0.42,
        hspace=0.72
    )

    figure.text(
        0.5,
        0.975,
        "Vanilla score distributions: "
        "larger values indicate greater unknownness",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold"
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

    top_legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        fontsize=6.1,
        title=(
            "Evaluation groups and "
            "validation-calibrated decision rule"
        ),
        title_fontsize=6.4,
        frameon=True,
        borderpad=0.4,
        handlelength=1.6,
        columnspacing=1.2
    )

    top_legend.get_frame().set_linewidth(0.6)

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
        ylabel="AUROC (%)",
        show_legend=False
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
        ylabel="Unknown rejection rate (%)",
        show_legend=True
    )

    png_path = (
        OUTPUT_DIRECTORY
        / "compact_task4_results_final.png"
    )

    pdf_path = (
        OUTPUT_DIRECTORY
        / "compact_task4_results_final.pdf"
    )

    figure.savefig(
        png_path,
        dpi=400
    )

    figure.savefig(
        pdf_path
    )

    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()