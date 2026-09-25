import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


SCORES_TO_PLOT = [
    "msp",
    "mls",
    "mahalanobis"
]


DISPLAY_NAMES = {
    "msp": "MSP Unknownness",
    "mls": "MLS Unknownness",
    "mahalanobis": "Mahalanobis Distance"
}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="vanilla"
    )

    arguments = parser.parse_args()

    results_directory = (
        Path("task4/results")
        / arguments.model_name
    )

    score_data = torch.load(
        results_directory / "posthoc_scores.pt",
        map_location="cpu",
        weights_only=False
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5)
    )

    for axis, score_name in zip(
        axes,
        SCORES_TO_PLOT
    ):
        known_scores = to_numpy(
            score_data[score_name]["test"]
        )

        near_scores = to_numpy(
            score_data[score_name][
                "near_unknown"
            ]
        )

        far_scores = to_numpy(
            score_data[score_name][
                "far_unknown"
            ]
        )

        threshold = score_data[
            score_name
        ]["threshold"]

        axis.hist(
            known_scores,
            bins=50,
            density=True,
            alpha=0.50,
            label="CIFAR-10 known"
        )

        axis.hist(
            near_scores,
            bins=50,
            density=True,
            alpha=0.50,
            label="Near unknown"
        )

        axis.hist(
            far_scores,
            bins=50,
            density=True,
            alpha=0.50,
            label="Far unknown"
        )

        axis.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="95th-percentile threshold"
        )

        axis.set_title(
            DISPLAY_NAMES[score_name]
        )

        axis.set_xlabel(
            "Unknownness score"
        )

        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)

    axes[0].legend(
        fontsize=8
    )

    figure.suptitle(
        "Vanilla Model: Known and Unknown Score Distributions"
    )

    figure.tight_layout()

    png_path = (
        results_directory
        / "score_distributions.png"
    )

    pdf_path = (
        results_directory
        / "score_distributions.pdf"
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