from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from PIL import Image, ImageChops


def trim_whitespace(image):
    image = image.convert("RGB")

    background = Image.new(
        "RGB",
        image.size,
        (255, 255, 255)
    )

    difference = ImageChops.difference(
        image,
        background
    )

    bounding_box = difference.getbbox()

    if bounding_box is None:
        return image

    padding = 10

    left = max(
        0,
        bounding_box[0] - padding
    )

    upper = max(
        0,
        bounding_box[1] - padding
    )

    right = min(
        image.width,
        bounding_box[2] + padding
    )

    lower = min(
        image.height,
        bounding_box[3] + padding
    )

    return image.crop(
        (
            left,
            upper,
            right,
            lower
        )
    )


def load_image(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Plot not found: {file_path}"
        )

    with Image.open(file_path) as image:
        return trim_whitespace(
            image.copy()
        )


def show_panel(
    axis,
    image,
    title
):
    axis.imshow(image)
    axis.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=10,
        loc="left"
    )
    axis.axis("off")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create the compact Task 3 combined "
            "report figure."
        )
    )

    parser.add_argument(
        "--plots-folder",
        default="task3/results/plots"
    )

    parser.add_argument(
        "--output-name",
        default="task3_combined_results"
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    plots_folder = Path(
        arguments.plots_folder
    )

    controlled_study = load_image(
        plots_folder
        / "controlled_lambda_study.png"
    )

    sharpness_comparison = load_image(
        plots_folder
        / "sharpness_comparison.png"
    )

    per_class_accuracy = load_image(
        plots_folder
        / "per_class_accuracy.png"
    )

    figure = plt.figure(
        figsize=(17, 10),
        facecolor="white"
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=5,
        height_ratios=[1.0, 1.1],
        width_ratios=[1, 1, 1, 1, 1],
        left=0.02,
        right=0.99,
        top=0.97,
        bottom=0.03,
        hspace=0.16,
        wspace=0.08
    )

    controlled_axis = figure.add_subplot(
        grid[0, :]
    )

    sharpness_axis = figure.add_subplot(
        grid[1, 0:2]
    )

    per_class_axis = figure.add_subplot(
        grid[1, 2:5]
    )

    show_panel(
        controlled_axis,
        controlled_study,
        "(a) Controlled DAN-DG study"
    )

    show_panel(
        sharpness_axis,
        sharpness_comparison,
        "(b) Fixed-radius local sharpness"
    )

    show_panel(
        per_class_axis,
        per_class_accuracy,
        "(c) Per-class Sketch accuracy"
    )

    png_path = plots_folder / (
        f"{arguments.output_name}.png"
    )

    pdf_path = plots_folder / (
        f"{arguments.output_name}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white"
    )

    plt.close(figure)

    print(
        f"Combined PNG saved to:\n"
        f"{png_path.resolve()}"
    )

    print(
        f"Combined PDF saved to:\n"
        f"{pdf_path.resolve()}"
    )


if __name__ == "__main__":
    main()