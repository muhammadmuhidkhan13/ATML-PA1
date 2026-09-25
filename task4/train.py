import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from task4.data.cifar10 import get_cifar10_loaders
from task4.methods.vanilla import evaluate
from task4.methods.vanilla import train_one_epoch
from task4.models.resnet_cifar import CIFARResNet18


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_output_paths(config, run_name):
    if run_name is None:
        return {
            "checkpoint": Path(
                config["output"]["checkpoint"]
            ),
            "history": Path(
                config["output"]["history"]
            ),
            "summary": Path(
                f"task4/results/{config['method']}_summary.json"
            )
        }

    run_directory = Path("task4/results") / run_name

    return {
        "checkpoint": run_directory / "best_checkpoint.pt",
        "history": run_directory / "training_history.csv",
        "summary": run_directory / "summary.json"
    }


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)


def train(config, maximum_epochs=None, run_name=None):
    method = config["method"]
    seed = config["seed"]

    if method not in {"vanilla", "gcsc"}:
        raise ValueError(
            "This training stage supports vanilla and gcsc."
        )

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_randaugment = config["data"]["randaugment"]

    loaders = get_cifar10_loaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_randaugment=use_randaugment,
        seed=seed
    )

    model = CIFARResNet18(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"]
    )

    configured_epochs = config["training"]["epochs"]

    if maximum_epochs is None:
        epochs = configured_epochs
    else:
        epochs = maximum_epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    output_paths = get_output_paths(
        config,
        run_name
    )

    output_paths["checkpoint"].parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history = []
    best_val_accuracy = -1.0
    best_epoch = None

    print(f"Method: {method}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"RandAugment: {use_randaugment}")

    for epoch in range(1, epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device
        )

        val_metrics = evaluate(
            model,
            loaders["val"],
            device
        )

        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"]
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train accuracy: {train_metrics['accuracy']:.4f} | "
            f"validation accuracy: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch

            torch.save({
                "method": method,
                "epoch": epoch,
                "validation_accuracy": best_val_accuracy,
                "model_state_dict": model.state_dict(),
                "config": config
            }, output_paths["checkpoint"])

            print("  Saved new best checkpoint.")

        scheduler.step()
        save_history(
            history,
            output_paths["history"]
        )

    checkpoint = torch.load(
        output_paths["checkpoint"],
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = evaluate(
        model,
        loaders["test"],
        device
    )

    summary = {
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "cifar10_test_accuracy": test_metrics["accuracy"],
        "cifar10_test_loss": test_metrics["loss"],
        "checkpoint": str(output_paths["checkpoint"])
    }

    output_paths["summary"].parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_paths["summary"], "w") as file:
        json.dump(summary, file, indent=2)

    print(f"\nBest epoch: {best_epoch}")
    print(
        "Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )
    print(
        "CIFAR-10 test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        "Checkpoint: "
        f"{output_paths['checkpoint']}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--maximum-epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)

    train(
        config=config,
        maximum_epochs=arguments.maximum_epochs,
        run_name=arguments.run_name
    )


if __name__ == "__main__":
    main()