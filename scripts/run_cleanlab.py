from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.cleanlab_detector import (
    detect_label_issues,
    load_noise_mask,
)
from src.cross_validation import (
    CrossValidationConfig,
    generate_out_of_sample_predictions,
)


VALID_NOISE_LEVELS = (
    "clean",
    "noise_10",
    "noise_20",
    "noise_30",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate out-of-sample predictions and "
            "detect label issues with Cleanlab."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/cleanlab.yaml"
        ),
        help="Path to the Cleanlab YAML configuration.",
    )

    parser.add_argument(
        "--noise-level",
        type=str,
        choices=VALID_NOISE_LEVELS,
        default=None,
        help=(
            "Override the noise level specified "
            "in the YAML file."
        ),
    )

    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help=(
            "Reuse saved out-of-sample probabilities "
            "instead of training the cross-validation models."
        ),
    )

    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help=(
            "Only save Cleanlab result files and skip "
            "Matplotlib/CIFAR-10 visualizations."
        ),
    )

    return parser.parse_args()


def load_yaml_config(
    config_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain "
            "a dictionary."
        )

    return config


def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def main() -> None:
    arguments = parse_arguments()

    configuration = load_yaml_config(
        arguments.config
    )

    experiment_config = configuration[
        "experiment"
    ]

    data_config = configuration["data"]

    cv_config = configuration[
        "cross_validation"
    ]

    cleanlab_config = configuration[
        "cleanlab"
    ]

    output_config = configuration["output"]

    noise_level = (
        arguments.noise_level
        if arguments.noise_level is not None
        else experiment_config["noise_level"]
    )

    seed = int(
        experiment_config["seed"]
    )

    data_dir = Path(
        data_config["data_dir"]
    )

    noise_root = Path(
        data_config["noise_root"]
    )

    predictions_directory = (
        Path(output_config["predictions_dir"])
        / noise_level
    )

    cleanlab_directory = (
        Path(output_config["cleanlab_dir"])
        / noise_level
    )

    figures_directory = (
        Path(output_config["figures_dir"])
        / noise_level
    )

    predictions_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleanlab_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_probs_path = (
        predictions_directory
        / "out_of_sample_pred_probs.npy"
    )

    labels_path = (
        predictions_directory
        / "labels.npy"
    )

    clean_labels_path = (
        predictions_directory
        / "clean_labels.npy"
    )

    fold_history_path = (
        predictions_directory
        / "cross_validation_history.csv"
    )

    if arguments.reuse_predictions:
        if not pred_probs_path.exists():
            raise FileNotFoundError(
                f"Predictions not found: "
                f"{pred_probs_path}"
            )

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels not found: {labels_path}"
            )

        print(
            "Reusing previously saved "
            "out-of-sample predictions."
        )

        pred_probs = np.load(
            pred_probs_path
        )

        labels = np.load(
            labels_path
        )

    else:
        import torch

        cross_validation_config = (
            CrossValidationConfig(
                data_dir=data_dir,
                noise_root=noise_root,
                noise_level=noise_level,
                num_classes=int(
                    data_config["num_classes"]
                ),
                num_folds=int(
                    cv_config["num_folds"]
                ),
                epochs_per_fold=int(
                    cv_config["epochs_per_fold"]
                ),
                batch_size=int(
                    cv_config["batch_size"]
                ),
                num_workers=int(
                    cv_config["num_workers"]
                ),
                learning_rate=float(
                    cv_config["learning_rate"]
                ),
                weight_decay=float(
                    cv_config["weight_decay"]
                ),
                seed=seed,
            )
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        (
            pred_probs,
            labels,
            fold_histories,
        ) = generate_out_of_sample_predictions(
            config=cross_validation_config,
            device=device,
        )

        np.save(
            pred_probs_path,
            pred_probs,
        )

        np.save(
            labels_path,
            labels,
        )

        pd.DataFrame(
            fold_histories
        ).to_csv(
            fold_history_path,
            index=False,
        )

        print(
            f"\nPredicted probabilities saved to: "
            f"{pred_probs_path}"
        )

    source_clean_labels_path = (
        noise_root
        / noise_level
        / "clean_labels.npy"
    )

    if source_clean_labels_path.exists():
        clean_labels = np.load(
            source_clean_labels_path
        )

        if len(clean_labels) != len(labels):
            raise ValueError(
                "clean_labels.npy and labels.npy must "
                "contain the same number of samples."
            )

        np.save(
            clean_labels_path,
            clean_labels,
        )

    true_noise_mask = load_noise_mask(
        noise_root=noise_root,
        noise_level=noise_level,
        number_of_samples=len(labels),
    )

    detection_result = detect_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        true_noise_mask=true_noise_mask,
        filter_by=cleanlab_config[
            "filter_by"
        ],
        rank_by=cleanlab_config[
            "rank_by"
        ],
    )

    np.save(
        cleanlab_directory
        / "issue_indices.npy",
        detection_result.issue_indices,
    )

    np.save(
        cleanlab_directory
        / "issue_mask.npy",
        detection_result.issue_mask,
    )

    np.save(
        cleanlab_directory
        / "label_quality_scores.npy",
        detection_result.quality_scores,
    )

    detection_result.results_table.to_csv(
        cleanlab_directory
        / "label_issues.csv",
        index=False,
    )

    detected_issues_table = (
        detection_result.results_table[
            detection_result.results_table[
                "is_label_issue"
            ]
        ]
        .sort_values(
            by="issue_rank",
            ascending=True,
        )
    )

    detected_issues_table.to_csv(
        cleanlab_directory
        / "detected_label_issues_only.csv",
        index=False,
    )

    save_json(
        data=detection_result.metrics,
        output_path=(
            cleanlab_directory
            / "detection_metrics.json"
        ),
    )

    if arguments.skip_figures:
        print("\nFigure generation skipped.")

    else:
        from src.cleanlab_visualization import (
            plot_confidence_comparison,
            plot_detection_metrics,
            plot_quality_score_histogram,
            plot_top_label_issues,
        )

        plot_quality_score_histogram(
            quality_scores=(
                detection_result.quality_scores
            ),
            issue_mask=(
                detection_result.issue_mask
            ),
            output_path=(
                figures_directory
                / "quality_score_histogram.png"
            ),
        )

        plot_confidence_comparison(
            results_table=(
                detection_result.results_table
            ),
            output_path=(
                figures_directory
                / "self_confidence_comparison.png"
            ),
        )

        plot_detection_metrics(
            metrics=detection_result.metrics,
            output_path=(
                figures_directory
                / "detection_metrics.png"
            ),
        )

        plot_top_label_issues(
            data_dir=data_dir,
            results_table=(
                detection_result.results_table
            ),
            output_path=(
                figures_directory
                / "top_label_issues.png"
            ),
            maximum_samples=16,
        )

    metrics = detection_result.metrics

    print("\nCleanlab analysis complete.")
    print(f"Noise level: {noise_level}")
    print(
        "Detected label issues: "
        f"{metrics['detected_issue_count']}"
    )

    if "precision" in metrics:
        print(
            f"Precision: {metrics['precision']:.4f}"
        )
        print(
            f"Recall: {metrics['recall']:.4f}"
        )
        print(
            f"F1: {metrics['f1']:.4f}"
        )

    print(f"Results: {cleanlab_directory}")
    if not arguments.skip_figures:
        print(f"Figures: {figures_directory}")

if __name__ == "__main__":
        main()
