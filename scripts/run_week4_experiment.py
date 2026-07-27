"""
Week 4: Label-noise comparison experiment.

This script compares model performance under:

    noise_10
    noise_20
    noise_30

It performs the following steps:

1. Reads the existing out-of-sample prediction probabilities.
2. Converts prediction probabilities into predicted classes.
3. Calculates:
   - Accuracy
   - Precision
   - Recall
   - F1-score
4. Saves the results as CSV.
5. Saves classification reports.
6. Saves confusion matrices.
7. Creates publication-style figures.

Expected prediction directory structure:

    outputs/
    └── predictions/
        ├── noise_10/
        │   ├── labels.npy
        │   ├── out_of_sample_pred_probs.npy
        │   └── cross_validation_history.csv
        ├── noise_20/
        │   ├── labels.npy
        │   ├── out_of_sample_pred_probs.npy
        │   └── cross_validation_history.csv
        └── noise_30/
            ├── labels.npy
            ├── out_of_sample_pred_probs.npy
            └── cross_validation_history.csv

Run from the project root directory:

    python -u -m scripts.run_week4_experiment --analysis-only

Automatically generate missing prediction files:

    python -u -m scripts.run_week4_experiment --run-cleanlab
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Project root:
# image-data-quality-pipeline/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "week4_experiment.yaml"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare classification performance across "
            "multiple label-noise levels."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the Week 4 YAML configuration file.",
    )

    run_mode_group = parser.add_mutually_exclusive_group()

    run_mode_group.add_argument(
        "--analysis-only",
        action="store_true",
        help=(
            "Only analyse existing prediction files. "
            "Do not run model training."
        ),
    )

    run_mode_group.add_argument(
        "--run-cleanlab",
        action="store_true",
        help=(
            "Run scripts.run_cleanlab for missing "
            "prediction files before analysis."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run every Cleanlab experiment even when "
            "prediction files already exist."
        ),
    )

    return parser.parse_args()


def load_yaml_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load and validate the YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found:\n"
            f"{config_path}"
        )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a dictionary."
        )

    required_sections = [
        "experiment",
        "paths",
        "files",
        "metrics",
        "plots",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]

    if missing_sections:
        raise KeyError(
            "The configuration file is missing sections: "
            + ", ".join(missing_sections)
        )

    return config


def create_output_directories(
    output_directory: Path,
) -> dict[str, Path]:
    """Create all Week 4 output directories."""

    figures_directory = (
        output_directory
        / "figures"
    )

    reports_directory = (
        output_directory
        / "classification_reports"
    )

    confusion_matrices_directory = (
        output_directory
        / "confusion_matrices"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_matrices_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "output": output_directory,
        "figures": figures_directory,
        "reports": reports_directory,
        "confusion_matrices": confusion_matrices_directory,
    }


def extract_noise_percentage(
    noise_level: str,
) -> int:
    """
    Convert a string such as noise_20 into integer 20.
    """

    try:
        percentage_text = noise_level.split("_")[-1]
        return int(percentage_text)

    except (ValueError, IndexError) as error:
        raise ValueError(
            f"Invalid noise-level name: {noise_level}. "
            "Expected a name such as noise_20."
        ) from error


def get_prediction_paths(
    predictions_directory: Path,
    noise_level: str,
    labels_filename: str,
    clean_labels_filename: str,
    prediction_probabilities_filename: str,
) -> dict[str, Path]:
    """Build all prediction paths for one noise level."""

    noise_directory = (
        predictions_directory
        / noise_level
    )

    return {
        "directory": noise_directory,
        "labels": (
            noise_directory
            / labels_filename
        ),
        "clean_labels": (
            noise_directory
            / clean_labels_filename
        ),
        "probabilities": (
            noise_directory
            / prediction_probabilities_filename
        ),
    }


def prediction_files_exist(
    prediction_paths: dict[str, Path],
) -> bool:
    """Check whether the required prediction files exist."""

    labels_path = prediction_paths["labels"]
    probabilities_path = prediction_paths["probabilities"]

    return (
        labels_path.exists()
        and probabilities_path.exists()
        and labels_path.is_file()
        and probabilities_path.is_file()
        and labels_path.stat().st_size > 0
        and probabilities_path.stat().st_size > 0
    )


def print_prediction_file_status(
    noise_level: str,
    prediction_paths: dict[str, Path],
) -> None:
    """Print the expected paths and whether they exist."""

    print(f"\nChecking prediction files for {noise_level}")
    print("-" * 60)

    print(
        f"Directory:\n"
        f"  {prediction_paths['directory']}"
    )

    print(
        f"Labels:\n"
        f"  {prediction_paths['labels']}\n"
        f"  Exists: {prediction_paths['labels'].exists()}"
    )

    print(
        f"Prediction probabilities:\n"
        f"  {prediction_paths['probabilities']}\n"
        f"  Exists: "
        f"{prediction_paths['probabilities'].exists()}"
    )

    print(
        f"Clean labels:\n"
        f"  {prediction_paths['clean_labels']}\n"
        f"  Exists: "
        f"{prediction_paths['clean_labels'].exists()}"
    )


def run_cleanlab_pipeline(
    noise_level: str,
) -> None:
    """
    Run the existing Cleanlab pipeline for one noise level.

    A subprocess is used so that its CPU, RAM, and CUDA resources
    are released when the process finishes.
    """

    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.run_cleanlab",
        "--noise-level",
        noise_level,
    ]

    print("\n" + "=" * 70)
    print(f"Running Cleanlab pipeline: {noise_level}")
    print("=" * 70)

    print(
        "Command:\n"
        + " ".join(command)
    )

    completed_process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            f"Cleanlab pipeline failed for {noise_level}.\n"
            f"Return code: {completed_process.returncode}"
        )

    print(
        f"\nCleanlab pipeline completed: {noise_level}"
    )

    gc.collect()


def load_numpy_array(
    file_path: Path,
) -> np.ndarray:
    """Load a NumPy file and verify that it is valid."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required NumPy file was not found:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"The path is not a file:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"The NumPy file is empty:\n"
            f"{file_path}"
        )

    try:
        array = np.load(
            file_path,
            allow_pickle=False,
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to read NumPy file:\n"
            f"{file_path}"
        ) from error

    return np.asarray(array)


def select_evaluation_labels(
    prediction_paths: dict[str, Path],
) -> tuple[np.ndarray, str]:
    """
    Select labels for performance evaluation.

    Priority:

    1. clean_labels.npy
    2. labels.npy

    Clean labels are preferred because they represent the original
    ground-truth classes. If clean_labels.npy does not exist, the
    program falls back to labels.npy.
    """

    clean_labels_path = prediction_paths["clean_labels"]
    labels_path = prediction_paths["labels"]

    if clean_labels_path.exists():
        labels = load_numpy_array(
            clean_labels_path
        )

        return labels, "clean_labels.npy"

    labels = load_numpy_array(
        labels_path
    )

    return labels, "labels.npy"


def validate_prediction_data(
    labels: np.ndarray,
    prediction_probabilities: np.ndarray,
    noise_level: str,
) -> None:
    """Validate label and prediction-probability arrays."""

    if labels.ndim != 1:
        raise ValueError(
            f"{noise_level}: labels must be one-dimensional.\n"
            f"Received shape: {labels.shape}"
        )

    if prediction_probabilities.ndim != 2:
        raise ValueError(
            f"{noise_level}: prediction probabilities must "
            "be a two-dimensional array.\n"
            "Expected shape: "
            "[number_of_samples, number_of_classes]\n"
            f"Received shape: "
            f"{prediction_probabilities.shape}"
        )

    if len(labels) != len(prediction_probabilities):
        raise ValueError(
            f"{noise_level}: labels and predictions contain "
            "different sample counts.\n"
            f"Labels: {len(labels)}\n"
            f"Predictions: {len(prediction_probabilities)}"
        )

    if not np.isfinite(
        prediction_probabilities
    ).all():
        raise ValueError(
            f"{noise_level}: prediction probabilities contain "
            "NaN or infinite values."
        )

    probability_sums = (
        prediction_probabilities.sum(axis=1)
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-3,
    ):
        print(
            "Warning: some prediction rows do not sum "
            "exactly to 1.0."
        )


def calculate_classification_metrics(
    labels: np.ndarray,
    prediction_probabilities: np.ndarray,
    average: str,
    zero_division: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Calculate the main classification metrics."""

    labels = (
        labels
        .reshape(-1)
        .astype(int)
    )

    predicted_labels = np.argmax(
        prediction_probabilities,
        axis=1,
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predicted_labels,
    )

    precision = precision_score(
        labels,
        predicted_labels,
        average=average,
        zero_division=zero_division,
    )

    recall = recall_score(
        labels,
        predicted_labels,
        average=average,
        zero_division=zero_division,
    )

    f1 = f1_score(
        labels,
        predicted_labels,
        average=average,
        zero_division=zero_division,
    )

    metrics: dict[str, float | int] = {
        "number_of_samples": int(len(labels)),
        "number_of_classes": int(
            prediction_probabilities.shape[1]
        ),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    return metrics, predicted_labels


def save_classification_report(
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    output_path: Path,
    zero_division: int,
) -> None:
    """Save a per-class classification report."""

    report = classification_report(
        labels,
        predicted_labels,
        digits=4,
        zero_division=zero_division,
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )


def save_confusion_matrix(
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion matrix as a CSV file."""

    matrix = confusion_matrix(
        labels,
        predicted_labels,
    )

    class_names = [
        f"class_{class_index}"
        for class_index in range(matrix.shape[0])
    ]

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_dataframe.index.name = "actual"
    matrix_dataframe.columns.name = "predicted"

    matrix_dataframe.to_csv(
        output_path,
        encoding="utf-8",
    )


def analyse_noise_level(
    noise_level: str,
    predictions_directory: Path,
    labels_filename: str,
    clean_labels_filename: str,
    prediction_probabilities_filename: str,
    average: str,
    zero_division: int,
    output_directories: dict[str, Path],
) -> dict[str, Any]:
    """Analyse one noise-level experiment."""

    prediction_paths = get_prediction_paths(
        predictions_directory=predictions_directory,
        noise_level=noise_level,
        labels_filename=labels_filename,
        clean_labels_filename=clean_labels_filename,
        prediction_probabilities_filename=(
            prediction_probabilities_filename
        ),
    )

    print_prediction_file_status(
        noise_level=noise_level,
        prediction_paths=prediction_paths,
    )

    if not prediction_files_exist(
        prediction_paths
    ):
        raise FileNotFoundError(
            f"\nPrediction files are missing for {noise_level}.\n\n"
            f"Expected directory:\n"
            f"{prediction_paths['directory']}\n\n"
            "Required files:\n"
            f"1. {labels_filename}\n"
            f"2. {prediction_probabilities_filename}\n\n"
            "Generate the files first with:\n"
            f"python -u -m scripts.run_cleanlab "
            f"--noise-level {noise_level}"
        )

    labels, label_source = select_evaluation_labels(
        prediction_paths
    )

    labels = (
        np.asarray(labels)
        .reshape(-1)
    )

    prediction_probabilities = load_numpy_array(
        prediction_paths["probabilities"]
    )

    validate_prediction_data(
        labels=labels,
        prediction_probabilities=prediction_probabilities,
        noise_level=noise_level,
    )

    metrics, predicted_labels = (
        calculate_classification_metrics(
            labels=labels,
            prediction_probabilities=(
                prediction_probabilities
            ),
            average=average,
            zero_division=zero_division,
        )
    )

    noise_percentage = extract_noise_percentage(
        noise_level
    )

    classification_report_path = (
        output_directories["reports"]
        / f"{noise_level}_classification_report.txt"
    )

    confusion_matrix_path = (
        output_directories["confusion_matrices"]
        / f"{noise_level}_confusion_matrix.csv"
    )

    save_classification_report(
        labels=labels,
        predicted_labels=predicted_labels,
        output_path=classification_report_path,
        zero_division=zero_division,
    )

    save_confusion_matrix(
        labels=labels,
        predicted_labels=predicted_labels,
        output_path=confusion_matrix_path,
    )

    result: dict[str, Any] = {
        "noise_level": noise_level,
        "noise_percentage": noise_percentage,
        "evaluation_label_source": label_source,
        **metrics,
    }

    print("\nResult")
    print("-" * 60)
    print(f"Noise level: {noise_percentage}%")
    print(f"Evaluation labels: {label_source}")
    print(
        f"Number of samples: "
        f"{metrics['number_of_samples']}"
    )
    print(
        f"Number of classes: "
        f"{metrics['number_of_classes']}"
    )
    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"F1-score: "
        f"{metrics['f1']:.4f}"
    )

    del labels
    del predicted_labels
    del prediction_probabilities

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        pass

    return result


def configure_plot_style() -> None:
    """Configure a clean publication-style plot layout."""

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_all_metrics(
    results_dataframe: pd.DataFrame,
    output_path: Path,
    figure_width: float,
    figure_height: float,
    dpi: int,
) -> None:
    """Plot Accuracy, Precision, Recall, and F1 in one figure."""

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
    ]

    for column_name, display_name in metrics:
        axis.plot(
            results_dataframe["noise_percentage"],
            results_dataframe[column_name],
            marker="o",
            linewidth=2,
            markersize=7,
            label=display_name,
        )

    axis.set_title(
        "Model Performance under Different Label-Noise Levels"
    )

    axis.set_xlabel(
        "Label noise (%)"
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_xticks(
        results_dataframe["noise_percentage"]
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend(
        frameon=False,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_individual_metric(
    results_dataframe: pd.DataFrame,
    metric_column: str,
    metric_name: str,
    output_path: Path,
    figure_width: float,
    figure_height: float,
    dpi: int,
) -> None:
    """Create one publication-style line graph for one metric."""

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    x_values = results_dataframe[
        "noise_percentage"
    ]

    y_values = results_dataframe[
        metric_column
    ]

    axis.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2,
        markersize=7,
    )

    for x_value, y_value in zip(
        x_values,
        y_values,
    ):
        axis.annotate(
            f"{y_value:.3f}",
            xy=(
                x_value,
                y_value,
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    axis.set_title(
        f"{metric_name} under Different Label-Noise Levels"
    )

    axis.set_xlabel(
        "Label noise (%)"
    )

    axis.set_ylabel(
        metric_name
    )

    axis.set_xticks(
        x_values
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_grouped_bar_chart(
    results_dataframe: pd.DataFrame,
    output_path: Path,
    figure_width: float,
    figure_height: float,
    dpi: int,
) -> None:
    """Create a grouped bar chart for all classification metrics."""

    noise_percentages = (
        results_dataframe["noise_percentage"]
        .to_numpy()
    )

    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
    ]

    x_positions = np.arange(
        len(noise_percentages)
    )

    bar_width = 0.18
    number_of_metrics = len(metrics)

    figure, axis = plt.subplots(
        figsize=(
            figure_width + 1,
            figure_height,
        )
    )

    for metric_index, (
        column_name,
        display_name,
    ) in enumerate(metrics):

        offset = (
            metric_index
            - (number_of_metrics - 1) / 2
        ) * bar_width

        axis.bar(
            x_positions + offset,
            results_dataframe[column_name],
            width=bar_width,
            label=display_name,
        )

    axis.set_title(
        "Classification Metrics at Different Noise Levels"
    )

    axis.set_xlabel(
        "Label noise (%)"
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_xticks(
        x_positions,
        labels=[
            f"{percentage}%"
            for percentage in noise_percentages
        ],
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_all_figures(
    results_dataframe: pd.DataFrame,
    figures_directory: Path,
    figure_width: float,
    figure_height: float,
    dpi: int,
) -> None:
    """Create all Week 4 experiment figures."""

    configure_plot_style()

    plot_all_metrics(
        results_dataframe=results_dataframe,
        output_path=(
            figures_directory
            / "all_metrics_vs_noise.png"
        ),
        figure_width=figure_width,
        figure_height=figure_height,
        dpi=dpi,
    )

    metric_settings = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
    ]

    for metric_column, metric_name in metric_settings:
        plot_individual_metric(
            results_dataframe=results_dataframe,
            metric_column=metric_column,
            metric_name=metric_name,
            output_path=(
                figures_directory
                / f"{metric_column}_vs_noise.png"
            ),
            figure_width=figure_width,
            figure_height=figure_height,
            dpi=dpi,
        )

    plot_grouped_bar_chart(
        results_dataframe=results_dataframe,
        output_path=(
            figures_directory
            / "metrics_grouped_bar_chart.png"
        ),
        figure_width=figure_width,
        figure_height=figure_height,
        dpi=dpi,
    )

    plt.close("all")
    gc.collect()


def create_experiment_summary(
    results_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a text summary of the experiment."""

    lowest_noise_result = (
        results_dataframe.iloc[0]
    )

    highest_noise_result = (
        results_dataframe.iloc[-1]
    )

    accuracy_change = (
        highest_noise_result["accuracy"]
        - lowest_noise_result["accuracy"]
    )

    precision_change = (
        highest_noise_result["precision"]
        - lowest_noise_result["precision"]
    )

    recall_change = (
        highest_noise_result["recall"]
        - lowest_noise_result["recall"]
    )

    f1_change = (
        highest_noise_result["f1"]
        - lowest_noise_result["f1"]
    )

    summary_lines = [
        "Week 4: Label-Noise Experiment",
        "=" * 50,
        "",
        (
            "Research question: How does increasing label "
            "noise affect image-classification performance?"
        ),
        "",
        (
            "Noise levels: "
            + ", ".join(
                f"{int(value)}%"
                for value in results_dataframe[
                    "noise_percentage"
                ]
            )
        ),
        "",
        "Changes from the lowest to highest noise level:",
        f"Accuracy change:  {accuracy_change:+.4f}",
        f"Precision change: {precision_change:+.4f}",
        f"Recall change:    {recall_change:+.4f}",
        f"F1-score change:  {f1_change:+.4f}",
        "",
        "Full result table:",
        results_dataframe.to_string(
            index=False
        ),
        "",
        "Interpretation:",
        (
            "A negative change indicates that classification "
            "performance decreased when label noise increased."
        ),
    ]

    output_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete Week 4 experiment."""

    arguments = parse_arguments()

    config_path = Path(
        arguments.config
    ).resolve()

    config = load_yaml_config(
        config_path
    )

    experiment_config = config["experiment"]
    paths_config = config["paths"]
    files_config = config["files"]
    metrics_config = config["metrics"]
    plots_config = config["plots"]

    noise_levels = list(
        experiment_config["noise_levels"]
    )

    predictions_directory = (
        PROJECT_ROOT
        / paths_config["predictions_directory"]
    ).resolve()

    output_directory = (
        PROJECT_ROOT
        / paths_config["output_directory"]
    ).resolve()

    output_directories = create_output_directories(
        output_directory
    )

    labels_filename = (
        files_config["labels_filename"]
    )

    clean_labels_filename = (
        files_config["clean_labels_filename"]
    )

    prediction_probabilities_filename = (
        files_config[
            "prediction_probabilities_filename"
        ]
    )

    average = str(
        metrics_config.get(
            "average",
            "macro",
        )
    )

    zero_division = int(
        metrics_config.get(
            "zero_division",
            0,
        )
    )

    run_cleanlab_from_config = bool(
        experiment_config.get(
            "run_cleanlab",
            False,
        )
    )

    skip_existing_predictions = bool(
        experiment_config.get(
            "skip_existing_predictions",
            True,
        )
    )

    if arguments.analysis_only:
        should_run_cleanlab = False

    elif arguments.run_cleanlab:
        should_run_cleanlab = True

    else:
        should_run_cleanlab = run_cleanlab_from_config

    if arguments.force:
        skip_existing_predictions = False

    print("=" * 70)
    print("Week 4: Label-Noise Comparison Experiment")
    print("=" * 70)

    print(f"Project root:\n  {PROJECT_ROOT}")

    print(
        f"\nPredictions directory:\n"
        f"  {predictions_directory}"
    )

    print(
        f"\nOutput directory:\n"
        f"  {output_directory}"
    )

    print(
        f"\nNoise levels:\n"
        f"  {', '.join(noise_levels)}"
    )

    print(
        f"\nRun Cleanlab automatically:\n"
        f"  {should_run_cleanlab}"
    )

    print(
        f"\nSkip existing predictions:\n"
        f"  {skip_existing_predictions}"
    )

    experiment_results: list[
        dict[str, Any]
    ] = []

    for experiment_index, noise_level in enumerate(
        noise_levels,
        start=1,
    ):
        print("\n" + "#" * 70)

        print(
            f"Experiment {experiment_index}/"
            f"{len(noise_levels)}: {noise_level}"
        )

        print("#" * 70)

        prediction_paths = get_prediction_paths(
            predictions_directory=predictions_directory,
            noise_level=noise_level,
            labels_filename=labels_filename,
            clean_labels_filename=clean_labels_filename,
            prediction_probabilities_filename=(
                prediction_probabilities_filename
            ),
        )

        files_exist = prediction_files_exist(
            prediction_paths
        )

        if should_run_cleanlab:
            if (
                skip_existing_predictions
                and files_exist
            ):
                print(
                    f"\nPrediction files already exist for "
                    f"{noise_level}. Training will be skipped."
                )

            else:
                run_cleanlab_pipeline(
                    noise_level=noise_level
                )

        result = analyse_noise_level(
            noise_level=noise_level,
            predictions_directory=predictions_directory,
            labels_filename=labels_filename,
            clean_labels_filename=clean_labels_filename,
            prediction_probabilities_filename=(
                prediction_probabilities_filename
            ),
            average=average,
            zero_division=zero_division,
            output_directories=output_directories,
        )

        experiment_results.append(
            result
        )

    results_dataframe = pd.DataFrame(
        experiment_results
    )

    results_dataframe = (
        results_dataframe
        .sort_values(
            by="noise_percentage"
        )
        .reset_index(
            drop=True
        )
    )

    results_csv_path = (
        output_directory
        / "noise_experiment_results.csv"
    )

    results_dataframe.to_csv(
        results_csv_path,
        index=False,
        encoding="utf-8",
    )

    create_all_figures(
        results_dataframe=results_dataframe,
        figures_directory=output_directories["figures"],
        figure_width=float(
            plots_config.get(
                "figure_width",
                8,
            )
        ),
        figure_height=float(
            plots_config.get(
                "figure_height",
                5,
            )
        ),
        dpi=int(
            plots_config.get(
                "dpi",
                300,
            )
        ),
    )

    summary_path = (
        output_directory
        / "experiment_summary.txt"
    )

    create_experiment_summary(
        results_dataframe=results_dataframe,
        output_path=summary_path,
    )

    print("\n" + "=" * 70)
    print("Week 4 experiment completed successfully.")
    print("=" * 70)

    print("\nExperiment results:\n")

    print(
        results_dataframe.to_string(
            index=False
        )
    )

    print(
        f"\nResults CSV:\n"
        f"  {results_csv_path}"
    )

    print(
        f"\nExperiment summary:\n"
        f"  {summary_path}"
    )

    print(
        f"\nFigures directory:\n"
        f"  {output_directories['figures']}"
    )

    print(
        f"\nClassification reports:\n"
        f"  {output_directories['reports']}"
    )

    print(
        f"\nConfusion matrices:\n"
        f"  {output_directories['confusion_matrices']}"
    )


if __name__ == "__main__":
    main()