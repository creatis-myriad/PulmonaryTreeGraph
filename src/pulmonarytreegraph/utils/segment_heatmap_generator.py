"""
Module for creating segment heatmap visualizations from batch log files.
"""

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_segment_summary_from_log(log_file_path: str) -> dict[str, dict[str, Any]]:
    """
    Extract segment chain summary information from a single log file.

    Args:
        log_file_path: Path to the log file

    Returns:
        Dictionary with patient IDs as keys and segment data as values
    """
    results = {}

    try:
        with open(log_file_path) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
        return results

    # Find all SEGMENT CHAIN SUMMARY blocks
    pattern = r"SEGMENT CHAIN SUMMARY:\s*\n.*?------------------------------\s*\n.*?Total segment chains in tree: (\d+)\s*\n.*?Level 4 edges \(individual\): (\d+)\s*\n(.*?)=================================================="
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        total_chains = int(match.group(1))
        level4_edges = int(match.group(2))
        segment_data = match.group(3)
        start_pos = match.start()

        # Look backwards to find the most recent patient reference
        content_before = content[:start_pos]
        patient_patterns = [
            r"PROCESSING PATIENT: (\d+)",
            r"Patient: (\d+)",
        ]

        found_patient = None
        for pattern in patient_patterns:
            patient_matches = list(re.finditer(pattern, content_before, re.IGNORECASE))
            if patient_matches:
                found_patient = patient_matches[-1].group(1)
                break

        if found_patient:
            # Extract segment information
            segments = {}
            segment_matches = re.findall(
                r"([A-Z]{3}|unknown): (\d+) segment chains", segment_data
            )

            for segment, chain_count in segment_matches:
                segments[segment] = int(chain_count)

            if segments:
                results[found_patient] = {
                    "total_chains": total_chains,
                    "level4_edges": level4_edges,
                    "segments": segments,
                }

    return results


def create_segment_heatmap_from_batch(
    batch_log_path: str,
    batch_name: str = None,
    save_path: str = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Create a heatmap visualization of segment chains from a single batch log file.

    Args:
        batch_log_path: Path to the batch log file
        batch_name: Name to display for the batch (defaults to filename)
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """

    # Extract segment data from the batch
    print(f"Extracting segment data from: {batch_log_path}")
    batch_data = extract_segment_summary_from_log(batch_log_path)

    if not batch_data:
        print("No segment data found in the log file")
        return None

    print(f"Found {len(batch_data)} patients with segment data")

    # Get all patients and segments
    all_patients = sorted(batch_data.keys())
    all_segments = set()

    for patient_data in batch_data.values():
        all_segments.update(patient_data["segments"].keys())

    # Order segments in anatomical order: RUL, RML, RLL, LUL, LLL
    segment_order = ["RUL", "RML", "RLL", "LUL", "LLL", "unknown"]
    found_segments = [seg for seg in segment_order if seg in all_segments]
    # Add any unexpected segments at the end
    found_segments.extend(
        sorted([seg for seg in all_segments if seg not in segment_order])
    )
    all_segments = found_segments

    print(f"Patients: {all_patients}")
    print(f"Segments: {all_segments}")

    # Create detailed comparison table
    detailed_data = []
    for patient in all_patients:
        for segment in all_segments:
            patient_segments = batch_data[patient]["segments"]
            segment_count = patient_segments.get(segment, 0)
            detailed_data.append(
                {"Patient": patient, "Segment": segment, "Chains": segment_count}
            )

    df = pd.DataFrame(detailed_data)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Pivot data for heatmap
    heatmap_data = df.pivot(index="Patient", columns="Segment", values="Chains")

    # Ensure column order matches our anatomical order
    heatmap_data = heatmap_data.reindex(columns=all_segments, fill_value=0)

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="g",
        cmap="Oranges",
        cbar_kws={"label": "Segment Chains"},
        ax=ax,
    )

    # Set title
    if batch_name is None:
        batch_name = Path(batch_log_path).stem

    ax.set_title(
        f"{batch_name}: Segment Chains by Patient and Lobe",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Lung Lobes", fontsize=12)
    ax.set_ylabel("Patient ID", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    return fig


def create_segment_heatmaps_from_multiple_batches(
    batch_log_paths: dict[str, str],
    save_directory: str = None,
    figsize: tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a 2x2 subplot of heatmaps from multiple batch log files.

    Args:
        batch_log_paths: Dictionary with batch names as keys and log file paths as values
        save_directory: Optional directory to save the figure
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """

    # Extract data from all batches
    all_batch_data = {}
    for batch_name, log_file in batch_log_paths.items():
        if Path(log_file).exists():
            all_batch_data[batch_name] = extract_segment_summary_from_log(log_file)
            print(f"✓ {batch_name}: {len(all_batch_data[batch_name])} patients")
        else:
            print(f"✗ {log_file} not found")

    if not all_batch_data:
        print("No valid batch data found")
        return None

    # Get all patients and segments across all batches
    all_patients = set()
    all_segments = set()

    for batch_data in all_batch_data.values():
        all_patients.update(batch_data.keys())
        for patient_data in batch_data.values():
            all_segments.update(patient_data["segments"].keys())

    all_patients = sorted(all_patients)

    # Order segments in anatomical order: RUL, RML, RLL, LUL, LLL
    segment_order = ["RUL", "RML", "RLL", "LUL", "LLL"]
    found_segments = [seg for seg in segment_order if seg in all_segments]
    # Add any unexpected segments at the end
    found_segments.extend(
        sorted([seg for seg in all_segments if seg not in segment_order])
    )
    all_segments = found_segments

    # Create subplot layout
    n_batches = len(batch_log_paths)
    if n_batches <= 2:
        rows, cols = 1, n_batches
    elif n_batches <= 4:
        rows, cols = 2, 2
    else:
        rows = int(n_batches**0.5) + 1
        cols = int(n_batches / rows) + 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single subplot case
    if n_batches == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Create heatmap for each batch
    for idx, (batch_name, _) in enumerate(batch_log_paths.items()):
        if batch_name not in all_batch_data:
            continue

        batch_data = all_batch_data[batch_name]

        # Create data for this batch
        detailed_data = []
        for patient in all_patients:
            for segment in all_segments:
                if patient in batch_data:
                    patient_segments = batch_data[patient]["segments"]
                    segment_count = patient_segments.get(segment, 0)
                else:
                    segment_count = 0

                detailed_data.append(
                    {"Patient": patient, "Segment": segment, "Chains": segment_count}
                )

        df = pd.DataFrame(detailed_data)
        heatmap_data = df.pivot(index="Patient", columns="Segment", values="Chains")

        # Ensure column order matches our anatomical order
        heatmap_data = heatmap_data.reindex(columns=all_segments, fill_value=0)

        # Create heatmap
        ax = axes[idx]
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="g",
            cmap="Oranges",
            cbar_kws={"label": "Segment Chains"},
            ax=ax,
        )

        ax.set_title(f"{batch_name}: Segment Chains by Patient", fontweight="bold")
        ax.set_xlabel("Lobes")
        ax.set_ylabel("Patient")

    # Hide unused subplots
    for idx in range(n_batches, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Segment Chain Heatmaps: Comparison Across Batches",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save if requested
    if save_directory:
        save_path = Path(save_directory) / "segment_heatmaps_comparison.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison heatmaps saved to: {save_path}")

    return fig


# Example usage
if __name__ == "__main__":
    # Example: Create heatmap from a single batch
    batch_path = "/home/desligneris/Documents/code/pulmonarytreegraph/experiments/batch_results_1/logs/batch_processing_1751301694.log"

    # Create single heatmap
    fig = create_segment_heatmap_from_batch(
        batch_log_path=batch_path,
        batch_name="Batch 1",
        save_path="/home/desligneris/Documents/code/pulmonarytreegraph/experiments/analysis/single_batch_heatmap.png",
    )

    if fig:
        plt.show()

    # Example: Create comparison heatmaps from multiple batches
    batch_paths = {
        "Batch_1": "/home/desligneris/Documents/code/pulmonarytreegraph/experiments/batch_results_1/logs/batch_processing_1751301694.log",
        "Batch_2": "/home/desligneris/Documents/code/pulmonarytreegraph/experiments/batch_results_2/logs/batch_processing_1751304427.log",
        "Batch_3": "/home/desligneris/Documents/code/pulmonarytreegraph/experiments/batch_results_3/logs/batch_processing_1751311498.log",
        "Batch_4": "/home/desligneris/Documents/code/pulmonarytreegraph/experiments/batch_results_4/logs/batch_processing_1751314408.log",
    }

    # Create comparison heatmaps
    comparison_fig = create_segment_heatmaps_from_multiple_batches(
        batch_log_paths=batch_paths,
        save_directory="/home/desligneris/Documents/code/pulmonarytreegraph/experiments/analysis",
    )

    if comparison_fig:
        plt.show()
