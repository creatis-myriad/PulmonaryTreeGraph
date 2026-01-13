"""
Module for creating comprehensive segment visualizations from CSV statistics files.
Enhanced with violin plots, summary statistics, and detailed analysis.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Expected segments for each lobe (anatomical reference)
EXPECTED_SEGMENTS = {
    "RUL": 3,
    "RML": 2,
    "RLL": 5,
    "LUL": 5,
    "LLL": 4,  # or 5
}


def load_segment_data_from_csv(csv_file_path: str) -> dict[str, dict[str, int]]:
    """
    Load segment data from global_labelling_stats.csv file.

    Args:
        csv_file_path: Path to the CSV statistics file

    Returns:
        Dictionary with patient IDs as keys and segment data as values
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV with {len(df)} patients from: {csv_file_path}")

        # Convert to the expected format
        results = {}

        for _, row in df.iterrows():
            patient_id = str(row["patient"])  # Ensure string format

            # Extract segment data
            segments = {
                "RUL": int(row["RUL"]),
                "RML": int(row["RML"]),
                "RLL": int(row["RLL"]),
                "LUL": int(row["LUL"]),
                "LLL": int(row["LLL"]),
            }

            # Add unknown and interlobar if they exist and are non-zero
            if "unknown" in row and row["unknown"] > 0:
                segments["unknown"] = int(row["unknown"])
            if "interlobar" in row and row["interlobar"] > 0:
                segments["interlobar"] = int(row["interlobar"])

            results[patient_id] = {
                "segments": segments,
                "total_chains": int(row["total_chains"]),
                "level4_edges": int(row["level4_edges"]),
            }

        print(f"Successfully loaded data for {len(results)} patients")
        return results

    except FileNotFoundError:
        print(f"CSV file not found: {csv_file_path}")
        return {}
    except Exception as e:
        print(f"Error loading CSV file {csv_file_path}: {e}")
        return {}


def create_comprehensive_analysis_from_csvs(
    csv_file_paths: dict[str, str],
    save_directory: str = None,
    include_totals: bool = True,
    figsize_heatmap: tuple[int, int] = (20, 12),
    figsize_violin: tuple[int, int] = (15, 10),
) -> dict[str, Any]:
    """
    Create comprehensive analysis including heatmaps, violin plots, and summary statistics.

    Args:
        csv_file_paths: Dictionary with batch names as keys and CSV file paths as values
        save_directory: Optional directory to save all outputs
        include_totals: Whether to include total chains in visualizations
        figsize_heatmap: Figure size for heatmap
        figsize_violin: Figure size for violin plots

    Returns:
        Dictionary containing all generated figures and statistics
    """

    if save_directory:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Load data from all CSV files
    print("Loading data from CSV files...")
    all_batch_data = {}
    for batch_name, csv_file in csv_file_paths.items():
        if Path(csv_file).exists():
            all_batch_data[batch_name] = load_segment_data_from_csv(csv_file)
            print(f"✓ {batch_name}: {len(all_batch_data[batch_name])} patients")
        else:
            print(f"✗ {csv_file} not found")

    if not all_batch_data:
        print("No valid batch data found")
        return results

    # 1. Create comparison heatmaps
    print("\n1. Creating comparison heatmaps...")
    heatmap_fig = create_comparison_heatmaps_from_csvs(
        csv_file_paths,
        save_directory=save_directory if save_directory else None,
        figsize=figsize_heatmap,
        include_totals=include_totals,
    )
    results["heatmap_figure"] = heatmap_fig

    # 2. Create summary statistics
    print("\n2. Calculating summary statistics...")
    summary_stats = create_summary_statistics_table(
        csv_file_paths, save_directory=save_directory if save_directory else None
    )
    results["summary_statistics"] = summary_stats

    # 3. Create violin plots for each lobe
    print("\n3. Creating violin plots...")
    violin_figs = {}

    for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
        violin_fig = create_violin_comparison_from_csvs(
            all_batch_data,
            lobe,
            save_path=save_dir / f"violin_comparison_{lobe}.png"
            if save_directory
            else None,
            figsize=figsize_violin,
        )
        violin_figs[lobe] = violin_fig

    results["violin_figures"] = violin_figs

    # 4. Create detailed configuration analysis
    print("\n4. Creating detailed configuration analysis...")
    config_analysis = create_detailed_configuration_analysis(
        all_batch_data, save_directory=save_directory if save_directory else None
    )
    results["configuration_analysis"] = config_analysis

    return results


def create_violin_comparison_from_csvs(
    all_batch_data: dict[str, dict[str, dict[str, Any]]],
    lobe: str,
    save_path: str = None,
    figsize: tuple[int, int] = (12, 8),
    annotate_outliers: bool = True,
) -> plt.Figure:
    """
    Create violin plots comparing all configurations for a specific lobe.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Prepare data for violin plots
    violin_data = []
    test_labels = []
    patient_ids_list = []

    for test_name, batch_data in all_batch_data.items():
        lobe_values = []
        patient_ids = []

        for patient_id, patient_data in batch_data.items():
            if lobe in patient_data["segments"]:
                lobe_values.append(patient_data["segments"][lobe])
                patient_ids.append(patient_id)

        if lobe_values:  # Only add if we have data
            violin_data.append(np.array(lobe_values))
            test_labels.append(test_name)
            patient_ids_list.append(patient_ids)

    if not violin_data:
        ax.text(
            0.5,
            0.5,
            f"No data available for {lobe}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        return fig

    # Create violin plot
    positions = list(range(len(violin_data)))
    parts = ax.violinplot(
        violin_data, positions=positions, widths=0.7, showmeans=True, showmedians=True
    )

    # Color the violin plots
    colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow", "lightpink"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    # Add expected line
    expected = EXPECTED_SEGMENTS.get(lobe, 0)
    if expected > 0:
        ax.axhline(
            y=expected,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Expected: {expected}",
        )

    # Annotate outliers if requested
    if annotate_outliers:
        for pos, data, pids in zip(
            positions, violin_data, patient_ids_list, strict=True
        ):
            if len(data) == 0:
                continue

            # Calculate outliers using IQR method
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (data < lower_bound) | (data > upper_bound)
            if outlier_mask.any():
                outlier_x = np.full(outlier_mask.sum(), pos)
                # Add small jitter for visibility
                outlier_x = outlier_x + np.random.uniform(
                    -0.1, 0.1, size=outlier_mask.sum()
                )
                outlier_y = data[outlier_mask]
                outlier_pids = np.array(pids)[outlier_mask]

                # Plot outlier points
                ax.scatter(outlier_x, outlier_y, c="red", s=50, marker="x", zorder=10)

                # Annotate with patient IDs
                for xx, yy, pid in zip(outlier_x, outlier_y, outlier_pids, strict=True):
                    ax.annotate(
                        pid,
                        (xx, yy),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        rotation=30,
                        ha="left",
                        color="darkred",
                        fontweight="bold",
                    )

    # Customize plot
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Number of Segments")
    ax.set_title(
        f"{lobe} - Distribution Comparison Across Configurations", fontweight="bold"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(test_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    stats_text = []
    for label, data in zip(test_labels, violin_data, strict=True):
        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)
        stats_text.append(
            f"{label}: μ={mean_val:.1f}±{std_val:.1f}, median={median_val:.1f}"
        )

    ax.text(
        0.02,
        0.98,
        "\n".join(stats_text),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Violin comparison for {lobe} saved to: {save_path}")

    return fig


def create_detailed_configuration_analysis(
    all_batch_data: dict[str, dict[str, dict[str, Any]]], save_directory: str = None
) -> dict[str, pd.DataFrame]:
    """
    Create detailed analysis comparing configurations for each lobe.
    """
    analysis_results = {}

    for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
        lobe_analysis = []

        for config_name, batch_data in all_batch_data.items():
            lobe_values = []

            for patient_data in batch_data.values():
                if lobe in patient_data["segments"]:
                    lobe_values.append(patient_data["segments"][lobe])

            if lobe_values:
                expected = EXPECTED_SEGMENTS.get(lobe, 0)
                mean_val = np.mean(lobe_values)

                lobe_analysis.append(
                    {
                        "configuration": config_name,
                        "n_patients": len(lobe_values),
                        "mean": mean_val,
                        "std": np.std(lobe_values),
                        "median": np.median(lobe_values),
                        "min": np.min(lobe_values),
                        "max": np.max(lobe_values),
                        "expected": expected,
                        "meets_expected": mean_val >= expected,
                        "deviation_from_expected": abs(mean_val - expected),
                        "exceeds_by": mean_val - expected,
                    }
                )

        lobe_df = pd.DataFrame(lobe_analysis)

        # Sort by meets_expected (True first), then by deviation
        if not lobe_df.empty:
            lobe_df = lobe_df.sort_values(
                ["meets_expected", "deviation_from_expected"], ascending=[False, True]
            )

        analysis_results[lobe] = lobe_df

        # Save individual lobe analysis
        if save_directory:
            save_path = Path(save_directory) / f"detailed_analysis_{lobe}.csv"
            lobe_df.to_csv(save_path, index=False)
            print(f"Detailed analysis for {lobe} saved to: {save_path}")

    return analysis_results


def create_summary_statistics_table(
    csv_file_paths: dict[str, str], save_directory: str = None
) -> pd.DataFrame:
    """
    Create enhanced summary statistics table from multiple CSV files.
    """

    summary_data = []

    for batch_name, csv_file in csv_file_paths.items():
        if not Path(csv_file).exists():
            print(f"✗ {csv_file} not found")
            continue

        batch_data = load_segment_data_from_csv(csv_file)

        if not batch_data:
            continue

        # Calculate statistics for this batch
        patients = list(batch_data.keys())
        n_patients = len(patients)

        # Segment totals and statistics
        segment_stats = {}
        for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
            lobe_values = []
            for patient_data in batch_data.values():
                if lobe in patient_data["segments"]:
                    lobe_values.append(patient_data["segments"][lobe])

            if lobe_values:
                expected = EXPECTED_SEGMENTS.get(lobe, 0)
                mean_val = np.mean(lobe_values)

                segment_stats[f"{lobe}_total"] = sum(lobe_values)
                segment_stats[f"{lobe}_mean"] = mean_val
                segment_stats[f"{lobe}_std"] = np.std(lobe_values)
                segment_stats[f"{lobe}_median"] = np.median(lobe_values)
                segment_stats[f"{lobe}_expected"] = expected
                segment_stats[f"{lobe}_meets_expected"] = mean_val >= expected
                segment_stats[f"{lobe}_deviation"] = abs(mean_val - expected)

        # Overall statistics
        total_chains_sum = sum(p["total_chains"] for p in batch_data.values())
        level4_edges_sum = sum(p["level4_edges"] for p in batch_data.values())

        summary_data.append(
            {
                "Configuration": batch_name,
                "N_Patients": n_patients,
                **segment_stats,
                "Total_Chains_Sum": total_chains_sum,
                "Total_Chains_Mean": total_chains_sum / n_patients,
                "Level4_Edges_Sum": level4_edges_sum,
                "Level4_Edges_Mean": level4_edges_sum / n_patients,
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Save comprehensive statistics
    if save_directory:
        save_path = Path(save_directory) / "comprehensive_statistics.csv"
        summary_df.to_csv(save_path, index=False)
        print(f"Comprehensive statistics saved to: {save_path}")

        # Also create a simplified version
        simplified_cols = ["Configuration", "N_Patients"]
        for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
            simplified_cols.extend(
                [f"{lobe}_mean", f"{lobe}_std", f"{lobe}_meets_expected"]
            )

        simplified_df = summary_df[simplified_cols]
        save_path_simple = Path(save_directory) / "simplified_statistics.csv"
        simplified_df.to_csv(save_path_simple, index=False)
        print(f"Simplified statistics saved to: {save_path_simple}")

    return summary_df


# Updated comparison heatmaps function (from previous implementation)
def create_comparison_heatmaps_from_csvs(
    csv_file_paths: dict[str, str],
    save_directory: str = None,
    figsize: tuple[int, int] = (20, 12),
    include_totals: bool = True,
) -> plt.Figure | None:
    """
    Create a comparison of heatmaps from multiple CSV statistics files.
    """
    # Load data from all CSV files
    all_batch_data = {}
    for batch_name, csv_file in csv_file_paths.items():
        if Path(csv_file).exists():
            all_batch_data[batch_name] = load_segment_data_from_csv(csv_file)
            print(f"✓ {batch_name}: {len(all_batch_data[batch_name])} patients")
        else:
            print(f"✗ {csv_file} not found")

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

    # Order segments in anatomical order
    segment_order = ["RUL", "RML", "RLL", "LUL", "LLL"]
    found_segments = [seg for seg in segment_order if seg in all_segments]

    # Add any unexpected segments at the end
    found_segments.extend(
        sorted([seg for seg in all_segments if seg not in segment_order])
    )

    # Add total if requested
    if include_totals:
        found_segments.append("TOTAL")

    # Create subplot layout
    n_batches = len(csv_file_paths)
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
    for idx, (batch_name, _) in enumerate(csv_file_paths.items()):
        if batch_name not in all_batch_data:
            continue

        batch_data = all_batch_data[batch_name]

        # Create data for this batch
        detailed_data = []
        for patient in all_patients:
            for segment in found_segments:
                if patient in batch_data:
                    if segment == "TOTAL":
                        segment_count = batch_data[patient]["total_chains"]
                    else:
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
        heatmap_data = heatmap_data.reindex(columns=found_segments, fill_value=0)

        # Create heatmap
        ax = axes[idx]
        cmap = "Oranges" if not include_totals else "RdYlBu_r"

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="g",
            cmap=cmap,
            cbar_kws={"label": "Segment Chains"},
            ax=ax,
            linewidths=0.5,
        )

        ax.set_title(f"{batch_name}", fontweight="bold")
        ax.set_xlabel("Lobes")
        ax.set_ylabel("Patient")

        # Rotate x-axis labels for better readability
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for idx in range(n_batches, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Segment Chain Heatmaps: Comparison Across Configurations",
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


def analyze_single_csv(
    csv_file_path: str,
    save_directory: str = None,
    create_plots: bool = True,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Analyze a single CSV file and generate summary statistics and plots.

    Args:
        csv_file_path: Path to the CSV statistics file
        save_directory: Optional directory to save outputs
        create_plots: Whether to create violin plots (set to False for quick stats only)
        print_summary: Whether to print summary to console

    Returns:
        Dictionary containing analysis results
    """

    if save_directory:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Load data
    batch_data = load_segment_data_from_csv(csv_file_path)
    if not batch_data:
        print("No valid data found in CSV file")
        return results

    # Calculate summary statistics
    summary_stats = calculate_single_csv_statistics(csv_file_path, save_directory)
    results["summary_statistics"] = summary_stats

    # Create violin plots if requested
    if create_plots:
        violin_figs = {}
        for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
            violin_fig = create_violin_plot_single_csv(
                batch_data,
                lobe,
                save_path=save_dir / f"violin_{lobe}.png" if save_directory else None,
            )
            violin_figs[lobe] = violin_fig
        results["violin_figures"] = violin_figs

    # Print summary if requested
    if print_summary:
        print_single_csv_summary(batch_data, Path(csv_file_path).stem)

    return results


def calculate_single_csv_statistics(
    csv_file_path: str, save_directory: str = None
) -> pd.DataFrame:
    """Calculate statistics for a single CSV file."""

    batch_data = load_segment_data_from_csv(csv_file_path)
    if not batch_data:
        return pd.DataFrame()

    config_name = Path(csv_file_path).stem
    patients = list(batch_data.keys())
    n_patients = len(patients)

    # Segment statistics
    segment_stats = {}
    for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
        lobe_values = []
        for patient_data in batch_data.values():
            if lobe in patient_data["segments"]:
                lobe_values.append(patient_data["segments"][lobe])

        if lobe_values:
            expected = EXPECTED_SEGMENTS.get(lobe, 0)
            mean_val = np.mean(lobe_values)

            segment_stats[f"{lobe}_total"] = sum(lobe_values)
            segment_stats[f"{lobe}_mean"] = mean_val
            segment_stats[f"{lobe}_std"] = np.std(lobe_values)
            segment_stats[f"{lobe}_median"] = np.median(lobe_values)
            segment_stats[f"{lobe}_expected"] = expected
            segment_stats[f"{lobe}_meets_expected"] = mean_val >= expected
            segment_stats[f"{lobe}_deviation"] = abs(mean_val - expected)

    # Overall statistics
    total_chains_sum = sum(p["total_chains"] for p in batch_data.values())
    level4_edges_sum = sum(p["level4_edges"] for p in batch_data.values())

    summary_data = {
        "Configuration": config_name,
        "N_Patients": n_patients,
        **segment_stats,
        "Total_Chains_Sum": total_chains_sum,
        "Total_Chains_Mean": total_chains_sum / n_patients,
        "Level4_Edges_Sum": level4_edges_sum,
        "Level4_Edges_Mean": level4_edges_sum / n_patients,
    }

    summary_df = pd.DataFrame([summary_data])

    # Save if requested
    if save_directory:
        save_path = Path(save_directory) / f"{config_name}_statistics.csv"
        summary_df.to_csv(save_path, index=False)
        print(f"Statistics saved to: {save_path}")

    return summary_df


def create_violin_plot_single_csv(
    batch_data: dict[str, dict[str, Any]],
    lobe: str,
    save_path: str = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Create violin plot for a single CSV/configuration."""

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Extract lobe values
    lobe_values = []
    for patient_data in batch_data.values():
        if lobe in patient_data["segments"]:
            lobe_values.append(patient_data["segments"][lobe])

    if not lobe_values:
        ax.text(
            0.5,
            0.5,
            f"No data available for {lobe}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        return fig

    # Create violin plot
    parts = ax.violinplot(
        [lobe_values], positions=[0], widths=0.7, showmeans=True, showmedians=True
    )

    # Color the violin plot
    parts["bodies"][0].set_facecolor("lightblue")
    parts["bodies"][0].set_alpha(0.7)

    # Add expected line
    expected = EXPECTED_SEGMENTS.get(lobe, 0)
    if expected > 0:
        ax.axhline(
            y=expected,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Expected: {expected}",
        )

    # Customize plot
    ax.set_ylabel("Number of Segments")
    ax.set_title(f"{lobe} - Segment Distribution", fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["Current Run"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics annotation
    mean_val = np.mean(lobe_values)
    std_val = np.std(lobe_values)
    median_val = np.median(lobe_values)

    stats_text = f"μ={mean_val:.1f}±{std_val:.1f}, median={median_val:.1f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Violin plot for {lobe} saved to: {save_path}")

    return fig


def print_single_csv_summary(batch_data: dict[str, dict[str, Any]], config_name: str):
    """Print a nice summary of the analysis results."""

    print("\n" + "=" * 60)
    print(f"SEGMENT ANALYSIS SUMMARY: {config_name}")
    print("=" * 60)

    n_patients = len(batch_data)
    print(f"Number of patients: {n_patients}")

    print(f"\n{'Lobe':<6} {'Expected':<8} {'Mean':<8} {'Std':<8} {'Status':<10}")
    print("-" * 50)

    all_meet_expected = True

    for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
        lobe_values = []
        for patient_data in batch_data.values():
            if lobe in patient_data["segments"]:
                lobe_values.append(patient_data["segments"][lobe])

        if lobe_values:
            expected = EXPECTED_SEGMENTS.get(lobe, 0)
            mean_val = np.mean(lobe_values)
            std_val = np.std(lobe_values)

            meets_expected = mean_val >= expected
            status = "✓ PASS" if meets_expected else "✗ FAIL"

            print(
                f"{lobe:<6} {expected:<8} {mean_val:<8.1f} {std_val:<8.1f} {status:<10}"
            )

            if not meets_expected:
                all_meet_expected = False

    # Overall statistics
    total_chains = sum(p["total_chains"] for p in batch_data.values())
    level4_edges = sum(p["level4_edges"] for p in batch_data.values())

    print(f"\nTotal segment chains: {total_chains}")
    print(f"Average chains per patient: {total_chains / n_patients:.1f}")
    print(f"Total level 4 edges: {level4_edges}")
    print(f"Average level 4 edges per patient: {level4_edges / n_patients:.1f}")

    print(
        f"\nOVERALL STATUS: {'✓ ALL LOBES MEET EXPECTED' if all_meet_expected else '⚠ SOME LOBES BELOW EXPECTED'}"
    )
    print("=" * 60)


# Quick analysis function for end-of-run usage
def quick_analysis(csv_file_path: str, verbose: bool = True) -> dict[str, float]:
    """
    Quick analysis function perfect for calling at the end of a labeling run.

    Args:
        csv_file_path: Path to the global_labelling_stats.csv file
        verbose: Whether to print detailed summary

    Returns:
        Dictionary with key metrics
    """
    batch_data = load_segment_data_from_csv(csv_file_path)
    if not batch_data:
        return {}

    # Calculate key metrics
    metrics = {}
    lobes_meeting_expected = 0

    for lobe in ["RUL", "RML", "RLL", "LUL", "LLL"]:
        lobe_values = []
        for patient_data in batch_data.values():
            if lobe in patient_data["segments"]:
                lobe_values.append(patient_data["segments"][lobe])

        if lobe_values:
            expected = EXPECTED_SEGMENTS.get(lobe, 0)
            mean_val = np.mean(lobe_values)
            metrics[f"{lobe}_mean"] = mean_val
            metrics[f"{lobe}_meets_expected"] = mean_val >= expected

            if mean_val >= expected:
                lobes_meeting_expected += 1

    metrics["total_lobes_meeting_expected"] = lobes_meeting_expected
    metrics["percentage_lobes_meeting_expected"] = (lobes_meeting_expected / 5) * 100

    # Print summary if verbose
    if verbose:
        print_single_csv_summary(batch_data, Path(csv_file_path).stem)
        print(
            f"\nQuick Summary: {lobes_meeting_expected}/5 lobes meet expected segment counts"
        )

    return metrics


# Example usage scenarios:
if __name__ == "__main__":
    # 1. Quick analysis at end of labeling run
    csv_path = "./global_labelling_stats.csv"

    # Just get the key metrics
    metrics = quick_analysis(csv_path, verbose=True)

    # 2. Full analysis with plots
    results = analyze_single_csv(
        csv_file_path=csv_path,
        save_directory="./analysis_output",
        create_plots=True,
        print_summary=True,
    )

    # 3. Just statistics, no plots (faster)
    results_no_plots = analyze_single_csv(
        csv_file_path=csv_path,
        save_directory="./analysis_output",
        create_plots=False,
        print_summary=True,
    )
