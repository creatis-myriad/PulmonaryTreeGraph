# PulmonaryTreeGraph

A pipeline for extracting morphological and embolic biomarkers from masks of arteries, embolies and lungs segmented from CT pulmonary angiography (CTPA) scans. It creates graph-based models of the pulmonary vascular tree and enrich the graph with pulmonary embolism (PE) information.

## Features

- **Graph-based vascular modeling**: Complete pulmonary arterial tree representation
- **Anatomical labeling**: Hierarchical classification with laterality, lobe and hierarchical level
- **Embolic quantification**: Precise localization and volumetric information of PE
- **Obstruction metrics**: Volumetric and transversal obstruction calculations
- **Modular pipeline**: Step-by-step analysis with flexible execution

## Installation

Create a virtual environment using uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Structure

```bash
PulmonaryTreeGraph/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── pulmonarytreegraph/           # Core library
│       ├── emboli/                   # Emboli analysis and characterization
│       │   ├── embolie_construction.py  # Individual emboli detection and properties
│       │   └── embolie_analysis.py      # Complete emboli analysis pipeline
│       ├── graph/                    # Graph creation, processing and labeling
│       │   ├── graph_construction.py    # Skeleton to graph conversion
│       │   ├── graph_cleaning.py        # Artifact and cycles removal
│       │   ├── graph_orienting.py       # Physiological flow direction
│       │   └── graph_labelling.py       # Anatomical hierarchy and lobe assignment
│       ├── pipeline/                 # Core pipeline orchestration
│       │   └── core.py                  # End-to-end pipeline functions
│       ├── utils/                    # Common utilities and validation
│       │   ├── io.py                    # File I/O and data management
│       │   ├── metrics.py               # Quantitative calculations
│       │   └── curve_planar_reformat.py # CPR for cross-sectional analysis
│       ├── preprocessing.py          # Skeletonization and distance map creation
│       └── graph_enrichment_with_embolism.py  # Vascular-embolic integration
│
├── scripts/                          # CLI utilities for standalone use
│   ├── 1_preprocess.py                   # Preprocessing script
│   ├── 2_build_graph.py                   # Graph construction script
│   ├── 3_clean_graph.py                   # Graph cleaning script
│   ├── 4_orient_graph.py                  # Graph orientation script
│   ├── 5_label_graph.py                   # Anatomical labeling script
│   ├── 6_analyze_embolism.py            # Emboli analysis script
│   ├── batch_process_patients.py        # Complete pipeline execution for multiple patients
│   ├── collect_enriched_graphs.py         # Copy enriched graphs to a central directory
│   └── scores.py                         # Compute obstruction and literature scores
│
└── tests/
```

# Pipeline Usage

The pipeline can be executed either step-by-step using individual scripts or end-to-end using the complete pipeline scripts.

## Complete Pipeline

### Batch Processing

Use `batch_process_patients.py` for processing multiple patients:

```bash
# Process all patients in a directory structure
python scripts/batch_process_patients.py \
    --data-dir /path/to/patient/data/ \
    --output-dir /path/to/results/
```

### Batch Processing Options

These options are grouped by purpose to make it easier to configure batch runs.

#### Input / Output

- `--data-dir`: Base directory containing `arteries_mask/`, `embolie_mask/`, `images/`, and `lungs_mask/`.
  - If `--data-dir` is used, the script automatically builds the four per-type paths from it.
  - In this case, `--arteries-dir`, `--embolies-dir`, `--images-dir`, and `--lungs-dir` are not required.
- `--arteries-dir`: Path to the artery masks folder. Optional when `--data-dir` is provided.
- `--embolies-dir`: Path to the embolism masks folder. Optional when `--data-dir` is provided.
- `--images-dir`: Path to the original images folder. Optional; current batch logic uses image files only for metadata/logging and does not require them to run the pipeline.
- `--lungs-dir`: Path to the lung masks folder. Optional when `--data-dir` is provided.
- `--output-dir`: Base output directory for all patients (default: `batch_results`).

#### Patient selection

- `--patient-list`: Text file with patient IDs to process, one per line.
- `--exclude-patients`: Space-separated list of patient IDs to exclude.
- `--max-patients`: Maximum number of patients to process.
- `--validate-only`: Only validate patient files and do not run the pipeline.

#### Processing options

##### Cleaning options
- `--save-cycle-skeletons`: Save skeletons of detected cycles before and after cleaning.
##### Cleaning options
- `--save-cycle-skeletons`: Save skeletons of detected cycles before and after cleaning.
- `--radius-ratio-threshold`: Threshold for radius ratio to remove thin branches (default: `0.3`).
- `--protect-not-only-central-vessels`: Whether to protect only the largest central vessel or all large terminal vessels during cleaning. If set, all terminal vessels above the diameter threshold will be protected, regardless of their centrality (default: `False`, adapted to pulmonary anatomy).
- `--min-diameter-vessel-protected`: Minimum diameter (in mm) of vessels to protect during cleaning (default: 10.0).

##### Labeling options
- `--protect-not-only-central-vessels`: Whether to protect only the largest central vessel or all large terminal vessels during cleaning. If set, all terminal vessels above the diameter threshold will be protected, regardless of their centrality (default: `False`, adapted to pulmonary anatomy).
- `--min-diameter-vessel-protected`: Minimum diameter (in mm) of vessels to protect during cleaning (default: 10.0).

##### Labeling options
- `--diameter-threshold`: Minimum diameter ratio for trunk vessels (default: `0.7`).
- `--length-threshold`: Maximum length/diameter ratio for trunk vessels (default: `1.5`).
- `--save-colored-volumes`: Save vessel volumes with specified coloring. Valid values: `edge_id`, `level`, `lobe`. (The `edge_id` option is required for embolism analysis and will be automatically enabled if `--vessel-graph` is provided for embolism analysis, but can also be enabled during cleaning and labeling steps for visual validation).

##### Visualization options
- `--save-colored-volumes`: Save vessel volumes with specified coloring. Valid values: `edge_id`, `level`, `lobe`. (The `edge_id` option is required for embolism analysis and will be automatically enabled if `--vessel-graph` is provided for embolism analysis, but can also be enabled during cleaning and labeling steps for visual validation).

##### Visualization options
- `--coordinate-format`: Coordinate format for saved graphs: `voxel`, `real`, or `both` (default: `both`).
- `--save-skeletons`: Save binary and edge_id skeletons at each step.
- `--save-all-step-volumes`: Save vessel volumes at every processing step.

#### Embolism and enriched graph options

- `--embolism-separation-method`: Method for separating embolisms: `connected_components` or `connected_components_with_threshold`, default `connected_components`.
- `--embolism-fusion-threshold`: Distance threshold for fusing nearby embolism components (default: `10.0`mm).
- `--embolism-separation-method`: Method for separating embolisms: `connected_components` or `connected_components_with_threshold`, default `connected_components`.
- `--embolism-fusion-threshold`: Distance threshold for fusing nearby embolism components (default: `10.0`mm).
- `--save-enriched-graphs`: Save enriched graphs into a central `enriched_graphs` folder under the output directory.

#### Literature scoring

- `--compute-qanadli`: Compute Qanadli obstruction score.
- `--compute-mastora`: Compute Mastora obstruction scores (central, peripheral, global).
- `--compute-all-scores`: Compute all available literature scores (`--compute-qanadli` + `--compute-mastora`).
- `--obstruction-attr`: Choose obstruction attribute for scoring: `volumetric_obstruction` or `transversal_obstruction_max` (default: `volumetric_obstruction`).

#### Processing control

- `--skip-steps`: Skip specific processing steps. Valid values: `preprocess`, `build`, `clean`, `orient`, `label`, `embolism`.
- `--stop-after`: Stop processing after the specified step.
- `--compute-transversal-obstruction`: Compute transversal obstruction metrics for each vessel segment.
- `--cpr-padding`: Padding for CPR in transversal obstruction (default: `1.0`).

#### Execution and logging

- `--continue-on-error`: Continue processing other patients if one fails (default: `True`).
- `--log-level`: Logging level: `DEBUG`, `INFO`, `WARNING`, or `ERROR` (default: `INFO`).

#### Heatmap generation

- `--generate-heatmaps`: Generate segmental arteries heatmaps after processing.
- `--generate-heatmaps`: Generate segmental arteries heatmaps after processing.
- `--batch-name`: Name for the batch (used in heatmap titles, defaults to the output directory name).
- `--comparison-batches`: List of other batch directories for comparison heatmaps.

#### Scores
- `--scores-only`: Only compute literature scores from existing enriched graphs and skip the pipeline.

## Step-by-Step Pipeline

### Step 1: Preprocessing (`1_preprocess.py`)

**Purpose**: Extract skeleton and distance map from binary artery mask.

**Process**: Applies skeletonization algorithm and computes distance transform for radius estimation.

```bash
# Basic preprocessing
python scripts/1_preprocess.py \
    --artery data/patient_001_arteries.nii.gz

# With custom output directory
python scripts/1_preprocess.py \
    --artery data/patient_001_arteries.nii.gz \
    --output-dir custom_results/patient_001/
```

**Output**:
```
results/patient_001/1_preprocessing/
├── skeleton.nii.gz         # 3D skeleton (centerlines) of vessels
└── distance_map.nii.gz     # Distance transform for radius estimation
```

### Step 2: Graph Construction (`2_build_graph.py`)

**Purpose**: Convert skeleton and distance map into mathematical graph representation.

**Process**: Creates NetworkX graph with nodes (bifurcations/endpoints) and edges (vessel segments) including geometric properties.

```bash
# Build graph with both coordinate formats
python scripts/2_build_graph.py \
    --skeleton results/patient_001/1_preprocessing/skeleton.nii.gz \
    --distance-map results/patient_001/1_preprocessing/distance_map.nii.gz
    --output-dir results/patient_001/2_build_graph/
```

**Key Options**:
- `--coordinate-format`: `voxel`, `real`, or `both` (default: both) :
    - `voxel`: Save graph with voxel coordinates.
    - `real`: Save graph with real-world coordinates in millimeters.
    - `both`: Save both versions of the graph.
- `--save-skeletons`: Generate skeleton reconstructions for validation :
    - `branch_graph_skeleton.nii.gz`: Binary skeleton of the graph.
    - `branch_graph_skeleton_edge_id.nii.gz`: Skeleton colored by edge ID for visual validation.

**Output**:
```
results/patient_001/2_build_graph/
├── graph/
│   ├── branch_graph.json                 # Graph with voxel coordinates
│   └── branch_graph_real_coords.json     # Graph with real-world coordinates (mm)
├── skeleton/
│   ├── branch_graph_skeleton.nii.gz      # Reconstructed skeleton (validation)
│   └── branch_graph_skeleton_edge_id.nii.gz  # Colored skeleton by segment
└── stats/
    └── graph_constructions_stats.csv    # Construction statistics for the graph
```

### Step 3: Graph Cleaning (`3_clean_graph.py`)

**Purpose**: Remove artifacts and optimize graph topology for physiological accuracy.

**Process**: Filters spurious branches, removes cycles, eliminates thin branches below threshold.

```bash
# Basic cleaning with default threshold
python scripts/3_clean_graph.py \
    --graph results/patient_001/2_build_graph/graph/branch_graph.json

# Custom radius ratio threshold and visualization
python scripts/3_clean_graph.py \
    --graph results/patient_001/2_build_graph/graph/branch_graph.json \
    --radius-ratio-threshold 0.3 \
    --save-skeletons \
    --save-colored-volumes edge_id

# Multiple visualization types
python scripts/3_clean_graph.py \
    --graph results/patient_001/2_build_graph/graph/branch_graph.json \
    --save-colored-volumes edge_id level lobe
```

**Key Options**:
- `--radius-ratio-threshold`: Minimum radius ratio for branch preservation (default: 0.3)
- `--save-skeletons`: Generate cleaned skeleton files :
    - `cleaned_graph_skeleton.nii.gz`: Binary skeleton of the cleaned graph.
    - `cleaned_graph_skeleton_edge_id.nii.gz`: Skeleton colored by edge ID for validation.
- `--save-colored-volumes`: Create colored volume visualizations :
    - `edge_id`: Color by segment ID.
    - `level`: Color by hierarchical level.
    - `lobe`: Color by lobe assignment (requires lung masks).
- `--save-cycle-skeletons`: Save skeletons of detected cycles before and after cleaning for validation :
    - `before_cleaning/`: Skeleton of detected cycles before cleaning.
    - `after_cleaning/`: Skeleton of remaining cycles after cleaning.
- `--protect-not-only-central-vessels`: Protect all large terminal vessels above the diameter threshold during cleaning, not just the largest central vessel (default: `False`, adapted to pulmonary anatomy).
- `--min-diameter-vessel-protected`: Minimum diameter (in mm) of vessels to protect during cleaning (default: 10.0). This option works in conjunction with `--protect-largest-central` and defines the minimum diameter for any vessel to be protected during cleaning, whether it's the largest central vessel or other large terminal vessels (if `--protect-not-only-central-vessels` is set).

**Output**:
```
results/patient_001/3_clean_graph/
├── cycle_skeletons/
│   ├── before_cleaning/
│   │   ├── 001_cycle_before_1.nii.gz
│   │   └── 001_cycle_before_2.nii.gz
│   └── after_cleaning/
│       ├── 001_cycle_after_1.nii.gz
│       └── 001_cycle_after_2.nii.gz
├── graph/
│   ├── cleaned_graph.json                # Cleaned graph structure (voxel coords)
│   └── cleaned_graph_real_coords.json    # Cleaned graph (real coords)
├── skeleton/
│   ├── cleaned_graph_skeleton.nii.gz     # Cleaned skeleton (binary)
│   └── cleaned_graph_skeleton_edge_id.nii.gz  # Colored skeleton by segment
├── stats/
│   └── graph_cleaning_stats.csv          # Cleaning statistics and metrics
└── volumes/
    └── cleaned_graph_vessel_volume_edge_id.nii.gz  # Colored vessel volume (if enabled)
```

### Step 4: Graph Orientation (`4_orient_graph.py`)

**Purpose**: Establish physiological flow direction from pulmonary trunk to periphery.

**Process**: Identifies pulmonary trunk as root, propagates orientation based on decreasing vessel diameter.

```bash
# Basic orientation
python scripts/4_orient_graph.py \
    --graph results/patient_001/3_clean_graph/graph/cleaned_graph.json

# With visualization outputs
python scripts/4_orient_graph.py \
    --graph results/patient_001/3_clean_graph/graph/cleaned_graph.json \
    --save-skeleton \
    --save-colored-volumes edge_id
```

**Output**:
```
results/patient_001/4_orient_graph/
├── graph/
│   ├── oriented_graph.json               # Oriented graph with flow direction
│   └── oriented_graph_real_coords.json   # Real-world coordinates
├── skeleton/
│   ├── oriented_graph_skeleton.nii.gz    # Skeleton colored by lowest hierarchy level
│   └── oriented_graph_skeleton_edge_id.nii.gz  # Colored skeleton by segment
└── volumes/
    └── oriented_graph_vessel_volume_edge_id.nii.gz  # Volume colored by edge_id (if enabled)
```

### Step 5: Anatomical Labeling (`5_label_graph.py`)

**Purpose**: Assign anatomical information including laterality, lobe assignment, and vessel classification.

**Process**: Uses spatial intersection with lung masks and morphological criteria to label vessels.

```bash
# Basic labeling with lung masks, and enabling edge_id volume visualization (required for embolism analysis)
python scripts/5_label_graph.py \
    --graph results/patient_001/4_orient_graph/graph/oriented_graph.json \
    --lung-masks-dir data/patient_001_lungs/
    --save-colored-volumes edge_id

# With custom thresholds and volume visualization
python scripts/5_label_graph.py \
    --graph results/patient_001/4_orient_graph/graph/oriented_graph.json \
    --lung-masks-dir data/patient_001_lungs/ \
    --diameter-threshold 0.6 \
    --length-threshold 2.0 \
    --save-colored-volumes edge_id level lobe
```

**Key Options**:
- `--diameter-threshold`: Minimum diameter for trunk vessels (default: 0.7)
- `--length-threshold`: Minimum length for trunk vessels (default: 1.5)
- `--save-colored-volumes`: Visualization by different anatomical properties

**Required Lung Masks**:
```
data/patient_001_lungs/
├── lung_left.nii.gz
├── lung_right.nii.gz
├── lung_upper_lobe_left.nii.gz
├── lung_lower_lobe_left.nii.gz
├── lung_upper_lobe_right.nii.gz
├── lung_middle_lobe_right.nii.gz
└── lung_lower_lobe_right.nii.gz
```

**Output**:
```
results/patient_001/5_label_graph/
├── graph/
│   ├── labeled_graph.json                    # Anatomically labeled graph
│   └── labeled_graph_real_coords.json        # Real-world coordinates
├── skeleton/
│   ├── labeled_graph_skeleton.nii.gz         # Clean binary skeleton from labeled graph
│   ├── labeled_graph_skeleton_edge_id.nii.gz # Colored skeleton by segment
│   └── labeled_graph_skeleton_level.nii.gz   # Colored skeleton by hierarchy level
├── stats/
│   └── graph_labelling_stats.csv             # Labeling statistics and metrics
└── volumes/
    ├── labeled_graph_vessel_volume_edge_id.nii.gz  # Volume colored by edge_id (if enabled, require for embolism analysis)
    ├── labeled_graph_vessel_volume_level.nii.gz    # Volume colored by level (if enabled)
    └── labeled_graph_vessel_volume_lobe.nii.gz     # Volume colored by lobe (if enabled)
```

### Step 6: Embolic Analysis (`6_analyze_embolism.py`)

**Purpose**: Characterize individual emboli and calculate obstruction metrics when integrated with vascular data.

**Process**: Separates emboli into individual components, calculates volumes, and optionally integrates with vessel graph for obstruction analysis.

```bash
# Basic embolic analysis (standalone)
python scripts/6_analyze_embolism.py \
    --embolism-mask data/patient_001_embolism.nii.gz \
    --patient-name patient_001

# Complete analysis with vascular integration
python scripts/6_analyze_embolism.py \
    --embolism-mask data/patient_001_embolism.nii.gz \
    --patient-name patient_001 \
    --vessel-graph results/patient_001/5_label_graph/graph/labeled_graph.json \
    --vessel-mask data/patient_001_arteries.nii.gz

# With custom separation parameters
python scripts/6_analyze_embolism.py \
    --embolism-mask data/patient_001_embolism.nii.gz \
    --patient-name patient_001 \
    --vessel-graph results/patient_001/5_label_graph/graph/labeled_graph.json \
    --vessel-mask data/patient_001_arteries.nii.gz \
    --separation-method connected_components_with_threshold \
    --fusion-threshold 8.0 \
    --calculate-transversal-obstruction
```

**Key Options**:
- `--separation-method`: `connected_components` or `connected_components_with_threshold`
- `--fusion-threshold`: Distance threshold (mm) for fusing nearby emboli (default: 10.0)
- `--calculate-transversal-obstruction`: Enable CPR-based cross-sectional analysis
- `--vessel-graph`: Labeled vessel graph for integration analysis
- `--vessel-mask`: Segmented vessel mask for spatial analysis

**Output**:
```
results/patient_001/6_embolism_analysis/
├── graph/                                # Only created when full vascular integration is performed
│   ├── enriched_graph.json               # Enriched embolism-vessel graph
│   └── enriched_graph_real_coords.json   # Real-world coordinate version
├── reports/
│   └── 001_embolism_report.txt   # Statistical summary
├── scores/
│   ├── mastora_scores.csv  # Mastora obstruction scores (if enabled)
│   └── qanadli_scores.csv   # Qanadli obstruction scores (if enabled)
├── stats/
│   ├── 001_embolism_metrics.csv  # Embolism metrics (volume, location, etc.)
│   └── 001_embolism_metrics_enriched.csv  # Enriched embolism metrics with vascular integration (if performed)
└── volumes/
    ├── 001_embolisms_colored.nii.gz      # Visualization with emboli IDs
    └── individual_emboli/                # Individual embolism masks
        ├── 001_embolism_1.nii.gz
        ├── 001_embolism_2.nii.gz
        └── ...
```

### Batch Report and Global Statistics Layout

Batch runs now collect report files and global stats in dedicated, timestamped folders under the batch output root.

For example:

```
/path/to/results/
├── reports/
│   └── 1700000000/                          # run ID folder
│       └── batch_processing_report_1700000000.txt
└── global_stats/
    └── 1700000000/                          # same run ID folder
        ├── global_construction_stats.csv
        ├── global_cleaning_stats.csv
        ├── global_labelling_stats.csv
        ├── global_embolism_statistics_enriched.csv
        ├── global_qanadli_scores.csv
        └── global_mastora_scores.csv
```

### Patient-level Scores

When literature scores are computed, patient score files are saved inside a dedicated `scores/` folder beneath the patient output directory.

Example:

```
results/patient_001/
├── scores/
│   ├── qanadli_scores.csv
│   └── mastora_scores.csv
└── 6_embolism_analysis/
    ├── graph/
    ├── reports/
    └── volumes/
```

## Pipeline Outputs Summary

After complete processing, the full output structure contains:

```
results/patient_001/
├── 1_preprocessing/
│   ├── skeleton.nii.gz
│   └── distance_map.nii.gz
├── 2_build_graph/
│   ├── graph/
│   │   ├── branch_graph.json
│   │   └── branch_graph_real_coords.json
│   ├── skeleton/
│   │   ├── branch_graph_skeleton.nii.gz
│   │   └── branch_graph_skeleton_edge_id.nii.gz
│   └── stats/
│       └──graph_construction_stats.csv
├── 3_clean_graph/
│   ├── cycle_skeletons/
│   │   ├── before_cleaning/
│   │   │   ├── 001_cycle_before_1.nii.gz
│   │   │   └── 001_cycle_before_2.nii.gz
│   │   └── after_cleaning/
│   │       ├── 001_cycle_after_1.nii.gz
│   │       └── 001_cycle_after_2.nii.gz
│   ├── graph/
│   │   ├── cleaned_graph.json
│   │   └── cleaned_graph_real_coords.json
│   ├── skeleton/
│   │   ├── cleaned_graph_skeleton.nii.gz
│   │   └── cleaned_graph_skeleton_edge_id.nii.gz
│   ├── stats/
│   │   └── graph_cleaning_stats.csv
│   └── volumes/
│       └── cleaned_graph_vessel_volume_edge_id.nii.gz
├── 4_orient_graph/
│   ├── graph/
│   │   ├── oriented_graph.json
│   │   └── oriented_graph_real_coords.json
│   ├── skeleton/
│   │   ├── oriented_graph_skeleton.nii.gz
│   │   └── oriented_graph_skeleton_edge_id.nii.gz
│   └── volumes/
│       └── oriented_graph_vessel_volume_edge_id.nii.gz
├── 5_label_graph/
│   ├── graph/
│   │   ├── labeled_graph.json
│   │   └── labeled_graph_real_coords.json
│   ├── skeleton/
│   │   ├── labeled_graph_skeleton.nii.gz
│   │   ├── labeled_graph_skeleton_edge_id.nii.gz
│   │   └── labeled_graph_skeleton_level.nii.gz
│   ├── stats/
│   │   └── graph_labelling_stats.csv
│   └── volumes/
│       ├── labeled_graph_vessel_volume_edge_id.nii.gz
│       ├── labeled_graph_vessel_volume_level.nii.gz
│       └── labeled_graph_vessel_volume_lobe.nii.gz
└── 6_embolism_analysis/
    ├── graph/
    │   ├── enriched_graph.json
    │   └── enriched_graph_real_coords.json
    ├── reports/
    │   ├── 001_embolism_report.txt
    │   ├── 001_embolism_metrics.csv
    │   └── 001_embolism_metrics_enriched.csv
    ├── scores/
    │   ├── 001_qanadli_scores.csv
    │   └── 001_mastora_scores.csv
    ├── stats/
    │   ├── 001_embolism_metrics.csv
    │   └── 001_embolism_metrics_enriched.csv
    └── volumes/
        ├── 001_embolisms_colored.nii.gz
        └── individual_emboli/
            ├── 001_embolism_1.nii.gz
            ├── 001_embolism_2.nii.gz
            └── ...
```
