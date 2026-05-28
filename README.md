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

### Single Patient Processing

Use `process_patient.py` for complete analysis of one patient:

```bash
# Complete pipeline with artery mask only
python scripts/process_patient.py \
    --artery-mask data/patient_001_arteries.nii.gz \
    --lung-masks-dir data/patient_001_lungs/

# Complete pipeline including embolic analysis
python scripts/process_patient.py \
    --artery-mask data/patient_001_arteries.nii.gz \
    --lung-masks-dir data/patient_001_lungs/ \
    --embolism-mask data/patient_001_embolism.nii.gz \
    --output-dir results/patient_001/
```

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
- `--arteries-dir`: Path to the artery masks folder.
- `--embolies-dir`: Path to the embolism masks folder.
- `--images-dir`: Path to the original images folder.
- `--lungs-dir`: Path to the lung masks folder.
- `--output-dir`: Base output directory for all patients (default: `batch_results`).

#### Patient selection

- `--patient-list`: Text file with patient IDs to process, one per line.
- `--exclude-patients`: Space-separated list of patient IDs to exclude.
- `--max-patients`: Maximum number of patients to process.
- `--validate-only`: Only validate patient files and do not run the pipeline.

#### Processing options

- `--radius-ratio-threshold`: Threshold for radius ratio to remove thin branches (default: `0.3`).
- `--diameter-threshold`: Minimum diameter ratio for trunk vessels (default: `0.5`).
- `--length-threshold`: Maximum length/diameter ratio for trunk vessels (default: `1.5`).
- `--coordinate-format`: Coordinate format for saved graphs: `voxel`, `real`, or `both` (default: `both`).
- `--save-skeletons`: Save binary and edge_id skeletons at each step.
- `--save-colored-volumes`: Save vessel volumes with specified coloring. Valid values: `edge_id`, `level`, `lobe`.
- `--save-all-step-volumes`: Save vessel volumes at every processing step.
- `--save-cycle-skeletons`: Save skeletons of detected cycles before and after cleaning.

#### Embolism and enriched graph options

- `--embolism-separation-method`: Method for separating embolisms: `connected_components` or `connected_components_with_threshold`.
- `--embolism-fusion-threshold`: Distance threshold for fusing nearby embolism components (default: `5.0`).
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

- `--generate-heatmaps`: Generate segment heatmaps after processing.
- `--batch-name`: Name for the batch (used in heatmap titles, defaults to the output directory name).
- `--comparison-batches`: List of other batch directories for comparison heatmaps.
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
results/patient_001/
├── skeleton.nii.gz         # 3D skeleton (centerlines) of vessels
└── distance_map.nii.gz     # Distance transform for radius estimation
```

### Step 2: Graph Construction (`2_build_graph.py`)

**Purpose**: Convert skeleton and distance map into mathematical graph representation.

**Process**: Creates NetworkX graph with nodes (bifurcations/endpoints) and edges (vessel segments) including geometric properties.

```bash
# Build graph with both coordinate formats
python scripts/2_build_graph.py \
    --skeleton results/patient_001/skeleton.nii.gz \
    --distance-map results/patient_001/distance_map.nii.gz

# With skeleton visualization and validation
python scripts/2_build_graph.py \
    --skeleton results/patient_001/skeleton.nii.gz \
    --distance-map results/patient_001/distance_map.nii.gz \
    --save-skeleton \
    --detailed-validation

# Specify coordinate format
python scripts/2_build_graph.py \
    --skeleton results/patient_001/skeleton.nii.gz \
    --distance-map results/patient_001/distance_map.nii.gz \
    --coordinate-format real
```

**Key Options**:
- `--coordinate-format`: `voxel`, `real`, or `both` (default: both)
- `--save-skeleton`: Generate skeleton reconstructions for validation
- `--detailed-validation`: Show comprehensive validation metrics

**Output**:
```
results/patient_001/
├── branch_graph.json                     # Graph with voxel coordinates
├── branch_graph_real_coords.json         # Graph with real-world coordinates (mm)
├── branch_graph_skeleton.nii.gz          # Reconstructed skeleton (validation)
└── branch_graph_skeleton_edge_id.nii.gz  # Colored skeleton by segment
```

### Step 3: Graph Cleaning (`3_clean_graph.py`)

**Purpose**: Remove artifacts and optimize graph topology for physiological accuracy.

**Process**: Filters spurious branches, removes cycles, eliminates thin branches below threshold.

```bash
# Basic cleaning with default threshold
python scripts/3_clean_graph.py \
    --graph results/patient_001/branch_graph.json

# Custom radius ratio threshold and visualization
python scripts/3_clean_graph.py \
    --graph results/patient_001/branch_graph.json \
    --radius-ratio-threshold 0.3 \
    --save-skeleton \
    --save-colored-volumes edge_id

# Multiple visualization types
python scripts/3_clean_graph.py \
    --graph results/patient_001/branch_graph.json \
    --save-colored-volumes edge_id level lobe
```

**Key Options**:
- `--radius-ratio-threshold`: Minimum radius ratio for branch preservation (default: 0.3)
- `--save-skeleton`: Generate cleaned skeleton files
- `--save-colored-volumes`: Create colored volume visualizations

**Output**:
```
results/patient_001/
├── cleaned_graph.json                    # Cleaned graph structure
├── cleaned_graph_real_coords.json        # Real-world coordinate version
├── cleaned_graph_skeleton.nii.gz         # Cleaned skeleton
└── cleaned_graph_vessel_volume_edge_id.nii.gz  # Colored vessel volume
```

### Step 4: Graph Orientation (`4_orient_graph.py`)

**Purpose**: Establish physiological flow direction from pulmonary trunk to periphery.

**Process**: Identifies pulmonary trunk as root, propagates orientation based on decreasing vessel diameter.

```bash
# Basic orientation
python scripts/4_orient_graph.py \
    --graph results/patient_001/cleaned_graph.json

# With visualization outputs
python scripts/4_orient_graph.py \
    --graph results/patient_001/cleaned_graph.json \
    --save-skeleton \
    --save-colored-volumes edge_id level
```

**Output**:
```
results/patient_001/
├── oriented_graph.json                   # Oriented graph with flow direction
├── oriented_graph_real_coords.json       # Real-world coordinates
├── oriented_graph_skeleton_level.nii.gz  # Skeleton colored by hierarchy level
└── oriented_graph_vessel_volume_level.nii.gz  # Volume colored by hierarchy
```

### Step 5: Anatomical Labeling (`5_label_graph.py`)

**Purpose**: Assign anatomical information including laterality, lobe assignment, and vessel classification.

**Process**: Uses spatial intersection with lung masks and morphological criteria to label vessels.

```bash
# Basic labeling with lung masks
python scripts/5_label_graph.py \
    --graph results/patient_001/oriented_graph.json \
    --lung-masks-dir data/patient_001_lungs/

# With custom thresholds and visualization
python scripts/5_label_graph.py \
    --graph results/patient_001/oriented_graph.json \
    --lung-masks-dir data/patient_001_lungs/ \
    --diameter-threshold 0.6 \
    --length-threshold 2.0 \
    --save-colored-volumes edge_id level lobe
```

**Key Options**:
- `--diameter-threshold`: Minimum diameter for trunk vessels (default: 0.5)
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
results/patient_001/
├── labeled_graph.json                    # Anatomically labeled graph
├── labeled_graph_real_coords.json        # Real-world coordinates
├── labeled_graph_vessel_volume_lobe.nii.gz      # Volume colored by lobe
├── labeled_graph_vessel_volume_level.nii.gz     # Volume colored by level
└── labeled_graph_skeleton_level.nii.gz          # Skeleton colored by level
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
    --vessel-graph results/patient_001/labeled_graph.json \
    --vessel-mask data/patient_001_arteries.nii.gz

# With custom separation parameters
python scripts/6_analyze_embolism.py \
    --embolism-mask data/patient_001_embolism.nii.gz \
    --patient-name patient_001 \
    --vessel-graph results/patient_001/labeled_graph.json \
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
results/patient_001/embolism_analysis/
├── patient_001_embolism_report.txt       # Statistical summary
├── patient_001_embolism_metrics.csv      # Quantitative metrics
├── embolic_masks_colored.nii.gz          # Visualization with emboli IDs
├── individual_emboli/                    # Individual embolic masks
│   ├── embolism_1.nii.gz
│   ├── embolism_2.nii.gz
│   └── ...
└── enriched_graph.json                   # Graph with obstruction metrics (if integrated)
```

## Pipeline Outputs Summary

After complete processing, the full output structure contains:

```
results/patient_001/
├── skeleton.nii.gz                       # Step 1: Preprocessing
├── distance_map.nii.gz
├── branch_graph.json                     # Step 2: Graph construction
├── branch_graph_real_coords.json
├── cleaned_graph.json                    # Step 3: Graph cleaning
├── cleaned_graph_real_coords.json
├── oriented_graph.json                   # Step 4: Graph orientation
├── oriented_graph_real_coords.json
├── labeled_graph.json                    # Step 5: Anatomical labeling
├── labeled_graph_real_coords.json
├── enriched_graph.json                   # Step 6: Embolic integration (if applicable)
├── *_skeleton*.nii.gz                    # Skeleton visualizations
├── *_vessel_volume_*.nii.gz              # Vessel volume visualizations
└── embolism_analysis/                    # Embolic analysis results
    ├── patient_001_embolism_report.txt
    ├── patient_001_embolism_metrics.csv
    ├── embolic_masks_colored.nii.gz
    └── individual_emboli/
```

## Configuration Options

Most scripts support common configuration options:

- `--output-dir`: Specify custom output directory
- `--coordinate-format`: Choose coordinate system (`voxel`, `real`, `both`)
- `--save-skeleton`: Generate skeleton visualizations
- `--save-colored-volumes`: Create colored volume visualizations
- Various thresholds and parameters specific to each analysis step

For detailed options of each script, use the `--help` flag:

```bash
python scripts/[script_name].py --help
```
