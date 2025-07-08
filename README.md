# PulmonaryTreeGraph

A pipeline for extracting morphological and embolic biomarkers from masks of arteries, embolies and lungs segmented from CT pulmonary angiography (CTPA) scans.

## Overview

PulmonaryTreeGraph creates graph-based models of the pulmonary vascular tree and enrich the graph with pulmonary embolism (PE) information.

## Features

- Graph-based modeling of pulmonary vasculature
- Precise localization and quantification of emboli
- Modular, step-by-step analysis pipeline
- Extraction of novel quantitative biomarkers

## Installation

Create a virtual environment using uv

```bash
uv venv
source .venv/bin/activate
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
│   └── pulmonarytreegraph/ # Core library
│       ├── emboli/ # Emboli detection & mapping
│       ├── graph/ # Graph creation and cleaning, hierarchical & anatomical labeling
│       ├── pipeline/ # Core pipeline functions
│       ├── utils/ # Common helpers and validation
│       └── preprocessing.py # Skeletonization and distancemap creation
│
├── scripts/ # CLI utilities for Snakemake or standalone use
│   ├── analyze_embolie.py
│   ├── build_graph.py
│   ├── clean_graph.py
│   ├── label_graph.py
│   ├── preprocess_arteries_mask.py
│   └── process_patient.py # End-to-end or step-wise execution
│
├── workflows/
│   └── Snakefile # Full Snakemake pipeline
│   └── config.yaml # Inputs and parameters
│
├── data/ # Example inputs
│
└── tests/ # Unit and integration tests
```

# Usage

### 1. Preprocessing Artery Mask

The first step converts binary artery masks into skeleton and distance map representations needed for graph construction.

**Description**: Creates a 3D skeleton (centerline) of the vascular tree and computes distance transforms for radius estimation.

**Usage**:
```bash
# Basic usage - output goes to results/patient_name/
python scripts/preprocess_arteries_mask.py --artery data/0001_arteries.nii.gz

# With custom output directory
python scripts/preprocess_arteries_mask.py \
    --artery data/0001_arteries.nii.gz \
    --output-dir custom_output/
```

**Expected Output**:
```
results/0001/
├── skeleton.nii.gz         # 3D skeleton (centerline) of vessels
└── distance_map.nii.gz     # Distance transform for radius estimation
```

### 2. Building Vessel Graph

Converts the skeleton and distance map into a graph-based representation with nodes and edges.

**Description**: Creates a NetworkX graph where:
- **Nodes**: Junction points and endpoints of vessels
- **Edges**: Vessel segments with centerline coordinates and radius information
- **Attributes**: Each edge contains 3D centerline points, radius values, and geometric properties

**Usage**:
```bash
# Basic usage - creates both voxel and real-world coordinate graphs
python scripts/build_graph.py \
    --skeleton results/0001/skeleton.nii.gz \
    --distance-map results/0001/distance_map.nii.gz

# With skeleton visualization and detailed validation
python scripts/build_graph.py \
    --skeleton results/0001/skeleton.nii.gz \
    --distance-map results/0001/distance_map.nii.gz \
    --save-skeleton \
    --detailed-validation

# Save only voxel coordinates
python scripts/build_graph.py \
    --skeleton results/0001/skeleton.nii.gz \
    --distance-map results/0001/distance_map.nii.gz \
    --coordinate-format voxel
```

**Arguments**:
- `--skeleton`: Path to skeleton file from preprocessing
- `--distance-map`: Path to distance map file from preprocessing
- `--output-dir`: Custom output directory (default: results/patient_name)
- `--coordinate-format`: `voxel`, `real`, or `both` (default: both)
- `--save-skeleton`: Save binary and colored skeleton reconstructions
- `--detailed-validation`: Show detailed graph validation statistics

**Expected Output**:
```
results/0001/
├── branch_graph.json                     # Graph with voxel coordinates
├── branch_graph_real_coords.json         # Graph with real-world coordinates
├── branch_graph_skeleton.nii.gz          # Binary skeleton from graph (if --save-skeleton)
└── branch_graph_skeleton_edge_id.nii.gz  # Colored skeleton by edge ID (if --save-skeleton)
```

**Graph Structure**:
The JSON files contain NetworkX graphs with:
```json
{
  "directed": false,
  "multigraph": false,
  "graph": {},
  "nodes": [
    {
      "pos": [66, 92, 148],
      "radius": 1.0,
      "id": 0
    },
    {
      "pos": [128, 110, 169],
      "radius": 3.818119098098647,
      "id": 534
    }
  ],
  "links": [
    {
      "id": 1,
      "name": "edge_1",
      "centerline": [[66,92,148], [67,93,149], [68,94,150], ...],
      "radius": [1.0, 1.0, 0.8787673501354397, ...],
      "length": 59.20872252575063,
      "min_radius": 0.8787673501354397,
      "max_radius": 3.818119098098647,
      "mean_radius": 1.704776715195705,
      "source": 0,
      "target": 534
    }
  ]
}
```

### Node Properties:
- pos: 3D coordinates [x, y, z] in voxel space
- radius: Vessel radius at this node location
- id: Unique node identifier

### Edge Properties:
- centerline: List of 3D coordinates along the vessel centerline
- radius: Corresponding radius values for each centerline point
- length: Total length of the vessel segment
- min_radius/max_radius/mean_radius: Radius statistics
- source/target: Connected node IDs
