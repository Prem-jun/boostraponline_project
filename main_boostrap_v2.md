# main_boostrap_v2.py — Documentation

## Overview

`main_boostrap_v2.py` is the **main entry point** for running Online Bootstrap experiments in version 2 of the project. It reads YAML configuration files and JSON data chunks, then executes both **online** and **offline (traditional)** bootstrap methods to compare their statistical performance.

This script replaces the older monolithic `main_btonline2.py` by leveraging the modular `online_bootstrap` package.

---

## Features

| Feature | Description |
|---|---|
| **YAML-based configuration** | Reads multi-document YAML files to define experiment parameters |
| **Online Bootstrap** | Processes data in streaming chunks using `boot_stream.booststream()` |
| **Online Bootstrap with Min-Max** | Online bootstrap variant that applies min-max normalization |
| **Traditional (Offline) Bootstrap** | Accumulates all data and performs standard bootstrap as a baseline |
| **Outlier Detection** | Optional batch outlier detection via `BatchOutlierDetection.ZBatchOutlierDetector` |
| **CLI Support** | Full `argparse` interface for specifying config directory, file, and outlier mode |

---

## Dependencies

### Python Standard Libraries
- `json`, `pickle`, `os`, `argparse`, `yaml`
- `typing` (List, Union, Dict)
- `dataclasses` (dataclass, field)
- `pathlib` (Path)

### External Libraries
- `pandas`
- `numpy`

### Internal Modules (from `online_bootstrap/`)

| Module | Purpose |
|---|---|
| `boot_stream` | Core streaming bootstrap engine (`booststream` class) |
| `res_bootstrap` | Result collection and storage (`Res_boostrap` class) |
| `BatchOutlierDetection` | Z-score based batch outlier detector (`ZBatchOutlierDetector`) |
| `stat_dist` | Statistical distribution utilities |
| `bootstrap_v1` | Legacy bootstrap functions |
| `bootstrap_online` | Online bootstrap process manager |
| `samp1d` | 1D sampling utilities |

---

## Usage

### Basic Command

```bash
python main_boostrap_v2.py --dir config_sim_data/fdist --file config_fdist_simulate.yaml
```

### With Outlier Detection

```bash
python main_boostrap_v2.py --dir config_sim_data/fdist --file config_fdist_simulate.yaml --outlier
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dir` | `str` | `config_sim_data/fdist` | Working directory containing config and data files |
| `--file` | `str` | `config_fdist_simulate.yaml` | YAML configuration filename |
| `--outlier` | flag | `False` | Enable batch outlier detection during online bootstrap |

---

## Function Reference

### `read_json_file(file_path) → dict`

Reads a JSON file and returns its contents as a Python object.

- **Input**: Path to a `.json` file
- **Output**: Parsed JSON data (typically a list of chunk dictionaries)

---

### `parse_opt() → argparse.Namespace`

Parses command-line arguments using `argparse`.

- **Returns**: Namespace with `dir`, `file`, and `outlier` attributes

---

### `read_yaml_config(file_path) → list[dict]`

Reads a multi-document YAML configuration file (separated by `---`).

- **Input**: Path to a `.yaml` file
- **Output**: A list of configuration dictionaries, one per YAML document

---

### `run(dir, file, outlier)`

**Core execution function.** Orchestrates the entire bootstrap experiment pipeline:

#### Workflow

```
┌─────────────────────────────────────────────────────┐
│  1. Read YAML config (may contain multiple configs) │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼───────────────┐
          │  2. For each config:       │
          │     Load JSON data chunks  │
          └────────────┬───────────────┘
                       │
          ┌────────────▼───────────────────────────────┐
          │  3. For each dataset in JSON:              │
          │     Initialize 3 bootstrap networks:       │
          │       • net_online      (online)            │
          │       • net_online_mm   (online + min-max)  │
          │       • net_trad_cum    (traditional)       │
          └────────────┬───────────────────────────────┘
                       │
          ┌────────────▼───────────────────────────────┐
          │  4. For each chunk of samples:             │
          │     • Online:      expand_bt_online(chunk) │
          │     • Online MM:   expand_bt_online(chunk) │
          │     • Traditional: expand_bt_trad(whole)   │
          │     • Collect results via Res_boostrap     │
          └────────────┬───────────────────────────────┘
                       │
          ┌────────────▼───────────────────────────────┐
          │  5. Save all results to .pkl file          │
          │     • Normal:  {name}_re.pkl               │
          │     • Outlier: {name}_re_outlier.pkl       │
          └────────────────────────────────────────────┘
```

#### Bootstrap Methods Compared

| Method | Variable | Function | Description |
|---|---|---|---|
| Online | `net_online` | `expand_bt_online(chunk)` | Processes each chunk incrementally |
| Online + Min-Max | `net_online_mm` | `expand_bt_online(chunk)` | Online with min-max normalization |
| Traditional (Offline) | `net_trad_cum` | `expand_bt_trad(whole)` | Re-bootstraps entire accumulated dataset each round |

---

### `main(opt)`

Wrapper that unpacks parsed arguments and calls `run()`.

---

## Input / Output

### Input Files

| File Type | Format | Description |
|---|---|---|
| YAML config | `.yaml` | Experiment parameters; must contain `file_data_chunk` key |
| JSON data | `.json` | Chunked sample data with `samp_chuck` and `chunk_size` fields |

### Output Files

| Condition | Output File |
|---|---|
| Normal mode | `{file_data_chunk}_re.pkl` |
| Outlier mode (`--outlier`) | `{file_data_chunk}_re_outlier.pkl` |

The `.pkl` file contains a dictionary:
```python
{
    'result_all': [Res_boostrap, Res_boostrap, ...]  # list of result objects
}
```

---

## Project Structure Context

```
boostraponline_project/
├── main_boostrap_v2.py          ← This script (entry point)
├── online_bootstrap/            ← Core library package
│   ├── __init__.py
│   ├── boot_stream.py           ← booststream class (online/offline methods)
│   ├── res_bootstrap.py         ← Res_boostrap result collector
│   ├── BatchOutlierDetection.py ← ZBatchOutlierDetector
│   ├── bootstrap_v1.py          ← Legacy bootstrap functions
│   ├── bootstrap_online.py      ← Online bootstrap process manager
│   ├── stat_dist.py             ← Statistical distribution calculations
│   └── samp1d.py                ← 1D sampling utilities
├── config_sim_data/             ← YAML configuration files
└── sim_data/                    ← Simulated JSON data files
```

---

## Version History (Git)

| Date | Commit | Description |
|---|---|---|
| 2025-05-21 | `af1c068` | Add batch outlier detection functionality |
| 2025-02-19 | `6ed6d96` | General updates |
| 2025-01-29 | `f8eb02f` | Update distribution support |
| 2025-01-28 | `d1e2623` | Major refactoring (reduced 50 lines) |
| 2025-01-28 | `7f0bcbc` | Initial creation (177 lines) |
