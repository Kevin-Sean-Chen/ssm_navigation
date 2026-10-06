# ssm_navigation
ssm approach to navigation strategies, inference, and algorithm

This is relatively messy for test analysis. The formalized packages are currently in the https://git.yale.edu/emonetlab/ repo

## Environment & installation

This repository includes two environment manifests to reproduce the Python environment used for analysis:

- `environment.yml` — conda environment (recommended)
- `requirements.txt` — pip-installable requirements (alternative)

Recommended Python version: 3.10

### Using conda (recommended)

1. Create the conda environment:

```powershell
conda env create -f environment.yml
conda activate ssm_navigation
```

2. Run a quick smoke test (example):

```powershell
python -c "import numpy, matplotlib, sklearn; print('env OK')"
```

## Optogui database analysis

`gap_crossing/gap_cross_db.py` requires this environment and an editable optogui install from the adjacent repository:

```powershell
python -m pip install -e ..\optogui
$env:OPTOGUI_ROOT = (Resolve-Path ..\optogui).Path
```

Create the local `parameter_files/computer/settings.yaml` file that selects the computer profile. The profile defines the read-only server database and data paths.

Gap-crossing analysis has two stages. Each run writes a new timestamped folder under `gap_crossing/` in the Dropbox `projects/optogui` folder (set `GAP_CROSSING_OUTPUT_ROOT` to use another root). Runs never replace or delete files.

**1. Load datasets from the database.** Edit `GENOTYPE_FILES`, `QUERY_FILTERS`, `QUERY_PERIODS`, and the track filters in `gap_crossing/load_db.py`, then run:

```powershell
conda activate ssm_navigation
$env:OPTOGUI_ROOT = (Resolve-Path ..\optogui).Path
python .\gap_crossing\load_db.py
```

Each genotype gets one folder in `datasets/` with the processed tracks (`tracks.joblib`), recording and experiment tables, the query (`query.json`), track filters (`params.json`), provenance (git and optogui commits, package versions), a code snapshot with any uncommitted diff, and the console log. The script prints `DATASET_DIRS` lines to paste into analysis scripts.

**2. Analyze saved datasets.** Set `DATASET_DIRS` (label to dataset folder) at the top of an analysis script, then run it:

```powershell
python .\gap_crossing\gap_cross_db.py
python .\gap_crossing\gap_cross_db_vial.py
python -m gap_crossing.memory_analysis.gap_cross_db_memory
```

Each analysis writes a folder in `analyses/` with figures, tables, exports, all uppercase settings (`params.json`), and provenance that points to its datasets.

`gap_cross_db.py` and `gap_cross_db_vial.py` also accept `DATA_SOURCE = "database"`. That mode queries and loads with the settings in `load_db.py`, then plots, and saves nothing.

Activate the environment (`conda activate ssm_navigation`) before running. Calling the environment's `python.exe` without activation can load the wrong DLLs and crash Matplotlib. Datasets combine only when their query and track filters match apart from `genotype_file` and their recordings share one camera setup.
