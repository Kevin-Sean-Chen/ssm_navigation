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

Run the database-backed gap analysis from this repository:

```powershell
conda activate ssm_navigation
$env:OPTOGUI_ROOT = (Resolve-Path ..\optogui).Path
python .\gap_crossing\gap_cross_db.py
```

For day-vial motif uncertainty, run `python .\gap_crossing\gap_cross_db_vial.py`.

Edit `QUERY_FILTERS` in `gap_crossing/gap_cross_db.py` before each analysis. The current filters match the July 2026 gap-ribbon query in the Dropbox analysis example.
