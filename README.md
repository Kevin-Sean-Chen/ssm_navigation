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
