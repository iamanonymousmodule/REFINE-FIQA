# REFINE-FIQA
REFINE-FIQA: Stability-driven feature selection for reliable and consistent FIQA explanations using ISO/IEC 29794-5 quality attributes

This repository provides a reproducible pipeline to:

1. Load **ISO/IEC 29794-5** face quality attributes (e.g., from **OFIQ**) for **AgeDB**
2. Train a regression model to predict **UnifiedQualityScore** from ISO attributes
3. Measure coefficient sign-stability across repeated CV using **CoSS** (sign entropy)
4. Optionally apply a **stability-driven feature selection** procedure
5. Predict per-image quality, merge it into an embedding index using **stem-based matching**, and evaluate verification performance using:
   - **EDC** (Error-vs-Discard Curve): FNMR as a function of discarded low-quality samples
   - **pAUC@20%** computed from the EDC curve at a target FMR

> **MODEL_KEY** is used to select the regression approach (OLS, Ridge_0.1, Ridge_0.5, Ridge_0.9) and **MODEL** is used to select full features/ feature selection.
> The core script does **not** compute embeddings; it assumes you already have embeddings and an index file that maps embeddings to filenames.

## EDC + pAUC + Bootstrap CI (Verification Evaluation)

In addition to the quality-prediction pipeline (`main.py`), this repository includes a second script that **evaluates verification performance using EDC curves and paired bootstrap confidence intervals**.

### What this script is for
After you run `main.py`, it produces (per method/case) an embedding index file like:

- `AgeDB_index_with_<CASE_ID>_<FRS>.csv`

These files contain the **predicted quality scores** (e.g., `PredQuality_RAW_OLS`, `FSEL_Quality_OLS`, `FSEL_Quality_Ridge_0.1`, etc.) aligned to each image in the embedding index.  
The evaluation script uses those per-case CSVs to:

- Compute **repo-style EDC (FNMR vs discard fraction)** based on predicted quality
- Compute **pAUC** at discard limits (e.g., 10%–60%)
- Run **paired bootstrap** to report the CI of **(baseline pAUC − method pAUC)**

### Script
- `EDC_Plot.py`

### Expected inputs
This script reuses the same embeddings and protocol files, and additionally requires the directory containing the per-case index CSVs:

- `EMB_PATH` (e.g., embeddings `.npy`)
- `EMB_INDEX_PATH` (base embedding index `.csv` with `basename` and `embedding_index`)
- `AGEDB_PROTOCOL` (pairs file: `stem1,stem2`)
- `INDEX_DIR` (directory containing `AgeDB_index_with_<CASE_ID>_<FRS>.csv` files)

### Running (recommended via env vars)
```bash
export EMB_PATH="/path/to/embeddings_AgeDB_arcface.npy"
export EMB_INDEX_PATH="/path/to/AgeDB_arcface_index.csv"
export AGEDB_PROTOCOL="/path/to/pairs_AgeDB.txt"
export INDEX_DIR="./outputs"     # same OUT_DIR used in main.py
export FRS="arcface"

python EDC_Plot.py
