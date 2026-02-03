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
