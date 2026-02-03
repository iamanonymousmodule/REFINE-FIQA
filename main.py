"""
REFINE-FIQA: REliable Feature INtErpretation for FIQA

This script:
1) Loads OFIQ/ISO quality attributes for AgeDB and trains a regression model to predict UnifiedQualityScore.
2) Optionally performs stability-driven feature selection.
3) Runs repeated CV to save RMSE, coefficients, CoSS, and a coefficient-scatter plot.
4) Predicts per-image quality and merges it into an embedding index.
5) Uses AgeDB verification protocol pairs + embeddings to compute EDC + pAUC@20%.

NOTE: Replace the PATH placeholders below with your actual paths (or export as env vars).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, gaussian_kde
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

# -----------------------------
# Matplotlib style
# -----------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"

# ============================================================
# CONFIG (edit these placeholders)
# ============================================================
DATASET = "AgeDB"
FRS = "arcface"  # arcface / facenet / magface

# You can either edit these placeholders directly, or set them via env vars.
AGEDB_CSV = os.getenv("AGEDB_CSV", "/path/to/quality_AgeDB_OFIQ.csv")
EMB_PATH = os.getenv("EMB_PATH", "/path/to/embeddings_AgeDB_arcface.npy")
EMB_INDEX_PATH = os.getenv("EMB_INDEX_PATH", "/path/to/AgeDB_arcface_index.csv")
AGEDB_PROTOCOL = os.getenv("AGEDB_PROTOCOL", "/path/to/pairs_AgeDB.txt")
OUT_DIR = os.getenv("OUT_DIR", "/path/to/outputs_FSEL_arcface/")

# EDC config
TARGET_FMR = 0.001
DISCARD_FRACS = np.linspace(0.0, 0.6, 13)

# Cross-validation config
RKF_SPLITS = 5
RKF_REPEATS = 30
RANDOM_STATE = 42


# ============================================================
# MODELS
# ============================================================
def make_model(model_key: str):
    if model_key == "OLS":
        return LinearRegression(fit_intercept=True)
    if model_key.startswith("Ridge_"):
        alpha = float(model_key.split("_", 1)[1])
        return Ridge(alpha=alpha, fit_intercept=True)
    raise ValueError(f"Unknown model_key: {model_key}")


# ============================================================
# FEATURE SELECTION CORE
# ============================================================
def calculate_entropy_dist(mu, sigma):
    p_negative = norm.cdf(0, mu, sigma)
    p_positive = 1 - p_negative

    mask = (p_positive > 0) & (p_positive < 1)
    entropy = np.zeros_like(p_positive)
    entropy[mask] = -(
        p_positive[mask] * np.log2(p_positive[mask])
        + p_negative[mask] * np.log2(p_negative[mask])
    )
    return entropy


def fsel_select_features(X_train, y_train, estimator, slack=0.0454, num_iter=10):
    """
    Returns:
      sel_idx : np.ndarray of selected feature indices
      mean_entropy : float
      iter_info : []
    """
    model = clone(estimator)
    train_mat_sel_idx = np.zeros(X_train.shape[1])
    sign_entropies_final = np.zeros(X_train.shape[1])

    for _iter in range(num_iter):
        zero_indices = np.where(train_mat_sel_idx == 0)[0]
        if len(zero_indices) == 0:
            break

        X_selected = X_train[:, zero_indices]
        model.fit(X_selected, y_train)

        beta_means = model.coef_
        beta_stds = np.sqrt(np.diag(model.sigma_))

        sign_entropies = np.array(
            [
                calculate_entropy_dist(beta_mean, beta_std)
                for beta_mean, beta_std in zip(beta_means, beta_stds)
            ]
        )

        non_zero_indices = np.where(sign_entropies > slack)[0]
        original_0_indices = np.where(train_mat_sel_idx == 0)[0]
        mapped_non0_indices = original_0_indices[non_zero_indices]

        if mapped_non0_indices.size != 0:
            train_mat_sel_idx[mapped_non0_indices] = 1
            sign_entropies_final = np.zeros(X_train.shape[1])
            sign_entropies_final[mapped_non0_indices] = sign_entropies[
                sign_entropies > slack
            ]
        else:
            break

    top_idx = np.array(np.where(train_mat_sel_idx == 0)[0])
    if len(top_idx) == 0:
        top_idx = np.arange(0, len(train_mat_sel_idx), step=1)

    return top_idx, float(np.mean(sign_entropies_final)), []


def remove_highly_correlated_features_pd(X, threshold):
    df = pd.DataFrame(X)
    corr_matrix = df.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix.mask(mask, inplace=True)
    cols_to_drop = [
        column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)
    ]
    reduced_df = df.drop(columns=cols_to_drop)
    X_reduced = reduced_df.to_numpy()
    retained_cols = reduced_df.columns.tolist()
    return X_reduced, cols_to_drop, retained_cols


# ============================================================
# DATA LOADING (ISO features)
# ============================================================
def get_fiqa_XY(filename):
    """
    Expects:
      - 'Filename' column
      - 'UnifiedQualityScore' column
      - ISO feature columns (numeric)
    """
    df_raw = pd.read_csv(filename, sep=";")
    df = df_raw.copy()

    if "Filename" in df.columns:
        filenames = df["Filename"].values
        df = df.drop(columns=["Filename"])
    else:
        filenames = np.array([f"row_{i}" for i in range(len(df))])

    # Drop scalar columns and known junk columns
    df = df.loc[:, ~df.columns.str.endswith(".scalar")]
    df = df.loc[:, ~df.columns.str.endswith("57")]

    if "UnifiedQualityScore" not in df.columns:
        raise RuntimeError("Expected column 'UnifiedQualityScore' not found in FIQA CSV.")

    y = df["UnifiedQualityScore"].values.astype(float)
    X = df.drop(columns=["UnifiedQualityScore"]).values.astype(float)

    # Impute NaNs
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Correlation pruning
    threshold = 0.80
    X, _, retained_feature_names = remove_highly_correlated_features_pd(X, threshold)

    # Min-max scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-12)

    print("[FIQA] X:", X.shape, " y:", y.shape)
    return X, y, retained_feature_names, filenames


# ============================================================
# CoSS utilities
# ============================================================
def _p_positive_kde(samples, grid_points=2048):
    samples = np.asarray(samples).ravel()
    if np.allclose(samples, 0):
        return 0.5
    try:
        kde = gaussian_kde(samples)
        lo = min(samples.min(), 0) - 5.0 * samples.std(
            ddof=1 if samples.size > 1 else 0
        )
        hi = max(samples.max(), 0) + 5.0 * samples.std(
            ddof=1 if samples.size > 1 else 0
        )
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -1.0, 1.0
        grid = np.linspace(lo, hi, grid_points)
        pdf = kde(grid)
        dx = (hi - lo) / (grid_points - 1)
        p_plus = pdf[grid >= 0].sum() * dx
        return float(np.clip(p_plus, 0.0, 1.0))
    except Exception:
        return float((samples > 0).mean())


def _sign_entropy_from_p(p_plus, eps=1e-12):
    p_plus = float(np.clip(p_plus, eps, 1 - eps))
    p_minus = 1.0 - p_plus
    return -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))


def compute_coss_from_coef_matrix(coef_matrix):
    coef_matrix = np.asarray(coef_matrix)
    n_models, n_features = coef_matrix.shape
    entropies = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        betas = coef_matrix[:, j]
        p_plus = _p_positive_kde(betas)
        entropies[j] = _sign_entropy_from_p(p_plus)
    return float(entropies.mean()), entropies


# ============================================================
# Plotting (coeff scatter)
# ============================================================
def scatter_coefficients_per_fold(name, coef_matrix, sel_features, fname_out, jitter=0.10):
    coef_matrix = np.asarray(coef_matrix)
    sel_features = np.asarray(sel_features)
    order = np.argsort(sel_features)
    x_sorted = sel_features[order]
    coef_sorted = coef_matrix[:, order]

    n_runs, n_feats = coef_sorted.shape
    x_positions = np.arange(n_feats)

    plt.figure(figsize=(max(10, n_feats * 0.20), 6))
    rng = np.random.RandomState(123)
    for i in range(n_runs):
        y_i = coef_sorted[i, :]
        colors = np.where(y_i >= 0, "blue", "red")
        jitter_i = rng.randn(n_feats) * jitter
        plt.scatter(
            x_positions + jitter_i,
            y_i,
            s=16,
            c=colors,
            alpha=0.65,
            edgecolors="none",
        )

    plt.axhline(0.0, color="k", linewidth=1, linestyle="--")
    plt.xlabel("Selected feature index (ascending)", fontsize=24)
    plt.ylabel("Coefficient value", fontsize=24)

    if n_feats <= 60:
        plt.xticks(x_positions, [str(i) for i in x_sorted], rotation=90, fontsize=20)
    else:
        step = max(1, n_feats // 60)
        plt.xticks(
            x_positions[::step],
            [str(i) for i in x_sorted[::step]],
            rotation=90,
            fontsize=20,
        )
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(fname_out, dpi=200)
    plt.close()


def take_or_all(X, sel_idx):
    if sel_idx is None or len(sel_idx) == 0:
        print("[WARN] sel_features is empty. Falling back to ALL features.")
        return X, np.arange(X.shape[1])
    return X[:, sel_idx], np.asarray(sel_idx)


# ============================================================
# RMSE + CoSS saving
# ============================================================
def run_repeated_cv_save_rmse_coefs_coss(
    X,
    y,
    model_key: str,
    case_id: str,
    out_dir: str,
    frs: str,
    feat_names: list,
    feat_indices_for_plot: np.ndarray,
    n_splits=5,
    n_repeats=30,
    random_state=42,
    jitter=0.10,
):
    os.makedirs(out_dir, exist_ok=True)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    rmse_rows = []
    rmses = []
    coefs_list = []

    for run_id, (tr, te) in enumerate(rkf.split(X), start=1):
        model = make_model(model_key)
        model.fit(X[tr], y[tr])
        yhat = model.predict(X[te])

        rmse = float(np.sqrt(mean_squared_error(y[te], yhat)))
        rmses.append(rmse)
        coefs_list.append(model.coef_.ravel().astype(float))

        rmse_rows.append(
            {
                "case_id": case_id,
                "model_key": model_key,
                "run_id": run_id,
                "rmse": rmse,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
            }
        )

    rmses = np.asarray(rmses, dtype=float)
    coef_matrix = np.vstack(coefs_list)

    # RMSE
    pd.DataFrame(rmse_rows).to_csv(
        os.path.join(out_dir, f"RMSE_runs_{case_id}_{frs}.csv"), index=False
    )
    pd.DataFrame(
        [
            {
                "case_id": case_id,
                "model_key": model_key,
                "n_runs": int(len(rmses)),
                "rmse_mean": float(rmses.mean()),
                "rmse_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
                "rmse_min": float(rmses.min()),
                "rmse_max": float(rmses.max()),
            }
        ]
    ).to_csv(os.path.join(out_dir, f"RMSE_summary_{case_id}_{frs}.csv"), index=False)

    # Coefs per run
    coef_df = pd.DataFrame(coef_matrix, columns=feat_names)
    coef_df.insert(0, "run_id", np.arange(1, coef_matrix.shape[0] + 1))
    coef_df.to_csv(os.path.join(out_dir, f"COEFS_runs_{case_id}_{frs}.csv"), index=False)

    # CoSS
    coss_val, per_feat_entropy = compute_coss_from_coef_matrix(coef_matrix)
    pd.DataFrame([{"case_id": case_id, "model_key": model_key, "coss": float(coss_val)}]).to_csv(
        os.path.join(out_dir, f"CoSS_summary_{case_id}_{frs}.csv"), index=False
    )
    pd.DataFrame(
        {
            "feature": feat_names,
            "feature_index": feat_indices_for_plot.astype(int),
            "sign_entropy": per_feat_entropy.astype(float),
        }
    ).to_csv(os.path.join(out_dir, f"CoSS_per_feature_{case_id}_{frs}.csv"), index=False)

    # Scatter
    scatter_out = os.path.join(out_dir, f"coeff_scatter_{case_id}_{frs}.pdf")
    scatter_coefficients_per_fold(
        f"{case_id} — coefficients across CV runs",
        coef_matrix,
        feat_indices_for_plot,
        scatter_out,
        jitter=jitter,
    )

    # Final fit + coef table
    final_model = make_model(model_key).fit(X, y)
    final_coef_df = pd.DataFrame(
        {
            "feature": feat_names,
            "coef": final_model.coef_.ravel().astype(float),
            "abs_coef": np.abs(final_model.coef_.ravel().astype(float)),
        }
    ).sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    final_coef_df.to_csv(os.path.join(out_dir, f"coeff_final_{case_id}_{frs}.csv"), index=False)

    return final_model, rmses, coef_matrix, coss_val


# ============================================================
# EDC
# ============================================================
def compute_fnmr_at_fmr(scores, labels, target_fmr):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    imp = scores[labels == 0]
    gen = scores[labels == 1]
    if len(imp) == 0 or len(gen) == 0:
        return np.nan, np.nan

    imp_sorted = np.sort(imp)[::-1]
    n_imp = len(imp_sorted)
    k = int(np.ceil(target_fmr * n_imp))

    if k <= 0:
        thresh = imp_sorted[0] + 1e-6
    elif k >= n_imp:
        thresh = imp_sorted[-1] - 1e-6
    else:
        thresh = imp_sorted[k - 1]

    fnmr = np.mean(gen <= thresh)
    return fnmr, thresh


def compute_edc(scores, labels, qualities, discard_fracs, target_fmr):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    qualities = np.asarray(qualities)

    N = len(scores)
    order = np.argsort(qualities)  # worst first

    fnmrs, thresholds = [], []
    for frac in discard_fracs:
        n_discard = int(np.floor(frac * N))
        if n_discard >= N:
            fnmrs.append(np.nan)
            thresholds.append(np.nan)
            continue
        keep = order[n_discard:]
        fnmr, thr = compute_fnmr_at_fmr(scores[keep], labels[keep], target_fmr)
        fnmrs.append(fnmr)
        thresholds.append(thr)

    return np.array(fnmrs), np.array(thresholds)


def plot_edc(fnmrs, discards, title, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(discards, fnmrs, marker="o", label=title)
    plt.xlabel("Discard Fraction")
    plt.ylabel("FNMR at FMR=0.1%")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVED] {out_path}")


# ============================================================
# Helpers: stem keys & protocol parsing
# ============================================================
def normalize_stem_key(path_or_name: str) -> str:
    """
    Converts a path/filename to a stem key (basename without extension).
    Add suffix-stripping rules here if needed.
    """
    s = os.path.basename(str(path_or_name))
    s = os.path.splitext(s)[0]

    # If embedding index uses aligned/cropped suffixes, uncomment + tune:
    # for suf in ["_aligned_112", "_aligned", "_crop", "_cropped", "_112", "_112x112"]:
    #     if s.endswith(suf):
    #         s = s[:-len(suf)]

    return s


def merge_quality_into_index_stem(df_idx, df_bay, quality_col: str):
    """
    Stem-based merge to avoid .jpg/.png and path mismatches.
    """
    if "Filename" not in df_bay.columns:
        raise RuntimeError("FIQA CSV must contain 'Filename' column.")

    # Ensure index has something filename-like
    if "basename" not in df_idx.columns:
        if "Filename" in df_idx.columns:
            df_idx = df_idx.copy()
            df_idx["basename"] = df_idx["Filename"].apply(lambda p: os.path.basename(str(p)))
        else:
            raise RuntimeError("Embedding index must contain 'basename' or 'Filename'.")

    df_bay = df_bay.copy()
    df_idx = df_idx.copy()

    df_bay["stem_key"] = df_bay["Filename"].apply(normalize_stem_key)
    df_idx["stem_key"] = df_idx["basename"].apply(normalize_stem_key)

    merged = pd.merge(df_idx, df_bay[["stem_key", quality_col]], on="stem_key", how="left")
    return merged


def iter_agedb_pairs(protocol_path):
    """
    AgeDB pairs file: each line "stem1,stem2"
    Stems typically match normalize_stem_key() output.
    """
    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = [t.strip() for t in line.split(",")]
            if len(toks) != 2:
                continue
            yield toks[0], toks[1]


def build_stem_to_info(df_idx_bq, quality_col):
    """
    Build mapping: stem -> {emb_idx, BayQ, subj_name}
    subj_name extraction is same as original AgeDB parsing:
      stem: 3718_RogerAllam_30_m  -> parts[1] = RogerAllam
    """
    if "embedding_index" not in df_idx_bq.columns:
        raise RuntimeError("Embedding index must contain 'embedding_index' column.")

    stem_col = "stem_key" if "stem_key" in df_idx_bq.columns else None

    stem_to_info = {}
    for _, row in df_idx_bq.iterrows():
        if pd.isna(row.get(quality_col, np.nan)):
            continue

        stem = (
            str(row[stem_col])
            if stem_col is not None
            else normalize_stem_key(row.get("basename", row.get("Filename", "")))
        )
        parts = stem.split("_")
        subj_name = parts[1] if len(parts) > 1 else stem

        stem_to_info[stem] = {
            "emb_idx": int(row["embedding_index"]),
            "BayQ": float(row[quality_col]),
            "subj_name": subj_name,
        }
    return stem_to_info


# ============================================================
# MAIN
# ============================================================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- keep the original behavior (hard-coded here) ----
    MODEL_KEY = "OLS"  # "OLS", "Ridge_0.1", "Ridge_0.5", "Ridge_0.9"
    MODE = "RAW"       # "RAW" or "FSEL"

    quality_col = f"PredQuality_RAW_{MODEL_KEY}" if MODE == "RAW" else f"FSEL_Quality_{MODEL_KEY}"
    case_id_safe = f"{MODE}_{MODEL_KEY}".replace(".", "_")

    agedb_bayq_csv = out_dir / f"AgeDB_with_{case_id_safe}_{FRS}.csv"
    emb_index_bayq_csv = out_dir / f"AgeDB_index_with_{case_id_safe}_{FRS}.csv"

    # --------------------------------------------------------
    # 1) Load FIQA ISO features
    # --------------------------------------------------------
    X, y, retained_feature_names, _ = get_fiqa_XY(AGEDB_CSV)

    # --------------------------------------------------------
    # 2) Feature mode (RAW vs FSEL)
    # --------------------------------------------------------
    if MODE == "RAW":
        X_used = X
        feat_indices_for_plot = np.arange(X.shape[1])
    elif MODE == "FSEL":
        selector = BayesianRidge(lambda_init=1e-10, fit_intercept=True, compute_score=True)
        sel_features, _, _ = fsel_select_features(X, y, estimator=selector, slack=1e-6)
        X_used, _ = take_or_all(X, sel_features)
        feat_indices_for_plot = np.asarray(sel_features, dtype=int)
    else:
        raise ValueError("MODE must be 'RAW' or 'FSEL'")

    # Feature names for logs
    if retained_feature_names is not None and len(retained_feature_names) >= X.shape[1]:
        feat_names = [retained_feature_names[i] for i in feat_indices_for_plot]
    else:
        feat_names = [f"feat_{i}" for i in feat_indices_for_plot]

    # --------------------------------------------------------
    # 3) Repeated-CV RMSE + CoSS + coef scatter
    # --------------------------------------------------------
    trained_model, rmses, _, coss_val = run_repeated_cv_save_rmse_coefs_coss(
        X_used,
        y,
        model_key=MODEL_KEY,
        case_id=case_id_safe,
        out_dir=str(out_dir),
        frs=FRS,
        feat_names=feat_names,
        feat_indices_for_plot=feat_indices_for_plot,
        n_splits=RKF_SPLITS,
        n_repeats=RKF_REPEATS,
        random_state=RANDOM_STATE,
        jitter=0.10,
    )
    print(
        f"[FINAL] {case_id_safe} RMSE mean={rmses.mean():.4f}, std={rmses.std(ddof=1):.4f}, CoSS={coss_val:.6f}"
    )

    # --------------------------------------------------------
    # 4) Predict quality per image + save dataset CSV
    # --------------------------------------------------------
    bayq = trained_model.predict(X_used)
    df_raw = pd.read_csv(AGEDB_CSV, sep=";")
    if "Filename" not in df_raw.columns:
        raise RuntimeError("AGEDB_CSV must contain a 'Filename' column.")
    df_raw[quality_col] = bayq
    df_raw.to_csv(agedb_bayq_csv, sep=";", index=False)
    print(f"[INFO] Saved AgeDB with {quality_col} to: {agedb_bayq_csv}")

    # --------------------------------------------------------
    # 5) Merge quality into embedding index (STEM-based)
    # --------------------------------------------------------
    df_idx = pd.read_csv(EMB_INDEX_PATH)
    df_bay = pd.read_csv(agedb_bayq_csv, sep=";")

    df_idx_merged = merge_quality_into_index_stem(df_idx, df_bay, quality_col)
    df_idx_merged.to_csv(emb_index_bayq_csv, index=False)

    missing_q = int(df_idx_merged[quality_col].isna().sum())
    print(f"[INFO] Saved BayQ-augmented embedding index to: {emb_index_bayq_csv}")
    print(f"[INFO] Missing {quality_col} after stem-merge: {missing_q} / {len(df_idx_merged)}")

    # --------------------------------------------------------
    # 6) Build scores/labels/qualities using AgeDB protocol
    # --------------------------------------------------------
    embeddings = np.load(EMB_PATH)
    print("[DEBUG] embeddings shape:", embeddings.shape)
    print("[DEBUG] index rows:", len(df_idx_merged))

    df_idx_bq = pd.read_csv(emb_index_bayq_csv)
    stem_to_info = build_stem_to_info(df_idx_bq, quality_col)
    print("[INFO] Unique stems with BayQ in index:", len(stem_to_info))

    pair_scores, pair_labels, pair_qualities = [], [], []
    n_total = n_missing = n_used = 0

    for s1_raw, s2_raw in iter_agedb_pairs(AGEDB_PROTOCOL):
        n_total += 1
        s1 = normalize_stem_key(s1_raw)
        s2 = normalize_stem_key(s2_raw)

        if s1 not in stem_to_info or s2 not in stem_to_info:
            n_missing += 1
            continue

        i1 = stem_to_info[s1]["emb_idx"]
        i2 = stem_to_info[s2]["emb_idx"]

        v1 = embeddings[i1] / (np.linalg.norm(embeddings[i1]) + 1e-12)
        v2 = embeddings[i2] / (np.linalg.norm(embeddings[i2]) + 1e-12)
        score = float(np.dot(v1, v2))

        # Label from subject name parsed out of the stem
        lab = 1 if stem_to_info[s1]["subj_name"] == stem_to_info[s2]["subj_name"] else 0

        q_pair = min(stem_to_info[s1]["BayQ"], stem_to_info[s2]["BayQ"])

        pair_scores.append(score)
        pair_labels.append(int(lab))
        pair_qualities.append(float(q_pair))
        n_used += 1

    pair_scores = np.asarray(pair_scores, dtype=float)
    pair_labels = np.asarray(pair_labels, dtype=int)
    pair_qualities = np.asarray(pair_qualities, dtype=float)

    print("[INFO] Total protocol pairs:", n_total)
    print("[INFO] Pairs missing stems :", n_missing)
    print("[INFO] Pairs used          :", n_used)
    print("[INFO] Genuine pairs       :", int(np.sum(pair_labels == 1)))
    print("[INFO] Impostor pairs      :", int(np.sum(pair_labels == 0)))

    if n_used == 0:
        raise RuntimeError(
            "No pairs were used. This means the protocol stem keys do not match the embedding index stem keys.\n"
            "Edit normalize_stem_key() to strip suffixes used in the embedding index (aligned/cropped/etc.)."
        )

    # --------------------------------------------------------
    # 7) EDC + pAUC@20%
    # --------------------------------------------------------
    fnmrs, _ = compute_edc(pair_scores, pair_labels, pair_qualities, DISCARD_FRACS, TARGET_FMR)

    max_discard = 0.20
    valid = DISCARD_FRACS <= max_discard
    pAUC = np.trapz(fnmrs[valid], DISCARD_FRACS[valid]) / max_discard

    print("[INFO] Discard fractions:", DISCARD_FRACS)
    print("[INFO] FNMR (BayQ)       :", fnmrs)
    print(f"[INFO] pAUC@20% (BayQ): {pAUC:.6f}")

    out_path = out_dir / f"AgeDB_EDC_{case_id_safe}_{FRS}.png"
    plot_edc(fnmrs, DISCARD_FRACS, f"{DATASET} {MODE} {MODEL_KEY} ({FRS})", str(out_path))


if __name__ == "__main__":
    main()
