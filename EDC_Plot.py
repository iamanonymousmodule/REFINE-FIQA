"""
EDC + pAUC + Paired Bootstrap CI
------------------------------------------

This script:
1) Loads embeddings + an embedding index (basename -> embedding_index)
2) Builds cosine similarity scores for AgeDB protocol pairs
3) Loads predicted quality scores (per method) from "AgeDB_index_with_*" CSVs
4) Computes repo-style EDC curves (FNMR vs discard fraction)
5) Computes pAUC at multiple discard limits
6) Runs paired bootstrap CI on (baseline pAUC - method pAUC)
7) Saves combined EDC plot + optional per-case plots + CSV summary

Configuration:
- Prefer environment variables for paths (see CONFIG section).
- Replace placeholders if not using env vars.

IMPORTANT:
- Plot trimming is controlled by PLOT_DISCARD_MAX (default 0.30).
  Even if EDC is computed up to discard=1.0 internally, the plot is trimmed.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Matplotlib style
# -----------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"


# ============================================================
# CONFIG (use env vars or edit placeholders)
# ============================================================
EMB_PATH = os.getenv("EMB_PATH", "/path/to/embeddings_AgeDB_arcface.npy")
EMB_INDEX_PATH = os.getenv("EMB_INDEX_PATH", "/path/to/AgeDB_arcface_index.csv")
AGEDB_PROTOCOL = os.getenv("AGEDB_PROTOCOL", "/path/to/pairs_AgeDB.txt")

# Directory containing the per-case CSVs:
# AgeDB_index_with_<CASE_ID>_<FRS>.csv (or similar)
INDEX_DIR = os.getenv("INDEX_DIR", "/path/to/outputs_dir_with_index_with_csvs/")
FRS = "arcface" #facenet, magface

# Baseline used for bootstrap diff
BASELINE_CASE_ID = "RAW_OLS"

# Methods to show in combined plot (baseline RAW_OLS + FSEL variants)
CASE_CANDIDATES = [
    "RAW_OLS",
    "FSEL_OLS",
    "FSEL_Ridge_0_1",
    "FSEL_Ridge_0_5",
    "FSEL_Ridge_0_9",
]

#EDC settings
STARTING_ERROR = 0.05
DISCARD_LIMITS = np.arange(0.1, 0.7, 0.1)  # [0.1, 0.2, ..., 0.6]
N_BOOT = 500
SEED = 42

# Plot trimming (x-axis max).
PLOT_DISCARD_MAX = 0.30

# pAUC values that can be printed in legend text (currently not used in legend label)
LEGEND_PAUCS = [0.2, 0.3, 0.4]
LEGEND_PAUCS_DECIMALS = 4

PLOT_OUT = os.path.join(INDEX_DIR, f"EDC_combined_AgeDB_{FRS}.png")
CSV_OUT = os.path.join(INDEX_DIR, f"EDC_summary_AgeDB_{FRS}.csv")

SAVE_PER_CASE_PLOTS = True


# ============================================================
# EDC implementation
# ============================================================
QualityScore = float
SimilarityScore = float
SimilarityScores = Union[SimilarityScore, Iterable[SimilarityScore]]


class EdcSample(TypedDict):
    quality_score: QualityScore


class EdcSamplePair(TypedDict):
    samples: tuple[EdcSample, EdcSample]
    similarity_score: SimilarityScore


class EdcErrorType(Enum):
    FNMR = "FNMR"
    FMR = "FMR"


PairQualityScoreFunction = Callable[[QualityScore, QualityScore], QualityScore]


class EdcOutput(TypedDict):
    error_type: EdcErrorType
    error_fractions: np.ndarray
    discard_fractions: np.ndarray
    error_counts: np.ndarray
    discard_counts: np.ndarray
    comparison_count: int
    similarity_score_threshold: float


def _form_error_comparison_decision(
    error_type: EdcErrorType,
    similarity_score_or_scores: SimilarityScores,
    similarity_score_threshold: SimilarityScore,
    out: Optional[np.ndarray] = None,
):
    if error_type == EdcErrorType.FNMR:
        return np.less(similarity_score_or_scores, similarity_score_threshold, out=out)
    if error_type == EdcErrorType.FMR:
        return np.greater_equal(similarity_score_or_scores, similarity_score_threshold, out=out)
    raise ValueError(f"Unknown error_type={error_type}")


def compute_edc_repo_style(
    error_type: EdcErrorType,
    sample_pairs: List[EdcSamplePair],
    starting_error: float = 0.05,
    pair_quality_score_function: PairQualityScoreFunction = min,
) -> EdcOutput:
    similarity_scores = np.zeros(len(sample_pairs), dtype=np.float64)
    pair_quality_scores = np.zeros(len(sample_pairs), dtype=np.float64)

    for i, sp in enumerate(sample_pairs):
        similarity_scores[i] = float(sp["similarity_score"])
        s1, s2 = sp["samples"]
        pair_quality_scores[i] = float(pair_quality_score_function(s1["quality_score"], s2["quality_score"]))

    order = np.argsort(pair_quality_scores)
    pair_quality_scores = pair_quality_scores[order]
    similarity_scores = similarity_scores[order]

    q = starting_error
    if error_type == EdcErrorType.FMR:
        q = 1.0 - q
    thr = float(np.quantile(similarity_scores, q))

    comparison_count = len(pair_quality_scores)
    error_counts = np.zeros(comparison_count, dtype=np.uint32)
    _form_error_comparison_decision(error_type, similarity_scores, thr, out=error_counts)
    error_counts = np.flipud(np.cumsum(np.flipud(error_counts), out=error_counts))

    if comparison_count <= 1:
        discard_counts = np.array([0], dtype=np.int64)
    else:
        discard_counts = np.where(pair_quality_scores[:-1] != pair_quality_scores[1:])[0] + 1
        discard_counts = np.concatenate(([0], discard_counts))

    remaining_counts = comparison_count - discard_counts
    error_fractions = error_counts[discard_counts] / remaining_counts
    discard_fractions = discard_counts / comparison_count

    if discard_fractions[-1] != 1.0:
        discard_fractions = np.concatenate((discard_fractions, [1.0]))
        error_fractions = np.concatenate((error_fractions, [0.0]))

    return EdcOutput(
        error_type=error_type,
        error_fractions=error_fractions,
        discard_fractions=discard_fractions,
        error_counts=error_counts,
        discard_counts=discard_counts,
        comparison_count=int(comparison_count),
        similarity_score_threshold=float(thr),
    )


def compute_edc_pauc_repo_style(edc_output: EdcOutput, discard_fraction_limit: float) -> float:
    ef = edc_output["error_fractions"]
    df = edc_output["discard_fractions"]
    assert 0.0 <= discard_fraction_limit <= 1.0
    if discard_fraction_limit == 0:
        return 0.0

    pauc = 0.0
    for i in range(len(df)):
        if i == len(df) - 1 or df[i + 1] >= discard_fraction_limit:
            pauc += float(ef[i]) * float(discard_fraction_limit - df[i])
            break
        pauc += float(ef[i]) * float(df[i + 1] - df[i])
    return float(pauc)


def bootstrap_pauc_difference(
    sample_pairs_baseline: List[EdcSamplePair],
    sample_pairs_case: List[EdcSamplePair],
    error_type: EdcErrorType,
    starting_error: float,
    discard_limit: float,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, float, float, float]:
    rng = np.random.default_rng(seed)
    N = len(sample_pairs_baseline)
    assert N == len(sample_pairs_case), "Both must have same number of pairs"

    diffs = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        boot_base = [sample_pairs_baseline[i] for i in idx]
        boot_case = [sample_pairs_case[i] for i in idx]

        edc_base = compute_edc_repo_style(error_type, boot_base, starting_error=starting_error, pair_quality_score_function=min)
        edc_case = compute_edc_repo_style(error_type, boot_case, starting_error=starting_error, pair_quality_score_function=min)

        pauc_base = compute_edc_pauc_repo_style(edc_base, discard_limit)
        pauc_case = compute_edc_pauc_repo_style(edc_case, discard_limit)

        diffs[b] = pauc_base - pauc_case

    mean_diff = float(diffs.mean())
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5]).astype(float)
    return diffs, mean_diff, float(ci_low), float(ci_high)


# ============================================================
# load pairs + load per-case quality maps
# ============================================================
def _basename_to_stem(p: str) -> str:
    return os.path.splitext(os.path.basename(str(p)))[0]


def build_stem_to_embidx(df_idx: pd.DataFrame) -> Dict[str, int]:
    if "basename" not in df_idx.columns or "embedding_index" not in df_idx.columns:
        raise RuntimeError(f"Index CSV must contain basename + embedding_index. Found: {df_idx.columns.tolist()}")
    m: Dict[str, int] = {}
    for _, r in df_idx.iterrows():
        stem = _basename_to_stem(r["basename"])
        m[stem] = int(r["embedding_index"])
    return m


def load_case_index_with_csv(index_dir: str, case_id: str, frs: str) -> Tuple[str, pd.DataFrame]:
    case_id_safe = case_id.replace(".", "_")
    candidates: List[str] = []
    for fn in os.listdir(index_dir):
        if not fn.endswith(".csv"):
            continue
        if not fn.startswith("AgeDB_index_with_"):
            continue
        if f"_{case_id_safe}_" not in fn:
            continue
        if not fn.endswith(f"_{frs}.csv"):
            continue
        candidates.append(fn)

    if len(candidates) == 0:
        raise RuntimeError(f"No index_with CSV found for case_id={case_id} (safe={case_id_safe}) in {index_dir}")

    candidates.sort(key=len, reverse=True)
    path = os.path.join(index_dir, candidates[0])
    return path, pd.read_csv(path)


def infer_quality_col_from_case(case_id: str, cols: List[str]) -> str:
    cols_set = set(cols)

    if case_id.startswith("RAW_"):
        preferred = f"PredQuality_{case_id}"
        if preferred in cols_set:
            return preferred
        for c in cols:
            if c.startswith("PredQuality_RAW_"):
                return c
        if "FSEL_Quality_OLS" in cols_set:
            return "FSEL_Quality_OLS"
        raise RuntimeError(f"Cannot infer RAW quality col for {case_id}. Available: {cols}")

    if not case_id.startswith("FSEL_"):
        raise RuntimeError(f"Expected case_id like RAW_* or FSEL_*, got {case_id}")

    model_part = case_id.split("FSEL_", 1)[1]  # OLS or Ridge_0_1

    if model_part == "OLS":
        qcol = "FSEL_Quality_OLS"
        if qcol in cols_set:
            return qcol
        for c in cols:
            if ("FSEL_Quality" in c) and c.endswith("OLS"):
                return c
        raise RuntimeError(f"Expected '{qcol}' for case {case_id}. Available: {cols}")

    if model_part.startswith("Ridge_"):
        alpha_str = model_part.split("_", 1)[1]  # 0_1
        alpha_dot = alpha_str.replace("_", ".")  # 0.1
        candidates = [
            f"FSEL_Quality_Ridge_{alpha_dot}",
            f"FSEL_Quality_Ridge_{alpha_str}",
            f"FSEL_Quality_{model_part}",
        ]
        for qcol in candidates:
            if qcol in cols_set:
                return qcol
        for c in cols:
            if "FSEL_Quality_Ridge" in c and (alpha_dot in c or alpha_str in c):
                return c
        raise RuntimeError(f"Cannot infer Ridge quality col for {case_id}. Tried {candidates}. Available: {cols}")

    raise RuntimeError(f"Unsupported FSEL model_part={model_part} in case_id={case_id}")


def stem_quality_map_from_case_index(df_case: pd.DataFrame, case_id: str) -> Tuple[Dict[str, float], str]:
    cols = df_case.columns.tolist()

    if "basename_clean" in cols:
        bcol = "basename_clean"
    elif "basename_x" in cols:
        bcol = "basename_x"
    elif "basename" in cols:
        bcol = "basename"
    elif "Filename" in cols:
        bcol = "Filename"
    else:
        raise RuntimeError(f"No basename column in case CSV. Columns: {cols}")

    qcol = infer_quality_col_from_case(case_id, cols)

    m: Dict[str, float] = {}
    for _, r in df_case.iterrows():
        q = r[qcol]
        if pd.isna(q):
            continue
        stem = _basename_to_stem(r[bcol])
        m[stem] = float(q)
    return m, qcol


def build_pairs_similarity_and_labels(
    embeddings: np.ndarray,
    stem_to_embidx: Dict[str, int],
    protocol_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_scores: List[float] = []
    pair_labels: List[int] = []
    s1_list: List[str] = []
    s2_list: List[str] = []
    n_total = 0
    n_missing = 0

    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," not in line:
                continue
            a, b = [x.strip() for x in line.split(",", 1)]
            n_total += 1

            if a not in stem_to_embidx or b not in stem_to_embidx:
                n_missing += 1
                continue

            i1 = stem_to_embidx[a]
            i2 = stem_to_embidx[b]

            v1 = embeddings[i1]
            v2 = embeddings[i2]
            v1 = v1 / (np.linalg.norm(v1) + 1e-12)
            v2 = v2 / (np.linalg.norm(v2) + 1e-12)
            score = float(np.dot(v1, v2))

            p1 = a.split("_")
            p2 = b.split("_")
            subj1 = p1[1] if len(p1) > 1 else a
            subj2 = p2[1] if len(p2) > 1 else b
            label = 1 if subj1 == subj2 else 0

            pair_scores.append(score)
            pair_labels.append(label)
            s1_list.append(a)
            s2_list.append(b)

    scores = np.asarray(pair_scores, dtype=np.float64)
    labels = np.asarray(pair_labels, dtype=np.int32)
    s1 = np.asarray(s1_list, dtype=object)
    s2 = np.asarray(s2_list, dtype=object)

    print(
        f"[INFO] protocol lines={n_total} missing={n_missing} used={len(scores)} | "
        f"genuine={int((labels==1).sum())} impostor={int((labels==0).sum())}"
    )
    return scores, labels, s1, s2


def pair_availability_mask(s1: np.ndarray, s2: np.ndarray, stem_to_quality: Dict[str, float]) -> np.ndarray:
    ok = np.zeros(len(s1), dtype=bool)
    for i, (a, b) in enumerate(zip(s1, s2)):
        ok[i] = (a in stem_to_quality) and (b in stem_to_quality)
    return ok


def subset_sample_pairs(
    scores: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    idx_keep: np.ndarray,
    stem_to_quality: Dict[str, float],
) -> List[EdcSamplePair]:
    out: List[EdcSamplePair] = []
    for i in idx_keep:
        a = s1[i]
        b = s2[i]
        out.append(
            EdcSamplePair(
                samples=(
                    EdcSample(quality_score=float(stem_to_quality[a])),
                    EdcSample(quality_score=float(stem_to_quality[b])),
                ),
                similarity_score=float(scores[i]),
            )
        )
    return out


# ============================================================
# Plotting (trim to discard_max)
# ============================================================
def _trim_edc_to_discard_max(edc: EdcOutput, discard_max: float) -> EdcOutput:
    df = edc["discard_fractions"]
    ef = edc["error_fractions"]
    keep = df <= discard_max

    # Include a point exactly at discard_max (stepwise hold)
    if discard_max not in df and discard_max < 1.0:
        last_idx = np.where(keep)[0]
        if len(last_idx) == 0:
            df2 = np.array([discard_max], dtype=float)
            ef2 = np.array([ef[0]], dtype=float)
        else:
            j = last_idx[-1]
            df2 = np.concatenate([df[keep], [discard_max]])
            ef2 = np.concatenate([ef[keep], [ef[j]]])
    else:
        df2 = df[keep]
        ef2 = ef[keep]

    edc2 = dict(edc)
    edc2["discard_fractions"] = df2
    edc2["error_fractions"] = ef2
    return edc2  # type: ignore


def short_method_name(method: str) -> str:
    if method.startswith("RAW_"):
        return method.replace("RAW_", "R_")
    if method == "FSEL_OLS":
        return "F_OLS"
    if method.startswith("FSEL_Ridge_"):
        alpha = method.replace("FSEL_Ridge_", "").replace("_", ".")
        return f"F_R{alpha}"
    return method


def plot_edc_curves(curves: Dict[str, EdcOutput], out_path: str, title: str, discard_max: float):
    plt.figure(figsize=(6, 6))

    for method, edc in curves.items():
        edc_t = _trim_edc_to_discard_max(edc, discard_max)

        # compact method label
        short_name = short_method_name(method)

        # (kept for parity with code; legend currently uses only short_name)
        _ = [
            compute_edc_pauc_repo_style(edc, d) for d in LEGEND_PAUCS
        ]

        label = f"{short_name}"

        plt.plot(
            edc_t["discard_fractions"],
            edc_t["error_fractions"],
            linewidth=1.5,
            label=label,
        )

    plt.xlim(0.0, discard_max)
    plt.ylim(0.0, 0.06)
    plt.xlabel("Discard fraction", fontsize=20)
    plt.ylabel("FNMR", fontsize=20)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load embeddings + base mapping
    embeddings = np.load(EMB_PATH)
    df_base = pd.read_csv(EMB_INDEX_PATH)
    stem_to_embidx = build_stem_to_embidx(df_base)

    # Build shared pair list (scores + stems + labels)
    scores, labels, s1, s2 = build_pairs_similarity_and_labels(embeddings, stem_to_embidx, AGEDB_PROTOCOL)

    # Load all requested methods’ quality maps
    method_maps: Dict[str, Tuple[Dict[str, float], str, str]] = {}  # method -> (stem_to_q, qcol, path)
    for method in CASE_CANDIDATES:
        path, df_case = load_case_index_with_csv(INDEX_DIR, method, FRS)
        stem_to_q, qcol = stem_quality_map_from_case_index(df_case, method)
        method_maps[method] = (stem_to_q, qcol, path)
        print(f"[INFO] {method}: {path} | qcol={qcol} | stems={len(stem_to_q)}")

    if BASELINE_CASE_ID not in method_maps:
        raise RuntimeError(f"BASELINE_CASE_ID={BASELINE_CASE_ID} must be present in CASE_CANDIDATES")

    # Common valid mask across ALL methods we will plot
    mask_common = np.ones(len(scores), dtype=bool)
    for method, (stem_to_q, _, _) in method_maps.items():
        mask_common &= pair_availability_mask(s1, s2, stem_to_q)

    idx_keep = np.where(mask_common)[0]
    print(f"[INFO] pairs common to ALL plotted methods: {len(idx_keep)} / {len(scores)}")

    # Prepare baseline sample pairs
    stem_to_q_base = method_maps[BASELINE_CASE_ID][0]
    sp_baseline = subset_sample_pairs(scores, s1, s2, idx_keep, stem_to_q_base)

    error_type = EdcErrorType.FNMR
    curves: Dict[str, EdcOutput] = {}
    results_rows: List[Dict[str, object]] = []

    # Compute all EDC curves + pAUC
    for method, (stem_to_q, qcol, path) in method_maps.items():
        sp_method = subset_sample_pairs(scores, s1, s2, idx_keep, stem_to_q)

        edc = compute_edc_repo_style(
            error_type,
            sp_method,
            starting_error=STARTING_ERROR,
            pair_quality_score_function=min,
        )
        pauc_per_limit = {}
        for dlim in DISCARD_LIMITS:
            pauc_per_limit[f"pauc@{int(dlim * 100)}"] = compute_edc_pauc_repo_style(edc, dlim)

        curves[method] = edc

        row: Dict[str, object] = {
            "method": method,
            "case_csv": path,
            "quality_col": qcol,
            "starting_error": STARTING_ERROR,
            "threshold": edc["similarity_score_threshold"],
            "n_pairs": len(sp_method),
        }
        row.update(pauc_per_limit)
        results_rows.append(row)

    # Bootstrap differences baseline vs each method (paired)
    for dlim in DISCARD_LIMITS:
        base_pauc = compute_edc_pauc_repo_style(curves[BASELINE_CASE_ID], dlim)

        for method in CASE_CANDIDATES:
            if method == BASELINE_CASE_ID:
                continue

            stem_to_q = method_maps[method][0]
            sp_case = subset_sample_pairs(scores, s1, s2, idx_keep, stem_to_q)

            diffs, mean_diff, ci_low, ci_high = bootstrap_pauc_difference(
                sample_pairs_baseline=sp_baseline,
                sample_pairs_case=sp_case,
                error_type=error_type,
                starting_error=STARTING_ERROR,
                discard_limit=dlim,
                n_boot=N_BOOT,
                seed=SEED,
            )

            sig = "NO" if (ci_low <= 0.0 <= ci_high) else "YES"

            print("\n--------------------------------------------------")
            print(f"[BOOT {method}] discard≤{dlim:.1f}")
            print(f"Mean Difference (Baseline - Case): {mean_diff:.6f}")
            print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
            print("No significant difference" if sig == "NO" else "Significant difference")

            results_rows.append(
                {
                    "method": method,
                    "discard_limit": float(dlim),
                    "baseline_method": BASELINE_CASE_ID,
                    "baseline_pauc": float(base_pauc),
                    "mean_diff_baseline_minus_case": float(mean_diff),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "significant": sig,
                }
            )

            if SAVE_PER_CASE_PLOTS:
                out_case = os.path.join(
                    INDEX_DIR,
                    f"EDC_repoStyle_{BASELINE_CASE_ID}_vs_{method}_AgeDB_{FRS}_upto60.png",
                )
                plot_edc_curves(
                    {BASELINE_CASE_ID: curves[BASELINE_CASE_ID], method: curves[method]},
                    out_case,
                    title=f"Repo-style EDC | {BASELINE_CASE_ID} vs {method} | FRS={FRS}",
                    discard_max=PLOT_DISCARD_MAX,
                )

    # Combined plot
    plot_edc_curves(
        curves,
        PLOT_OUT,
        title=f"EDC | Methods: RAW_OLS + FSEL variants | FRS={FRS}",
        discard_max=PLOT_DISCARD_MAX,
    )

    # Save CSV summary
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(CSV_OUT, index=False)
    print(f"[SAVED] {CSV_OUT}")
    print("[DONE]")


if __name__ == "__main__":
    main()
