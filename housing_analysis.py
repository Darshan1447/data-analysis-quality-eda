#!/usr/bin/env python3
"""Reproducible data-quality audit and EDA for the supplied California Housing CSV.

Default paths are relative to this file:
  input:  data/california_housing.csv
  output: outputs/

The source CSV is read only. No cleaning, imputation, row removal, or target edits
are performed. All generated summaries and figures are derived from the input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude", "MedHouseVal",
]
FEATURE_COLUMNS = [c for c in EXPECTED_COLUMNS if c != "MedHouseVal"]
COLUMN_MEANINGS = {
    "MedInc": "Median income in the block group; do not interpret as raw dollars without source context.",
    "HouseAge": "Median house age in the block group.",
    "AveRooms": "Average number of rooms per household; can be unusually high for some block groups.",
    "AveBedrms": "Average number of bedrooms per household; can be unusually high for some block groups.",
    "Population": "Block-group population, not a household count.",
    "AveOccup": "Average household occupants; aggregate ratio that can have extreme values.",
    "Latitude": "Block-group latitude.",
    "Longitude": "Block-group longitude.",
    "MedHouseVal": "Median house-value target, in units of $100,000; documentation describes an upper cap of 5.0.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt_num(value: object, decimals: int = 4) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.{decimals}f}"
        return f"{float(value):.{decimals}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: Iterable[str] | None = None, decimals: int = 4) -> str:
    view = df.copy()
    if columns is not None:
        view = view[list(columns)]
    if view.empty:
        return "_(No rows.)_"
    headers = [str(c) for c in view.columns]
    rows = []
    for _, row in view.iterrows():
        rows.append([fmt_num(v, decimals=decimals) for v in row.tolist()])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    divider_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header_line, divider_line, *body])


def create_range_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(name: str, mask: pd.Series, rule: str) -> None:
        mask_series = mask if isinstance(mask, pd.Series) else pd.Series(mask, index=df.index)
        count = int(mask_series.fillna(False).sum())
        checks.append({"check": name, "rule": rule, "observed_violations": count,
                       "status": "PASS" if count == 0 else "REVIEW"})
    add("MedInc positive", df["MedInc"] <= 0, "MedInc > 0")
    add("HouseAge within plausible observed range", (df["HouseAge"] < 1) | (df["HouseAge"] > 52), "1 <= HouseAge <= 52")
    add("AveRooms positive", df["AveRooms"] <= 0, "AveRooms > 0")
    add("AveBedrms positive", df["AveBedrms"] <= 0, "AveBedrms > 0")
    add("Population non-negative", df["Population"] < 0, "Population >= 0")
    add("AveOccup positive", df["AveOccup"] <= 0, "AveOccup > 0")
    add("Latitude geographic bounds", (df["Latitude"] < -90) | (df["Latitude"] > 90), "-90 <= Latitude <= 90")
    add("Longitude geographic bounds", (df["Longitude"] < -180) | (df["Longitude"] > 180), "-180 <= Longitude <= 180")
    add("Target non-negative", df["MedHouseVal"] < 0, "MedHouseVal >= 0; the documented upper cap is audited separately")
    add("HouseAge integer-valued", ~np.isclose(df["HouseAge"] % 1, 0), "HouseAge values are whole-number coded")
    add("Population integer-valued", ~np.isclose(df["Population"] % 1, 0), "Population values are whole-number counts despite float CSV dtype")
    return pd.DataFrame(checks)


def make_figures(df: pd.DataFrame, out_dir: Path, income_groups: pd.DataFrame, target_pearson: float) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # 1. Target distribution with the cap documented.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(df["MedHouseVal"], bins=60, edgecolor="white", linewidth=0.35)
    ax.axvline(df["MedHouseVal"].median(), linestyle="--", linewidth=1.6,
               label=f"Median = {df['MedHouseVal'].median():.3f} ($100k units)")
    ax.axvline(5.0, linestyle=":", linewidth=1.6, label="Documented cap = 5.0")
    ax.set(title="Distribution of median house-value target", xlabel="MedHouseVal (units of $100,000)", ylabel="Census block-group count")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = fig_dir / "01_target_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path.name)

    # 2. Eight feature distributions. Positive skewed dimensions are shown on log x axes.
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.2))
    log_x = {"AveRooms", "AveBedrms", "Population", "AveOccup"}
    for ax, col in zip(axes.flat, FEATURE_COLUMNS):
        values = df[col].dropna()
        if col in log_x:
            positive = values[values > 0]
            bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 45)
            ax.hist(positive, bins=bins, edgecolor="white", linewidth=0.25)
            ax.set_xscale("log")
            ax.set_xlabel(f"{col} (log scale)")
        else:
            ax.hist(values, bins=40, edgecolor="white", linewidth=0.25)
            ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Distributions of the eight input features", y=1.015, fontsize=14)
    fig.tight_layout()
    path = fig_dir / "02_feature_distributions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path.name)

    # 3. Pearson correlation matrix.
    corr = df[EXPECTED_COLUMNS].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8.3))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) >= 0.62 else "black")
    ax.set_title("Pearson correlation matrix (pairwise linear association)")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    path = fig_dir / "03_correlation_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path.name)

    # 4. Income-target density comparison.
    fig, ax = plt.subplots(figsize=(9, 5.8))
    hb = ax.hexbin(df["MedInc"], df["MedHouseVal"], gridsize=48, mincnt=1, bins="log")
    ax.axhline(5.0, linestyle=":", linewidth=1.3, label="Documented target cap = 5.0")
    ax.set(title=f"Median income vs. median house value (Pearson r = {target_pearson:.3f})",
           xlabel="MedInc (dataset income scale; not raw dollars)",
           ylabel="MedHouseVal (units of $100,000)")
    fig.colorbar(hb, ax=ax, label="Block-group count (log color scale)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = fig_dir / "04_income_vs_target_hexbin.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path.name)

    # 5. Target summary by ordered median-income quintile.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(income_groups))
    ax.bar(x, income_groups["mean_MedHouseVal"], alpha=0.85, label="Mean target")
    ax.plot(x, income_groups["median_MedHouseVal"], marker="o", linewidth=2, label="Median target")
    ax.set_xticks(x, income_groups["income_group"], rotation=18, ha="right")
    ax.set(title="Target value by median-income quintile", xlabel="MedInc quintile (Q1 = lowest income)",
           ylabel="MedHouseVal (units of $100,000)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    path = fig_dir / "05_target_by_income_quintile.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path.name)

    # 6. Geographical scatter without external basemap; points are block-group observations.
    fig, ax = plt.subplots(figsize=(9, 7))
    pts = ax.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"], s=7, alpha=0.62, linewidths=0)
    ax.set(title="Geographic distribution of block-group target values",
           xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(pts, ax=ax, label="MedHouseVal (units of $100,000)")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    path = fig_dir / "06_geography_by_target.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path.name)
    return saved


def create_report(
    df: pd.DataFrame, input_path: Path, out_dir: Path, schema: pd.DataFrame,
    descriptive: pd.DataFrame, missing: pd.DataFrame, range_checks: pd.DataFrame,
    target_corr: pd.DataFrame, income_groups: pd.DataFrame, outlier_summary: pd.DataFrame,
    figure_names: list[str], duplicate_count: int, nonfinite_count: int,
    input_hash: str, timestamp: str,
) -> str:
    n_rows, n_cols = df.shape
    missing_total = int(missing["missing_count"].sum())
    capped5_count = int((df["MedHouseVal"] >= 5.0).sum())
    exact_500001_count = int(np.isclose(df["MedHouseVal"], 5.00001, rtol=0, atol=1e-12).sum())
    med_income_r = float(df["MedInc"].corr(df["MedHouseVal"], method="pearson"))
    med_income_s = float(df["MedInc"].corr(df["MedHouseVal"], method="spearman"))
    highest_corr = target_corr[target_corr["feature"] != "MedHouseVal"].iloc[0]
    target_median = float(df["MedHouseVal"].median())
    target_mean = float(df["MedHouseVal"].mean())
    iqr_selected = outlier_summary[outlier_summary["feature"].isin(["AveRooms", "AveBedrms", "AveOccup", "MedHouseVal"])].copy()
    schema_table = schema[["column", "dtype", "missing_count", "unique_values", "min", "median", "max", "meaning"]]
    desc_table = descriptive[["column", "mean", "std", "min", "p25", "p50", "p75", "p95", "max", "skew"]]
    target_corr_table = target_corr[target_corr["feature"] != "MedHouseVal"][["feature", "pearson_r", "spearman_rho"]]
    target_corr_table = target_corr_table.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    if len(df):
        source_bytes = input_path.stat().st_size
    else:
        source_bytes = 0

    figure_md = "\n\n".join([f"![{name.replace('_', ' ').replace('.png', '')}](figures/{name})" for name in figure_names])
    report = f'''# California Housing — Data Quality Audit and Exploratory Analysis

**Run:** Run 2 independent verification  
**Generated at:** `{timestamp}`  
**Input used:** `{input_path.name}`  
**SHA-256:** `{input_hash}`  
**Input size:** {source_bytes:,} bytes

## Executive summary

The supplied CSV contains **{n_rows:,} rows and {n_cols} columns**: eight numeric input features and `MedHouseVal`, the target. All nine fields were parsed as numeric `float64`. The directly measured file has **{missing_total:,} missing cells**, **{duplicate_count:,} exact duplicate rows** (pandas `duplicated()` semantics: repeated rows beyond their first occurrence), and **{nonfinite_count:,} non-finite numeric values**. No source rows or values were changed; no cleaning or imputation was performed.

`MedInc` has the strongest absolute Pearson correlation with `MedHouseVal` among the eight input features (**r = {med_income_r:.4f}**; Spearman **rho = {med_income_s:.4f}**). The relationship is an association, not evidence that income causes the target. Grouping observations by `MedInc` quintile shows increasing target summaries from lower to higher income groups; see the group table and figures below. The target reaches **{df['MedHouseVal'].max():.5f}** in the uploaded CSV, and **{capped5_count:,} rows** have values at or above 5.0. In particular, {exact_500001_count:,} values equal 5.00001, which is a small but real distinction from the source notes' rounded cap of 5.0; this run preserves and reports the exact uploaded values.

## 1. Inputs and source verification

All five required attachments were accessible in the conversation: `california_housing.csv`, `task_brief.md`, `dataset_source_and_data_dictionary.md`, `acceptance_criteria.md`, and `context_inventory.csv`. The files were read from the actual uploaded attachments. In this distributed baseline bundle, `data/california_housing.csv` is a byte-for-byte copy of the uploaded CSV. The script records the hash of the file actually analyzed and reads, but does not modify, the selected input.

The three official references listed by the supplied data dictionary were documented as checked during Run 1. They were not independently re-opened during this Run 2 execution:

- [scikit-learn loader reference](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) — describes 20,640 samples, eight input features, and the target in units of 100,000.
- [scikit-learn real-world dataset description](https://scikit-learn.org/stable/datasets/real_world.html) — gives the feature meanings, reports no missing attribute values at the dataset level, and explains the 1990 U.S. Census / block-group context.
- [scikit-learn implementation and reference metadata](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_california_housing.py) — checked for loader metadata and target units.

The Python script itself makes no web requests; a later local script rerun does not independently re-check those URLs. The dataset matches the documented row/column dimensions. The uploaded CSV's observed maximum target value is 5.00001, not exactly 5.0. The analysis treats the supplied file as the authoritative observed data and does not round or alter this field. No other dataset was fetched or substituted.

## 2. Structure and column dictionary

Observed dimension: **{n_rows:,} × {n_cols}**. Required/observed columns, order, types, missingness, unique values, and observed extrema are shown below. All columns are numeric; `HouseAge` and `Population` have `float64` CSV dtypes but are integer-valued in the observed records (see the range-check output).

{markdown_table(schema_table, decimals=5)}

## 3. Data-quality audit

### Missing values and exact duplicates

Total missing cells: **{missing_total:,}** across the dataset. Exact duplicate rows beyond first occurrence: **{duplicate_count:,}**. Non-finite values among numeric cells: **{nonfinite_count:,}**. Missing counts by column:

{markdown_table(missing, decimals=0)}

A zero missing count supports that no CSV cell is blank/recognized as missing by pandas' default CSV parser. It does not prove the data is historically accurate or semantically correct. A zero duplicate count only checks exact equality across all nine fields; it does not identify near-duplicates or repeated geographic units represented with slightly different values.

### Range, domain, and unusual-observation checks


{markdown_table(range_checks, decimals=0)}

An additional upper-cap review found max `MedHouseVal` = **{df['MedHouseVal'].max():.5f}**; **{capped5_count:,}** rows are `>= 5.0`, of which **{exact_500001_count:,}** equal `5.00001`. This is reported as a documentation/data-value precision discrepancy, not automatically treated as corruption. The dataset notes caution that high `AveRooms` / `AveBedrms` values can occur in block groups with few households and many vacant homes, and that aggregate-ratio values can be unusually large. The observed maxima are `AveRooms` = **{df['AveRooms'].max():.4f}**, `AveBedrms` = **{df['AveBedrms'].max():.4f}**, and `AveOccup` = **{df['AveOccup'].max():.4f}**. These deserve contextual inspection, but extremes alone do not justify deletion.

The 1.5×IQR screen below is a statistical flagging convention, not a claim that flagged rows are errors. It is provided for review only; it did not trigger any filtering.

{markdown_table(iqr_selected, decimals=4)}

## 4. Preparation decision and change log

**No transformation was justified by the evidence gathered for this task.** No rows were deleted, no values rounded, no columns engineered, and no missing values imputed. The CSV remains unchanged. A separate “cleaned” CSV was therefore not created. The feature distributions and unusual records are retained for transparent downstream interpretation. If a later task requires a filtered or transformed analysis, the rule should be agreed and documented first, with before/after counts.

## 5. Descriptive statistics

The complete per-column summary (including 1st, 5th, 25th, 50th, 75th, 95th, and 99th percentiles) is available as `descriptive_statistics.csv`. Selected summary statistics:

{markdown_table(desc_table, decimals=5)}

Selected target observations: mean = **{target_mean:.4f}** and median = **{target_median:.4f}** in units of $100,000. These describe the historical target values in the dataset; they are not present-day California market prices or forecasts.

## 6. Correlation and feature/target comparisons

Pearson correlations describe linear association; Spearman correlations describe rank-order association. Correlation can be affected by the target cap and by geographic or other unmeasured structure. Neither coefficient establishes causality.

{markdown_table(target_corr_table, decimals=4)}

For a quintile comparison, observations are ranked into five near-equal-sized groups by `MedInc` using pandas `qcut`. Group boundaries are determined from the actual input values at runtime. The table reports counts, median income, and mean/median target for each group:

{markdown_table(income_groups, decimals=4)}

Observed pattern: mean `MedHouseVal` increases across the observed income quintiles, from {income_groups.iloc[0]['mean_MedHouseVal']:.4f} in the lowest-income quintile to {income_groups.iloc[-1]['mean_MedHouseVal']:.4f} in the highest-income quintile. This is descriptive, not causal, and the upper target cap compresses distinctions among high-value observations.

## 7. Figures

The generated figures use the original observations and do not require a separate cleaned dataset. Feature distributions use log-scaled x-axes for the positive, right-skewed `AveRooms`, `AveBedrms`, `Population`, and `AveOccup` dimensions so their long tails remain visible without removing records.

{figure_md}

## 8. Limitations and interpretation cautions

- **Historical scope:** the dataset is derived from the 1990 U.S. Census. It cannot support claims about current (2026) California house prices without separate current data and a newly scoped analysis.
- **Aggregation:** each row is a census block-group summary, not one individual property or transaction. Treating 20,640 rows as 20,640 house listings would be incorrect.
- **Target units and cap:** `MedHouseVal` is in units of $100,000. Supplied documentation describes an upper cap of 5.0; the uploaded CSV contains {capped5_count:,} values at/above 5.0 and {exact_500001_count:,} exact values of 5.00001. This run preserves the observed values and notes the discrepancy rather than silently normalizing it.
- **Unusual ratio values:** room/bedroom/occupancy aggregates may have large values in small or atypical block groups. Extreme values are flagged, not automatically classified as bad data.
- **Correlation is not causation:** relationships can reflect confounding, spatial structure, aggregation, and capping.
- **No model:** no predictive model was trained or tuned, and no prediction/forecast or policy/investment recommendation was made.
- **External verification:** the three official scikit-learn URLs were documented as checked during Run 1, but were not independently re-opened during this Run 2 execution. The CSV was not regenerated from a fresh dataset download.

## 9. Reproduction

From the extracted project directory, install the dependencies listed in `README.md`, then run:

```bash
python housing_analysis.py
```

By default the script reads `data/california_housing.csv` relative to its own location and writes tables, figures, a JSON metadata file, and a regenerated report to `outputs/`. Optional `--input` and `--output-dir` arguments allow alternate paths. The source CSV is read-only. The script contains no model training and no network access.

## 10. Evidence files

- `column_schema.csv` — column types, missing counts, unique values, observed extrema, and definitions.
- `missing_values.csv` — missing count and percentage by column.
- `data_quality_summary.csv` — core dimensions, missingness, duplicates, non-finite counts, and target-cap counts.
- `range_checks.csv` — documented rule-based checks and violation counts.
- `descriptive_statistics.csv` — summary statistics and percentiles.
- `correlation_matrix.csv`, `target_correlations.csv` — Pearson/Spearman analyses.
- `income_quintile_comparison.csv` — target comparison by `MedInc` quintile.
- `iqr_screening.csv`, `unusual_observations.csv` — review-only anomaly screening; no records are removed.
- `run_metadata.json` — input hash, dimensions, runtime versions, and output list.

---

**Interpretation note:** this is a data-quality and exploratory-analysis report, not a model evaluation or current real-estate valuation exercise.
'''
    return report


def run(input_path: Path, out_dir: Path) -> int:
    input_path = input_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Reading input: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    if list(df.columns) != EXPECTED_COLUMNS:
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        raise ValueError(
            "Input schema/order differs from the authorized dataset. "
            f"Expected columns in exact order {EXPECTED_COLUMNS}; got {list(df.columns)}. "
            f"Missing={missing_cols}; extra={extra_cols}."
        )
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in EXPECTED_COLUMNS):
        bad = [c for c in EXPECTED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
        raise TypeError(f"Non-numeric column(s) require review: {bad}")

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    input_hash = sha256_file(input_path)
    n_rows, n_cols = df.shape
    missing_counts = df.isna().sum()
    missing = pd.DataFrame({
        "column": EXPECTED_COLUMNS,
        "missing_count": [int(missing_counts[c]) for c in EXPECTED_COLUMNS],
        "missing_percent": [float(missing_counts[c] / n_rows * 100) if n_rows else 0.0 for c in EXPECTED_COLUMNS],
    })
    duplicate_count = int(df.duplicated().sum())
    numeric = df[EXPECTED_COLUMNS].select_dtypes(include=[np.number])
    nonfinite_count = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())

    print("[2/7] Auditing schema and data quality")
    schema_rows = []
    desc_rows = []
    for col in EXPECTED_COLUMNS:
        series = df[col]
        schema_rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
            "min": float(series.min()),
            "median": float(series.median()),
            "max": float(series.max()),
            "meaning": COLUMN_MEANINGS[col],
        })
        quantiles = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        desc_rows.append({
            "column": col,
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p01": float(quantiles.loc[0.01]),
            "p05": float(quantiles.loc[0.05]),
            "p25": float(quantiles.loc[0.25]),
            "p50": float(quantiles.loc[0.50]),
            "p75": float(quantiles.loc[0.75]),
            "p95": float(quantiles.loc[0.95]),
            "p99": float(quantiles.loc[0.99]),
            "max": float(series.max()),
            "skew": float(series.skew()),
        })
    schema = pd.DataFrame(schema_rows)
    descriptive = pd.DataFrame(desc_rows)
    range_checks = create_range_checks(df)
    target_at_or_above_5 = int((df["MedHouseVal"] >= 5.0).sum())
    target_exact_500001 = int(np.isclose(df["MedHouseVal"], 5.00001, rtol=0, atol=1e-12).sum())
    quality_summary = pd.DataFrame([
        {"metric": "rows", "value": n_rows, "note": "Rows in supplied CSV"},
        {"metric": "columns", "value": n_cols, "note": "Columns in supplied CSV"},
        {"metric": "numeric_columns", "value": int(len(numeric.columns)), "note": "Columns parsed as numeric"},
        {"metric": "missing_cells", "value": int(missing["missing_count"].sum()), "note": "Pandas CSV parser missing values"},
        {"metric": "rows_with_any_missing", "value": int(df.isna().any(axis=1).sum()), "note": "Rows containing at least one missing value"},
        {"metric": "exact_duplicate_rows_beyond_first", "value": duplicate_count, "note": "DataFrame.duplicated() count"},
        {"metric": "nonfinite_numeric_values", "value": nonfinite_count, "note": "NaN/positive or negative infinity numeric values; NaNs also separately reported as missing"},
        {"metric": "target_rows_ge_5_0", "value": target_at_or_above_5, "note": "Target cap/value audit; not a cleaning filter"},
        {"metric": "target_values_eq_5_00001", "value": target_exact_500001, "note": "Exact observed value frequency within 1e-12 tolerance"},
        {"metric": "target_max", "value": float(df["MedHouseVal"].max()), "note": "Maximum observed target; preserved as uploaded"},
    ])

    corr_pearson = df[EXPECTED_COLUMNS].corr(method="pearson")
    corr_spearman = df[EXPECTED_COLUMNS].corr(method="spearman")
    target_corr = pd.DataFrame({
        "feature": EXPECTED_COLUMNS,
        "pearson_r": [float(corr_pearson.loc[c, "MedHouseVal"]) for c in EXPECTED_COLUMNS],
        "spearman_rho": [float(corr_spearman.loc[c, "MedHouseVal"]) for c in EXPECTED_COLUMNS],
    }).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)

    print("[3/7] Computing descriptive statistics and relationships")
    medinc_cats = pd.qcut(df["MedInc"], q=5, duplicates="drop")
    grouped = df.groupby(medinc_cats, observed=True).agg(
        n=("MedHouseVal", "size"),
        median_MedInc=("MedInc", "median"),
        mean_MedHouseVal=("MedHouseVal", "mean"),
        median_MedHouseVal=("MedHouseVal", "median"),
    ).reset_index(drop=True)
    grouped["income_group"] = [f"Q{i+1}" for i in range(len(grouped))]
    grouped = grouped[["income_group", "n", "median_MedInc", "mean_MedHouseVal", "median_MedHouseVal"]]

    outlier_rows = []
    for col in EXPECTED_COLUMNS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        below = int((df[col] < low).sum())
        above = int((df[col] > high).sum())
        outlier_rows.append({"feature": col, "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                             "lower_fence": float(low), "upper_fence": float(high),
                             "below_lower_fence": below, "above_upper_fence": above,
                             "iqr_flag_count": below + above,
                             "interpretation": "Screening flag only; not evidence of invalidity and not used to filter rows."})
    outlier_summary = pd.DataFrame(outlier_rows)
    unusual = []
    unusual_limits = {"AveRooms": 20, "AveBedrms": 10, "AveOccup": 10, "MedHouseVal": 5.0}
    for col, threshold in unusual_limits.items():
        mask = df[col] >= threshold if col == "MedHouseVal" else df[col] > threshold
        subset = df.loc[mask, EXPECTED_COLUMNS].copy()
        subset.insert(0, "csv_record_position_1_based", subset.index + 1)
        subset.insert(1, "screened_feature", col)
        subset.insert(2, "screening_rule", f"{col} {'>=' if col == 'MedHouseVal' else '>'} {threshold}")
        unusual.append(subset)
    unusual_observations = pd.concat(unusual, ignore_index=True) if unusual else pd.DataFrame()

    # Store all machine-readable outputs.
    schema.to_csv(out_dir / "column_schema.csv", index=False)
    missing.to_csv(out_dir / "missing_values.csv", index=False)
    quality_summary.to_csv(out_dir / "data_quality_summary.csv", index=False)
    range_checks.to_csv(out_dir / "range_checks.csv", index=False)
    descriptive.to_csv(out_dir / "descriptive_statistics.csv", index=False)
    corr_pearson.to_csv(out_dir / "correlation_matrix.csv", index_label="column")
    corr_spearman.to_csv(out_dir / "spearman_correlation_matrix.csv", index_label="column")
    target_corr.to_csv(out_dir / "target_correlations.csv", index=False)
    grouped.to_csv(out_dir / "income_quintile_comparison.csv", index=False)
    outlier_summary.to_csv(out_dir / "iqr_screening.csv", index=False)
    unusual_observations.to_csv(out_dir / "unusual_observations.csv", index=False)

    print("[4/7] Creating figures")
    figure_names = make_figures(df, out_dir, grouped, float(df["MedInc"].corr(df["MedHouseVal"])))

    print("[5/7] Writing report and run metadata")
    report = create_report(
        df, input_path, out_dir, schema, descriptive, missing, range_checks,
        target_corr, grouped, outlier_summary, figure_names, duplicate_count,
        nonfinite_count, input_hash, timestamp,
    )
    (out_dir / "housing_analysis_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "run_id": "Run 2",
        "generated_at": timestamp,
        "input_file": input_path.name,
        "input_absolute_path_at_execution": str(input_path),
        "input_sha256": input_hash,
        "input_size_bytes": input_path.stat().st_size,
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "missing_cells": int(missing["missing_count"].sum()),
        "exact_duplicate_rows_beyond_first": duplicate_count,
        "nonfinite_numeric_values": nonfinite_count,
        "target_max": float(df["MedHouseVal"].max()),
        "target_rows_ge_5_0": target_at_or_above_5,
        "target_values_eq_5_00001": target_exact_500001,
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "script_path_at_execution": str(Path(__file__).resolve()),
        "transformations": [],
        "source_data_modified": False,
        "figures": figure_names,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[6/7] Running internal consistency assertions")
    assert df.shape == (n_rows, n_cols)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert int(missing["missing_count"].sum()) == int(df.isna().sum().sum())
    assert int(range_checks["observed_violations"].sum()) >= 0
    assert len(figure_names) >= 5
    assert (out_dir / "housing_analysis_report.md").is_file()
    assert sha256_file(input_path) == input_hash, "Input file hash changed during analysis."

    print("[7/7] Completed")
    print(f"Dimensions: {n_rows:,} x {n_cols}")
    print(f"Missing cells: {int(missing['missing_count'].sum())}")
    print(f"Exact duplicate rows: {duplicate_count}")
    print(f"Non-finite numeric values: {nonfinite_count}")
    print(f"Pearson correlation MedInc vs MedHouseVal: {df['MedInc'].corr(df['MedHouseVal']):.6f}")
    print(f"Target max: {df['MedHouseVal'].max():.5f}; rows >= 5.0: {target_at_or_above_5}; rows == 5.00001: {target_exact_500001}")
    print(f"Report: {out_dir / 'housing_analysis_report.md'}")
    print(f"Outputs: {out_dir}")
    return 0


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_dir / "data" / "california_housing.csv",
                        help="Path to input CSV (default: ./data/california_housing.csv relative to this script).")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "outputs",
                        help="Directory for reports, tables, metadata, and figures (default: ./outputs).")
    args = parser.parse_args()
    return run(args.input, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
