#!/usr/bin/env python3
"""Compare two score exports with paired scene-level deltas and slices."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PDM_SUB_SCORES = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
]
DEFAULT_SLICE_COLUMNS = [
    "map_name",
    "maneuver",
    "speed_bin",
    "geometry_complexity",
    "in_intersection",
    "near_intersection",
    "has_traffic_light",
    "has_red_light",
    "agent_density_bin",
    "nearest_agent_distance_bin",
    "has_vru",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--variant_csv", required=True)
    parser.add_argument("--baseline_name", default="baseline")
    parser.add_argument("--variant_name", default="variant")
    parser.add_argument("--attributes_csv", default=None)
    parser.add_argument("--output_dir", default="analysis_output/scene_score_comparison")
    parser.add_argument("--score_column", default="auto", help="auto, score, or selected_score")
    parser.add_argument("--slice_columns", nargs="*", default=None)
    parser.add_argument("--min_slice_size", type=int, default=30)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tie_tolerance", type=float, default=1e-9)
    parser.add_argument("--extreme_count", type=int, default=50)
    return parser.parse_args()


def load_export(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in df.columns if str(col).startswith("Unnamed:")])
    if "token" not in df.columns:
        raise ValueError(f"{path} has no token column")
    df["token"] = df["token"].astype(str)
    df = df[df["token"] != "average"]
    if "valid" in df.columns:
        valid = df["valid"]
        if valid.dtype == object:
            valid = valid.astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
            if valid.isna().any():
                raise ValueError(f"{path} contains unrecognized valid values")
        df = df[valid.fillna(False).astype(bool)]
    duplicates = df[df["token"].duplicated()]["token"].unique()
    if len(duplicates):
        raise ValueError(f"{path} contains duplicate tokens, first={duplicates[0]}")
    return df.reset_index(drop=True)


def resolve_score_column(baseline: pd.DataFrame, variant: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        candidates = [requested]
    else:
        candidates = ["selected_score", "score"]
    for candidate in candidates:
        if candidate in baseline.columns and candidate in variant.columns:
            return candidate
    raise ValueError(f"Could not resolve score column from candidates {candidates}")


def comparison_columns(
    baseline: pd.DataFrame, variant: pd.DataFrame, score_column: str
) -> List[str]:
    columns = [score_column]
    if score_column == "selected_score":
        columns.extend(
            [f"selected_{metric}" for metric in PDM_SUB_SCORES]
            + ["oracle_score", "ranking_regret"]
        )
    else:
        columns.extend(PDM_SUB_SCORES)
    return [col for col in columns if col in baseline.columns and col in variant.columns]


def pair_exports(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    columns: Iterable[str],
    score_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_tokens = set(baseline["token"])
    variant_tokens = set(variant["token"])
    missing_rows = [
        {"token": token, "missing_from": "variant"}
        for token in sorted(baseline_tokens - variant_tokens)
    ] + [
        {"token": token, "missing_from": "baseline"}
        for token in sorted(variant_tokens - baseline_tokens)
    ]

    left = baseline[["token"] + list(columns)].rename(
        columns={column: f"baseline_{column}" for column in columns}
    )
    right = variant[["token"] + list(columns)].rename(
        columns={column: f"variant_{column}" for column in columns}
    )
    paired = left.merge(right, on="token", how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError("Baseline and variant have no common valid tokens")
    for column in columns:
        baseline_values = pd.to_numeric(paired[f"baseline_{column}"], errors="coerce")
        variant_values = pd.to_numeric(paired[f"variant_{column}"], errors="coerce")
        invalid = baseline_values.isna() | variant_values.isna()
        if invalid.any():
            bad_token = paired.loc[invalid, "token"].iloc[0]
            raise ValueError(f"Non-numeric {column} value for paired token {bad_token}")
        paired[f"delta_{column}"] = variant_values - baseline_values
    paired = paired.rename(columns={f"delta_{score_column}": "delta_score"})

    if score_column == "selected_score" and {
        "delta_oracle_score",
        "delta_ranking_regret",
    }.issubset(paired.columns):
        paired["delta_ranking_ability"] = -paired["delta_ranking_regret"]
        residual = paired["delta_score"] - (
            paired["delta_oracle_score"] + paired["delta_ranking_ability"]
        )
        if not np.allclose(residual.fillna(0.0), 0.0, atol=1e-8, rtol=1e-7):
            raise AssertionError("Generation/ranking decomposition does not reconstruct delta_score")
    return paired, pd.DataFrame(missing_rows, columns=["token", "missing_from"])


def bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0 or samples <= 0:
        return np.nan, np.nan
    means = np.empty(samples, dtype=np.float64)
    for sample_idx in range(samples):
        means[sample_idx] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_delta(
    values: pd.Series,
    metric: str,
    tolerance: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(array) == 0:
        raise ValueError(f"No numeric delta values for {metric}")
    ci_low, ci_high = bootstrap_ci(array, bootstrap_samples, rng)
    return {
        "metric": metric,
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
        "win_rate": float(np.mean(array > tolerance)),
        "tie_rate": float(np.mean(np.abs(array) <= tolerance)),
        "loss_rate": float(np.mean(array < -tolerance)),
        "mean_ci_low": ci_low,
        "mean_ci_high": ci_high,
    }


def delta_columns(paired: pd.DataFrame) -> List[str]:
    preferred = ["delta_score", "delta_oracle_score", "delta_ranking_ability"]
    remaining = [
        col
        for col in paired.columns
        if col.startswith("delta_")
        and col not in preferred
        and col != "delta_ranking_regret"
    ]
    return [col for col in preferred if col in paired.columns] + remaining


def plot_delta_distributions(paired: pd.DataFrame, columns: List[str], output_dir: Path) -> None:
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, column in zip(axes, columns):
        values = paired[column].dropna().to_numpy()
        ax.hist(values, bins=50, alpha=0.75, edgecolor="black")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.axvline(values.mean(), color="red", linestyle="--", label=f"mean={values.mean():.4f}")
        ax.set_title(column)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(columns) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for column in columns[:3]:
        values = np.sort(paired[column].dropna().to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, label=column)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("paired delta")
    ax.set_ylabel("ECDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "delta_ecdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_slices(
    paired: pd.DataFrame,
    attributes: pd.DataFrame,
    slice_columns: List[str],
    metrics: List[str],
    min_size: int,
    tolerance: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if "token" not in attributes.columns:
        raise ValueError("attributes CSV has no token column")
    if attributes["token"].astype(str).duplicated().any():
        raise ValueError("attributes CSV contains duplicate tokens")
    attributes = attributes.copy()
    attributes["token"] = attributes["token"].astype(str)
    joined = paired.merge(attributes, on="token", how="left", validate="one_to_one")
    rows = []
    for slice_column in slice_columns:
        if slice_column not in joined.columns:
            continue
        for slice_value, group in joined.dropna(subset=[slice_column]).groupby(slice_column):
            if len(group) < min_size:
                continue
            for metric in metrics:
                row = summarize_delta(
                    group[metric], metric, tolerance, bootstrap_samples, rng
                )
                row.update(
                    {
                        "slice_name": slice_column,
                        "slice_value": slice_value,
                        "num_scenes": len(group),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    baseline = load_export(args.baseline_csv)
    variant = load_export(args.variant_csv)
    score_column = resolve_score_column(baseline, variant, args.score_column)
    columns = comparison_columns(baseline, variant, score_column)
    paired, missing = pair_exports(baseline, variant, columns, score_column)
    paired.insert(1, "baseline_name", args.baseline_name)
    paired.insert(2, "variant_name", args.variant_name)

    metrics = delta_columns(paired)
    summary = pd.DataFrame(
        [
            summarize_delta(
                paired[column], column, args.tie_tolerance, args.bootstrap_samples, rng
            )
            for column in metrics
        ]
    )
    paired.to_csv(output_dir / "paired_scene_scores.csv", index=False)
    missing.to_csv(output_dir / "missing_tokens.csv", index=False)
    summary.to_csv(output_dir / "delta_summary.csv", index=False)

    n_extreme = min(args.extreme_count, len(paired))
    paired.nsmallest(n_extreme, "delta_score").to_csv(
        output_dir / "largest_regressions.csv", index=False
    )
    paired.nlargest(n_extreme, "delta_score").to_csv(
        output_dir / "largest_improvements.csv", index=False
    )
    plot_delta_distributions(paired, metrics, output_dir)

    if args.attributes_csv:
        attributes = pd.read_csv(args.attributes_csv)
        slice_columns = args.slice_columns if args.slice_columns is not None else DEFAULT_SLICE_COLUMNS
        if args.slice_columns is not None:
            missing_slice_columns = [col for col in slice_columns if col not in attributes.columns]
            if missing_slice_columns:
                raise ValueError(f"Attributes CSV is missing requested slices: {missing_slice_columns}")
        attribute_tokens = set(attributes["token"].astype(str)) if "token" in attributes else set()
        missing_attributes = paired.loc[~paired["token"].isin(attribute_tokens), ["token"]]
        missing_attributes.to_csv(output_dir / "missing_attribute_tokens.csv", index=False)
        slice_summary = summarize_slices(
            paired,
            attributes,
            slice_columns,
            metrics,
            args.min_slice_size,
            args.tie_tolerance,
            args.bootstrap_samples,
            rng,
        )
        slice_summary.to_csv(output_dir / "slice_summary.csv", index=False)

    print(f"Paired {len(paired)} scenes; unmatched={len(missing)}")
    print(summary.to_string(index=False))
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
