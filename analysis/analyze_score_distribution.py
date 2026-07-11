#!/usr/bin/env python3
"""Analyze NAVSIM PDM score CSV files.

This adapted version filters pandas index columns, invalid rows, and the
summary row before computing per-scene statistics by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUB_SCORE_COLS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
]
TOTAL_SCORE_COL = "score"
TOKEN_COL = "token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NAVSIM PDM score CSV")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to score CSV")
    parser.add_argument("--output_dir", type=str, default="./analysis_output/score_analysis")
    parser.add_argument("--low_score_ratio", type=float, default=0.3)
    parser.add_argument("--include_invalid", action="store_true", help="Keep valid=False rows if present")
    parser.add_argument("--include_average", action="store_true", help="Keep token == average row if present")
    return parser.parse_args()


def _drop_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    return df.drop(columns=drop_cols) if drop_cols else df


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required score columns: {missing}. Available columns: {list(df.columns)}"
        )


def load_score_csv(csv_path: str, include_invalid: bool, include_average: bool) -> pd.DataFrame:
    df = _drop_index_columns(pd.read_csv(csv_path))
    _validate_columns(df, [TOKEN_COL, TOTAL_SCORE_COL] + SUB_SCORE_COLS)

    if not include_average:
        df = df[df[TOKEN_COL].astype(str) != "average"]
    if not include_invalid and "valid" in df.columns:
        df = df[df["valid"].astype(bool)]

    if len(df) == 0:
        raise ValueError("No rows left after filtering score CSV")
    return df.reset_index(drop=True)


def compute_statistics(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    stats = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) == 0:
            raise ValueError(f"Column {col} has no numeric values")
        stats.append(
            {
                "metric": col,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "zero_fraction": float((values == 0).sum() / len(values)),
                "one_fraction": float((values == 1).sum() / len(values)),
            }
        )
    return pd.DataFrame(stats)


def plot_histograms(df: pd.DataFrame, cols: List[str], output_dir: Path) -> None:
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(cols):
        ax = axes[idx]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(s.values, bins=50, range=(0, 1), alpha=0.7, edgecolor="black")
        ax.axvline(float(s.mean()), color="red", linestyle="--", label=f"mean={s.mean():.3f}")
        ax.axvline(float(s.median()), color="green", linestyle="--", label=f"median={s.median():.3f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.set_title(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(cols), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "score_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, cols: List[str], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [pd.to_numeric(df[col], errors="coerce").dropna().values for col in cols]
    bp = ax.boxplot(data, labels=[c.replace("_", "\n") for c in cols], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_ylabel("score")
    ax.set_title("Sub-score Boxplot")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "score_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str], output_dir: Path) -> None:
    corr_cols = cols + [TOTAL_SCORE_COL]
    corr = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in corr_cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([c.replace("_", "\n") for c in corr_cols], fontsize=8)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Score Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "score_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def extract_low_score_tokens(df: pd.DataFrame, ratio: float, output_dir: Path) -> pd.DataFrame:
    if ratio <= 0:
        return df.head(0)
    n_low = max(1, int(len(df) * ratio))
    low_df = df.sort_values(by=TOTAL_SCORE_COL, ascending=True).head(n_low)

    pct = int(ratio * 100)
    token_file = output_dir / f"low_score_tokens_bottom{pct}pct.txt"
    with open(token_file, "w") as f:
        for token in low_df[TOKEN_COL].astype(str).tolist():
            f.write(f"{token}\n")

    low_df[[TOKEN_COL, TOTAL_SCORE_COL] + SUB_SCORE_COLS].to_csv(
        output_dir / f"low_score_cases_bottom{pct}pct.csv", index=False
    )
    return low_df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_score_csv(args.csv_path, args.include_invalid, args.include_average)
    print(f"Loaded {len(df)} filtered per-scene rows from {args.csv_path}")

    all_cols = SUB_SCORE_COLS + [TOTAL_SCORE_COL]
    stats_df = compute_statistics(df, all_cols)
    stats_df.to_csv(output_dir / "score_statistics.csv", index=False)
    print(stats_df.to_string(index=False))

    plot_histograms(df, all_cols, output_dir)
    plot_boxplots(df, SUB_SCORE_COLS, output_dir)
    plot_correlation_heatmap(df, SUB_SCORE_COLS, output_dir)

    low_df = extract_low_score_tokens(df, args.low_score_ratio, output_dir)
    if len(low_df) > 0:
        print(f"\nLow-score cases: n={len(low_df)}")
        print(f"  score range: [{low_df[TOTAL_SCORE_COL].min():.4f}, {low_df[TOTAL_SCORE_COL].max():.4f}]")
        for col in SUB_SCORE_COLS:
            print(f"  {col}: {low_df[col].mean():.4f}")

    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
