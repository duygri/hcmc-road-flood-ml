"""Kiểm định Độ chính xác Xếp hạng Nguy cơ (Top-K Capture & Lift), Spatial Validation (5kmx5km) và Phân tích Bias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_audit(processed_dir: Path, run_out_dir: Path, run_id: str) -> dict:
    df_feat = pd.read_parquet(processed_dir / "road_features.parquet")
    df_part = pd.read_parquet(processed_dir / "road_partitions.parquet")
    df_mcda = pd.read_parquet(processed_dir / "road_susceptibility.parquet")
    df_matches = pd.read_parquet(processed_dir / "road_flood_matches.parquet")
    df_confirmed = pd.read_parquet(processed_dir / "confirmed_event_segments.parquet")

    pred_path = run_out_dir / "a100_road_flood_predictions.parquet"
    if not pred_path.is_file():
        alt_pred = run_out_dir / "scores" / "a100_road_flood_predictions.parquet"
        pred_path = alt_pred if alt_pred.is_file() else pred_path
    pred_df = pd.read_parquet(pred_path)

    if len(df_confirmed) > 0 and "road_segment_id" in df_confirmed.columns:
        pos_ids = set(df_confirmed["road_segment_id"].dropna().unique())
    else:
        valid_matches = df_matches.dropna(subset=["road_segment_id"])
        if "match_distance_m" in valid_matches.columns:
            close_matches = valid_matches[valid_matches["match_distance_m"] <= 50.0]
            if len(close_matches) >= 50:
                valid_matches = close_matches
        pos_ids = set(valid_matches["road_segment_id"].unique())

    base_eval = df_feat.merge(
        df_part[["road_segment_id", "partition", "block_id"]],
        on="road_segment_id",
        how="inner",
    )
    base_eval["label"] = base_eval["road_segment_id"].isin(pos_ids).astype(np.float32)

    if "highway" not in base_eval.columns:
        base_eval["highway"] = (
            base_eval["highway_type"]
            if "highway_type" in base_eval.columns
            else "residential"
        )
    if "length_m" not in base_eval.columns:
        base_eval["length_m"] = (
            base_eval["road_length_m"]
            if "road_length_m" in base_eval.columns
            else 150.0
        )

    eval_df = (
        base_eval[
            [
                "road_segment_id",
                "partition",
                "block_id",
                "label",
                "highway",
                "length_m",
                "elevation_median_m",
                "relative_elevation_m",
                "dist_water_m",
                "builtup_ratio_250m",
            ]
        ]
        .merge(df_mcda[["road_segment_id", "score"]], on="road_segment_id", how="left")
        .merge(
            pred_df[["road_segment_id", "flood_susceptibility_prob"]],
            on="road_segment_id",
            how="left",
        )
    )
    eval_df = eval_df.dropna(
        subset=["score", "flood_susceptibility_prob", "length_m"]
    ).reset_index(drop=True)
    eval_df["elev_risk_score"] = -eval_df["elevation_median_m"]
    eval_df["hybrid_score"] = 0.5 * eval_df["flood_susceptibility_prob"] + 0.5 * (
        eval_df["score"] / 100.0
    )

    holdout_df = eval_df[eval_df["partition"] == "holdout"].copy()
    if (holdout_df["label"] == 1.0).sum() < 5:
        holdout_df = eval_df.copy()

    total_holdout_len_m = float(holdout_df["length_m"].sum())
    flooded_holdout_len_m = float(
        holdout_df.loc[holdout_df["label"] == 1.0, "length_m"].sum()
    )

    # 1. Risk Ranking Accuracy (Capture@K & Lift@K)
    top_k_budgets = (0.01, 0.05, 0.10, 0.20)
    methods = (
        ("Deep MLP (A100 Best)", "flood_susceptibility_prob"),
        ("Hybrid Ensemble (MLP + MCDA)", "hybrid_score"),
        ("Baseline MCDA (Địa vật lý)", "score"),
        ("Elevation-Only (Độ cao thuần)", "elev_risk_score"),
    )
    ranking_rows = []
    for method_label, col_name in methods:
        ordered = holdout_df.sort_values(
            [col_name, "road_segment_id"], ascending=[False, True], kind="stable"
        )
        cum_len = ordered["length_m"].cumsum().to_numpy()
        auc_val = (
            float(roc_auc_score(ordered["label"], ordered[col_name]))
            if len(np.unique(ordered["label"])) > 1
            else 0.0
        )
        row_item = {"Phương pháp": method_label, "Holdout ROC-AUC": round(auc_val, 4)}
        for k_frac in top_k_budgets:
            target_len = k_frac * total_holdout_len_m
            n_sel = min(
                int(np.searchsorted(cum_len, target_len, side="left")) + 1,
                len(ordered),
            )
            sub = ordered.iloc[:n_sel]
            cap_len = float(sub.loc[sub["label"] == 1.0, "length_m"].sum())
            cap_rate = cap_len / max(flooded_holdout_len_m, 1e-6)
            actual_frac = float(sub["length_m"].sum()) / max(total_holdout_len_m, 1e-6)
            lift = cap_rate / max(actual_frac, 1e-6)
            pct = int(k_frac * 100)
            row_item[f"Capture@{pct}%"] = round(cap_rate, 4)
            row_item[f"Lift@{pct}%"] = round(lift, 2)
        ranking_rows.append(row_item)

    # 2. Spatial Validation across 5km x 5km blocks
    block_rows = []
    for block_id, grp in holdout_df.groupby("block_id", sort=True):
        n_roads = len(grp)
        n_pos = int((grp["label"] == 1.0).sum())
        if n_roads < 30:
            continue
        rho, _ = spearmanr(grp["flood_susceptibility_prob"], grp["score"])
        b_auc = (
            float(roc_auc_score(grp["label"], grp["flood_susceptibility_prob"]))
            if 2 <= n_pos < n_roads
            else np.nan
        )
        mcda_auc = (
            float(roc_auc_score(grp["label"], grp["score"]))
            if 2 <= n_pos < n_roads
            else np.nan
        )
        ord_b = grp.sort_values("flood_susceptibility_prob", ascending=False)
        top10_cut = max(1, int(len(ord_b) * 0.10))
        top10_cap = (
            float(ord_b.iloc[:top10_cut].loc[ord_b["label"] == 1.0, "length_m"].sum())
            / max(float(grp.loc[grp["label"] == 1.0, "length_m"].sum()), 1e-6)
            if n_pos > 0
            else np.nan
        )
        block_rows.append(
            {
                "block_id": block_id,
                "Số đoạn đường": n_roads,
                "Số đoạn ngập": n_pos,
                "MLP ROC-AUC": round(b_auc, 4) if np.isfinite(b_auc) else None,
                "MCDA ROC-AUC": round(mcda_auc, 4) if np.isfinite(mcda_auc) else None,
                "Capture@10%": round(top10_cap, 4) if np.isfinite(top10_cap) else None,
                "Tương quan Spearman (MLP vs MCDA)": (
                    round(float(rho), 4) if np.isfinite(rho) else None
                ),
            }
        )

    # 3. Bias Analysis by Highway Class
    eval_df["road_group"] = eval_df["highway"].map(
        lambda h: (
            h
            if h
            in {
                "primary",
                "secondary",
                "tertiary",
                "residential",
                "service",
                "unclassified",
                "trunk",
            }
            else "other"
        )
    )
    bias_rows = []
    overall_pos_rate = float(eval_df["label"].mean())
    q90_mlp = float(eval_df["flood_susceptibility_prob"].quantile(0.90))
    q90_mcda = float(eval_df["score"].quantile(0.90))
    for r_group, grp in eval_df.groupby("road_group"):
        if len(grp) < 50:
            continue
        obs_rate = float(grp["label"].mean())
        mlp_top10_rate = float((grp["flood_susceptibility_prob"] >= q90_mlp).mean())
        mcda_top10_rate = float((grp["score"] >= q90_mcda).mean())
        bias_rows.append(
            {
                "Cấp đường (highway)": r_group,
                "Số đoạn đường": len(grp),
                "Tỷ lệ báo cáo ngập (%)": round(obs_rate * 100, 2),
                "Hệ số Reporting Bias (so với TB)": round(
                    obs_rate / max(overall_pos_rate, 1e-6), 2
                ),
                "Điểm MLP TB": round(
                    float(grp["flood_susceptibility_prob"].mean()), 4
                ),
                "Điểm MCDA TB (0-100)": round(float(grp["score"].mean()), 2),
                "Tỷ lệ vào Top-10% Nguy cơ (MLP %)": round(mlp_top10_rate * 100, 2),
                "Tỷ lệ vào Top-10% Nguy cơ (MCDA %)": round(mcda_top10_rate * 100, 2),
            }
        )

    report = {
        "run_id": run_id,
        "risk_ranking_accuracy_holdout": ranking_rows,
        "spatial_validation_by_block": block_rows,
        "highway_bias_analysis": bias_rows,
    }
    out_json = run_out_dir / "spatial_ranking_bias_report.json"
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--run-out-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="local_audit")
    args = parser.parse_args()
    report = run_audit(args.processed_dir, args.run_out_dir, args.run_id)
    print(json.dumps(report["risk_ranking_accuracy_holdout"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

