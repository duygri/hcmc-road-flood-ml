"""Kiểm định Độ chính xác Xếp hạng Nguy cơ (Top-K Capture & Lift), Spatial Validation (5kmx5km) và Phân tích Bias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "elevation_median_m",
    "elevation_p10_m",
    "relative_elevation_m",
    "slope_mean_deg",
    "builtup_ratio_250m",
    "vegetation_ratio_250m",
    "dist_water_m",
    "water_area_ratio_250m",
]


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

    feat_cols = [c for c in FEATURE_COLS if c in base_eval.columns]
    eval_df = (
        base_eval[
            [
                "road_segment_id",
                "partition",
                "block_id",
                "label",
                "highway",
                "length_m",
                *feat_cols,
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
        subset=["score", "flood_susceptibility_prob", "length_m", *feat_cols]
    ).reset_index(drop=True)

    # Huấn luyện mô hình tuyến tính ít tham số (Regularized L2 Logistic Regression - 9 tham số)
    # trên tập Reference để đối chiếu Ablation với Deep MLP (tránh nguy cơ Overfitting cỡ mẫu nhỏ)
    ref_df = eval_df[eval_df["partition"] == "reference"]
    if (ref_df["label"] == 1.0).sum() < 10:
        ref_df = eval_df
    scaler_lr = StandardScaler()
    X_ref = scaler_lr.fit_transform(ref_df[feat_cols].to_numpy(dtype=np.float32))
    y_ref = ref_df["label"].to_numpy(dtype=np.int32)
    clf_lr = LogisticRegression(C=0.25, class_weight="balanced", max_iter=500, random_state=42)
    clf_lr.fit(X_ref, y_ref)
    X_all_lr = scaler_lr.transform(eval_df[feat_cols].to_numpy(dtype=np.float32))
    eval_df["logreg_prob"] = clf_lr.predict_proba(X_all_lr)[:, 1]

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

    # 1. Risk Ranking Accuracy trên dữ liệu Positive-Unlabeled (PU):
    # Ưu tiên Capture@K% và Lift@K% theo chiều dài đường làm thước đo chính;
    # PU-AUC (Presence vs Unlabeled Background) chỉ mang tính tham khảo vì nhãn 0 là unknown.
    top_k_budgets = (0.01, 0.05, 0.10, 0.20)
    methods = (
        ("Deep MLP (Nonlinear Ablation)", "flood_susceptibility_prob"),
        ("Logistic Regression (L2 - 9 tham số)", "logreg_prob"),
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
        row_item: dict[str, object] = {"Phương pháp": method_label}
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

    # 2. Spatial Validation across 5km x 5km blocks (Capture@10%, Lift@10%, Capture@20% & Spearman)
    block_rows = []
    for block_id, grp in holdout_df.groupby("block_id", sort=True):
        n_roads = len(grp)
        n_pos = int((grp["label"] == 1.0).sum())
        if n_roads < 30:
            continue
        rho, _ = spearmanr(grp["flood_susceptibility_prob"], grp["score"])
        grp_total_len = max(float(grp["length_m"].sum()), 1e-6)
        grp_flood_len = max(float(grp.loc[grp["label"] == 1.0, "length_m"].sum()), 1e-6)

        def _block_capture(df_block: pd.DataFrame, score_col: str, budget_f: float = 0.10) -> tuple[float, float]:
            if n_pos == 0:
                return np.nan, np.nan
            ord_b = df_block.sort_values([score_col, "road_segment_id"], ascending=[False, True], kind="stable")
            c_len = ord_b["length_m"].cumsum().to_numpy()
            idx_cut = min(int(np.searchsorted(c_len, budget_f * grp_total_len, side="left")) + 1, len(ord_b))
            sub_b = ord_b.iloc[:idx_cut]
            cap_r = float(sub_b.loc[sub_b["label"] == 1.0, "length_m"].sum()) / grp_flood_len
            act_f = float(sub_b["length_m"].sum()) / grp_total_len
            return cap_r, cap_r / max(act_f, 1e-6)

        mlp_cap10, mlp_lift10 = _block_capture(grp, "flood_susceptibility_prob", 0.10)
        mlp_cap20, _ = _block_capture(grp, "flood_susceptibility_prob", 0.20)
        lr_cap10, _ = _block_capture(grp, "logreg_prob", 0.10)
        mcda_cap10, _ = _block_capture(grp, "score", 0.10)

        block_rows.append(
            {
                "block_id": block_id,
                "Số đoạn đường": n_roads,
                "Số đoạn ngập": n_pos,
                "MLP Capture@10%": round(mlp_cap10, 4) if np.isfinite(mlp_cap10) else None,
                "MLP Lift@10%": round(mlp_lift10, 2) if np.isfinite(mlp_lift10) else None,
                "MLP Capture@20%": round(mlp_cap20, 4) if np.isfinite(mlp_cap20) else None,
                "LogReg Capture@10%": round(lr_cap10, 4) if np.isfinite(lr_cap10) else None,
                "MCDA Capture@10%": round(mcda_cap10, 4) if np.isfinite(mcda_cap10) else None,
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
        "methodological_notes": {
            "pu_learning_evaluation": (
                "Dữ liệu điểm ngập là Positive-Unlabeled (PU): các đoạn đường chưa có báo cáo là "
                "unknown (unlabeled background), không phải verified negative. Vì vậy Capture@K% và "
                "Lift@K% theo chiều dài đường là thước đo chính; PU-AUC chỉ dùng để tham khảo."
            ),
            "model_capacity_ablation": (
                "Deep MLP được trình bày như một thí nghiệm thăm dò tính phi tuyến (Nonlinear Ablation Study) "
                "đặt cạnh Logistic Regression chuẩn hóa L2 (9 tham số) và Baseline MCDA địa vật lý."
            ),
        },
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

