"""Tính điểm nguy cơ ngập lụt cho toàn bộ 100% đoạn đường trong AOI TP.HCM."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from flood_ml.inference import FloodSusceptibilityPredictor, compute_local_basin_features

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


def score_entire_study_area(
    processed_dir: Path,
    interim_dir: Path,
    bundle_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = FloodSusceptibilityPredictor(bundle_dir)

    df_feat = pd.read_parquet(processed_dir / "road_features.parquet")
    df_part = pd.read_parquet(processed_dir / "road_partitions.parquet")
    df_mcda = pd.read_parquet(processed_dir / "road_susceptibility.parquet")

    # Ghép phân vùng không gian trước khi tính đặc trưng cục bộ theo ô lưới (block_id)
    merged_in = df_feat.merge(
        df_part[["road_segment_id", "partition", "block_id"]],
        on="road_segment_id",
        how="left",
    )

    # Dự đoán xác suất ngập từ mô hình Deep MLP cho toàn bộ 100% đoạn đường trong AOI
    scored = predictor.predict_dataframe(merged_in)
    scored = compute_local_basin_features(scored)

    # Ghép điểm MCDA vật lý
    full_df = scored.merge(
        df_mcda[["road_segment_id", "score", "mcda_eligible", "score_reason"]],
        on="road_segment_id",
        how="left",
    )

    # Ghép tên đường, cấp đường và chiều dài từ road_segments.gpkg
    roads_gpkg = interim_dir / "road_segments.gpkg"
    roads_gdf = None
    if roads_gpkg.is_file():
        roads_gdf = gpd.read_file(roads_gpkg)
        meta_cols = [
            c
            for c in (
                "road_segment_id",
                "osm_way_id",
                "road_name",
                "highway_type",
                "length_m",
            )
            if c in roads_gdf.columns
        ]
        full_df = full_df.merge(
            roads_gdf[meta_cols],
            on="road_segment_id",
            how="left",
            suffixes=("", "_osm"),
        )

    if "length_m" not in full_df.columns:
        full_df["length_m"] = (
            full_df["road_length_m"]
            if "road_length_m" in full_df.columns
            else 150.0
        )

    # Tính xác suất Logistic Regression + Local Basin (L2 - 11 tham số) để kết hợp bộ ba Ensemble
    confirmed_path = processed_dir / "confirmed_event_segments.parquet"
    feat_cols = [c for c in FEATURE_COLS if c in full_df.columns]
    local_feat_cols = [*feat_cols, "block_rel_elev_z", "local_pocket_index"]
    if confirmed_path.is_file():
        df_confirmed = pd.read_parquet(confirmed_path)
        pos_ids = set(df_confirmed["road_segment_id"].dropna().unique())
        full_df["_label"] = full_df["road_segment_id"].isin(pos_ids).astype(np.int32)
        valid_lr = (
            full_df[local_feat_cols].notna().all(axis=1)
            & full_df.get("score_eligible", pd.Series(True, index=full_df.index)).fillna(True)
        )
        ref_pool = full_df.loc[valid_lr & (full_df["partition"] == "reference")]
        if (ref_pool["_label"] == 1).sum() < 10:
            ref_pool = full_df.loc[valid_lr]
        pos_sub = ref_pool[ref_pool["_label"] == 1]
        bg_sub = ref_pool[ref_pool["_label"] == 0]
        if len(pos_sub) > 0 and len(bg_sub) > 0:
            rng = np.random.default_rng(42)
            n_bg = min(len(bg_sub), len(pos_sub) * 3)
            tr_lr = pd.concat(
                [pos_sub, bg_sub.iloc[rng.choice(len(bg_sub), size=n_bg, replace=False)]],
                ignore_index=True,
            ).sample(frac=1.0, random_state=42)
            scaler_lr = StandardScaler()
            x_tr_lr = scaler_lr.fit_transform(
                tr_lr[local_feat_cols].to_numpy(dtype=np.float32)
            )
            clf_lr = LogisticRegression(
                C=0.25, class_weight="balanced", max_iter=500, random_state=42
            )
            clf_lr.fit(x_tr_lr, tr_lr["_label"].to_numpy(dtype=np.int32))
            work_lr = pd.DataFrame(index=full_df.index)
            for col in local_feat_cols:
                med_v = float(pd.to_numeric(ref_pool[col], errors="coerce").median())
                work_lr[col] = (
                    pd.to_numeric(full_df[col], errors="coerce")
                    .fillna(med_v)
                    .astype(np.float32)
                )
            full_df["logreg_local_prob"] = clf_lr.predict_proba(
                scaler_lr.transform(work_lr[local_feat_cols].to_numpy(dtype=np.float32))
            )[:, 1]
        else:
            full_df["logreg_local_prob"] = full_df["flood_susceptibility_prob"]
        full_df = full_df.drop(columns=["_label"])
    else:
        full_df["logreg_local_prob"] = full_df["flood_susceptibility_prob"]

    # Quy đổi sang thang điểm 0 - 100 (MLP, LogReg-Local, MCDA và Tri-Model Ensemble)
    full_df["mlp_risk_score_100"] = (
        full_df["flood_susceptibility_prob"] * 100.0
    ).round(2)
    full_df["logreg_risk_score_100"] = (
        full_df["logreg_local_prob"] * 100.0
    ).round(2)
    full_df["mcda_score_100"] = pd.to_numeric(full_df["score"], errors="coerce").round(
        2
    )
    full_df["hybrid_risk_score_100"] = np.where(
        full_df["mcda_score_100"].notna(),
        0.45 * full_df["mlp_risk_score_100"]
        + 0.45 * full_df["logreg_risk_score_100"]
        + 0.10 * full_df["mcda_score_100"],
        0.50 * full_df["mlp_risk_score_100"] + 0.50 * full_df["logreg_risk_score_100"],
    ).round(2)

    # Với các đoạn cầu/hầm (score_eligible == False), đánh dấu rõ cấu trúc loại trừ
    eligible_mask = full_df.get(
        "score_eligible", pd.Series(True, index=full_df.index)
    ).fillna(True)
    full_df["risk_percentile"] = np.nan
    full_df.loc[eligible_mask, "risk_percentile"] = (
        full_df.loc[eligible_mask, "hybrid_risk_score_100"].rank(pct=True) * 100.0
    ).round(2)

    pct = full_df["risk_percentile"]
    full_df["risk_tier"] = np.where(
        ~eligible_mask,
        "Cầu/Hầm (Loại trừ cấu trúc)",
        np.where(
            pct >= 99.0,
            "1. Nguy kịch (Top 1%)",
            np.where(
                pct >= 95.0,
                "2. Rất cao (Top 5%)",
                np.where(
                    pct >= 90.0,
                    "3. Cao (Top 10%)",
                    np.where(
                        pct >= 80.0,
                        "4. Trung bình - Cao (Top 20%)",
                        np.where(pct >= 50.0, "5. Trung bình", "6. Thấp"),
                    ),
                ),
            ),
        ),
    )

    full_df = full_df.sort_values(
        ["score_eligible", "hybrid_risk_score_100", "mlp_risk_score_100"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    eligible_sorted = full_df.get(
        "score_eligible", pd.Series(True, index=full_df.index)
    ).fillna(True)
    full_df["risk_rank"] = np.where(
        eligible_sorted, np.arange(1, len(full_df) + 1), np.nan
    )

    # Lưu bảng Parquet đầy đủ và CSV Top 1.000 đoạn đường nguy cơ cao nhất
    parquet_out = output_dir / "full_study_area_road_risk_scores.parquet"
    csv_top1000 = output_dir / "top_1000_highest_risk_roads.csv"
    full_df.to_parquet(parquet_out, index=False)
    full_df.loc[eligible_sorted].head(1000).to_csv(
        csv_top1000, index=False, encoding="utf-8-sig"
    )

    # Xuất bản đồ tĩnh full_study_area_risk_map.png (đã căn chỉnh suptitle chống chồng chữ) & GeoJSON
    if roads_gdf is not None:
        aoi_gpkg = interim_dir / "study_area.gpkg"
        aoi_gdf = gpd.read_file(aoi_gpkg) if aoi_gpkg.is_file() else None
        map_gdf = roads_gdf[["road_segment_id", "geometry"]].merge(
            full_df[
                [
                    "road_segment_id",
                    "mlp_risk_score_100",
                    "hybrid_risk_score_100",
                    "risk_tier",
                    "score_eligible",
                ]
            ],
            on="road_segment_id",
            how="inner",
        )
        map_gdf = map_gdf[map_gdf["score_eligible"].fillna(True)]
        fig, axes = plt.subplots(1, 2, figsize=(18, 9.5))
        for ax, col_plot, title_str in (
            (
                axes[0],
                "mlp_risk_score_100",
                "1. Bản đồ Điểm Nguy cơ Ngập AI — Deep MLP (0–100)",
            ),
            (
                axes[1],
                "hybrid_risk_score_100",
                "2. Bản đồ Điểm Nguy cơ Ngập Tổng hợp — Tri-Model Ensemble (MLP + LogReg Local + MCDA)",
            ),
        ):
            if aoi_gdf is not None:
                aoi_gdf.boundary.plot(ax=ax, color="#172b4d", linewidth=1.2)
            map_gdf.plot(
                ax=ax,
                column=col_plot,
                cmap="RdYlBu_r",
                linewidth=0.35,
                legend=True,
                legend_kwds={"label": "Chỉ số Nguy cơ Ngập (0 – 100)", "shrink": 0.72},
            )
            ax.set_title(title_str, fontweight="bold", fontsize=11.5, pad=12)
            ax.set_axis_off()
        fig.suptitle(
            f"Bản đồ Phân bố Nguy cơ Ngập lụt trên Toàn bộ {len(full_df):,} Đoạn đường TP.HCM",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        fig.savefig(
            output_dir / "full_study_area_risk_map.png",
            dpi=160,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

        export_cols = [
            "road_segment_id",
            "road_name",
            "highway_type",
            "length_m",
            "mlp_risk_score_100",
            "logreg_risk_score_100",
            "mcda_score_100",
            "hybrid_risk_score_100",
            "risk_percentile",
            "risk_rank",
            "risk_tier",
            "elevation_median_m",
            "relative_elevation_m",
            "block_rel_elev_z",
            "local_pocket_index",
            "dist_water_m",
            "builtup_ratio_250m",
            "score_eligible",
        ]
        export_cols = [c for c in export_cols if c in full_df.columns]
        geo_merged = roads_gdf[["road_segment_id", "geometry"]].merge(
            full_df[export_cols], on="road_segment_id", how="inner"
        )
        geo_wgs84 = geo_merged.to_crs("EPSG:4326")
        midpoints = geo_wgs84.geometry.interpolate(0.5, normalized=True)
        geo_wgs84["lon"] = midpoints.x.round(6)
        geo_wgs84["lat"] = midpoints.y.round(6)

        # 1. GeoJSON Top 10% đoạn đường nguy cơ cao nhất (tối ưu cho Web Map)
        top10_gdf = geo_wgs84[
            geo_wgs84["score_eligible"].fillna(True)
            & (geo_wgs84["risk_percentile"] >= 90.0)
        ]
        top10_gdf.to_file(
            output_dir / "hcmc_road_flood_risk_top10pct.geojson",
            driver="GeoJSON",
            coordinate_precision=6,
        )

        # 2. GeoJSON toàn bộ mạng lưới trục đường chính
        major_types = [
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
        ]
        major_gdf = geo_wgs84[geo_wgs84["highway_type"].isin(major_types)]
        major_gdf.to_file(
            output_dir / "hcmc_road_flood_risk_major_roads.geojson",
            driver="GeoJSON",
            coordinate_precision=6,
        )

        # 3. GeoJSON toàn bộ 100% các đoạn đường trong phạm vi nghiên cứu
        geo_wgs84.to_file(
            output_dir / "hcmc_road_flood_risk_full.geojson",
            driver="GeoJSON",
            coordinate_precision=6,
        )

    return full_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    df = score_entire_study_area(
        args.processed_dir, args.interim_dir, args.bundle_dir, args.output_dir
    )
    print(f"Đã chấm điểm xong {len(df):,} đoạn đường trong phạm vi nghiên cứu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
