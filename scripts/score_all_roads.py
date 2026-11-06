"""Tính điểm nguy cơ ngập lụt cho toàn bộ 100% đoạn đường trong AOI TP.HCM."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from flood_ml.inference import FloodSusceptibilityPredictor


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

    # Dự đoán xác suất ngập từ mô hình Deep MLP cho toàn bộ 100% đoạn đường trong AOI
    scored = predictor.predict_dataframe(df_feat)

    # Ghép phân vùng không gian và điểm MCDA vật lý
    full_df = scored.merge(
        df_part[["road_segment_id", "partition", "block_id"]],
        on="road_segment_id",
        how="left",
    ).merge(
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

    # Quy đổi sang thang điểm 0 - 100
    full_df["mlp_risk_score_100"] = (
        full_df["flood_susceptibility_prob"] * 100.0
    ).round(2)
    full_df["mcda_score_100"] = pd.to_numeric(full_df["score"], errors="coerce").round(
        2
    )
    full_df["hybrid_risk_score_100"] = np.where(
        full_df["mcda_score_100"].notna(),
        0.5 * full_df["mlp_risk_score_100"] + 0.5 * full_df["mcda_score_100"],
        full_df["mlp_risk_score_100"],
    ).round(2)

    # Với các đoạn cầu/hầm (score_eligible == False), đánh dấu rõ cấu trúc loại trừ
    eligible_mask = full_df.get(
        "score_eligible", pd.Series(True, index=full_df.index)
    ).fillna(True)
    full_df["risk_percentile"] = np.nan
    full_df.loc[eligible_mask, "risk_percentile"] = (
        full_df.loc[eligible_mask, "mlp_risk_score_100"].rank(pct=True) * 100.0
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
        ["score_eligible", "mlp_risk_score_100", "hybrid_risk_score_100"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    full_df["risk_rank"] = np.where(
        eligible_mask, np.arange(1, len(full_df) + 1), np.nan
    )

    # Lưu bảng Parquet đầy đủ và CSV Top 1.000 đoạn đường nguy cơ cao nhất
    parquet_out = output_dir / "full_study_area_road_risk_scores.parquet"
    csv_top1000 = output_dir / "top_1000_highest_risk_roads.csv"
    full_df.to_parquet(parquet_out, index=False)
    full_df.loc[eligible_mask].head(1000).to_csv(
        csv_top1000, index=False, encoding="utf-8-sig"
    )

    # Xuất GeoJSON chuẩn Web GIS (WGS84 - EPSG:4326) ghép điểm nguy cơ với tọa độ đường
    if roads_gdf is not None:
        export_cols = [
            "road_segment_id",
            "road_name",
            "highway_type",
            "length_m",
            "mlp_risk_score_100",
            "mcda_score_100",
            "hybrid_risk_score_100",
            "risk_percentile",
            "risk_rank",
            "risk_tier",
            "elevation_median_m",
            "relative_elevation_m",
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
