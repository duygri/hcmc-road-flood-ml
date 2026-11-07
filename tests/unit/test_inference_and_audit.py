"""Unit tests for local basin feature consistency, inference predictor, and spatial audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flood_ml.inference import (
    FloodSusceptibilityNet,
    FloodSusceptibilityPredictor,
    compute_local_basin_features,
    fit_local_basin_stats,
)


def _sample_roads_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 80
    partitions = ["reference"] * 40 + ["holdout"] * 40
    blocks = ["100:200"] * 20 + ["101:200"] * 20 + ["102:200"] * 40
    return pd.DataFrame(
        {
            "road_segment_id": [f"seg_{i:03d}" for i in range(n)],
            "partition": partitions,
            "block_id": blocks,
            "elevation_median_m": rng.uniform(1.0, 15.0, size=n).astype(np.float32),
            "elevation_p10_m": rng.uniform(0.5, 14.0, size=n).astype(np.float32),
            "relative_elevation_m": rng.uniform(-1.5, 1.5, size=n).astype(np.float32),
            "slope_mean_deg": rng.uniform(0.1, 4.0, size=n).astype(np.float32),
            "builtup_ratio_250m": rng.uniform(0.1, 0.95, size=n).astype(np.float32),
            "vegetation_ratio_250m": rng.uniform(0.05, 0.6, size=n).astype(np.float32),
            "dist_water_m": rng.uniform(10.0, 800.0, size=n).astype(np.float32),
            "water_area_ratio_250m": rng.uniform(0.0, 0.3, size=n).astype(np.float32),
            "length_m": rng.uniform(50.0, 300.0, size=n).astype(np.float32),
            "highway_type": ["primary" if i % 4 == 0 else "residential" for i in range(n)],
        }
    )


def test_local_basin_features_single_row_matches_batch_with_frozen_stats() -> None:
    df = _sample_roads_df()
    ref_df = df[df["partition"] == "reference"]
    basin_stats = fit_local_basin_stats(ref_df)

    batch_enriched = compute_local_basin_features(df, basin_stats=basin_stats)
    target_row = df.iloc[[5]].copy()
    single_enriched = compute_local_basin_features(target_row, basin_stats=basin_stats)

    np.testing.assert_allclose(
        single_enriched["block_rel_elev_z"].to_numpy(),
        batch_enriched.iloc[[5]]["block_rel_elev_z"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        single_enriched["local_pocket_index"].to_numpy(),
        batch_enriched.iloc[[5]]["local_pocket_index"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_basin_features_no_holdout_partition_leakage() -> None:
    df = _sample_roads_df()
    ref_df = df[df["partition"] == "reference"]
    basin_stats = fit_local_basin_stats(ref_df)

    ref_only_out = compute_local_basin_features(ref_df, basin_stats=basin_stats)

    # Perturb holdout builtup_ratio_250m drastically; reference features must stay identical
    df_perturbed = df.copy()
    df_perturbed.loc[df_perturbed["partition"] == "holdout", "builtup_ratio_250m"] = 999.0
    combined_out = compute_local_basin_features(df_perturbed, basin_stats=basin_stats)
    combined_ref = combined_out[combined_out["partition"] == "reference"]

    np.testing.assert_allclose(
        ref_only_out["local_pocket_index"].to_numpy(),
        combined_ref["local_pocket_index"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_predictor_single_vs_batch_consistency(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    df = _sample_roads_df()
    ref_df = df[df["partition"] == "reference"]
    basin_stats = fit_local_basin_stats(ref_df)
    ref_enriched = compute_local_basin_features(ref_df, basin_stats=basin_stats)

    feat_cols = [
        "elevation_median_m",
        "elevation_p10_m",
        "relative_elevation_m",
        "slope_mean_deg",
        "builtup_ratio_250m",
        "vegetation_ratio_250m",
        "dist_water_m",
        "water_area_ratio_250m",
        "block_rel_elev_z",
        "local_pocket_index",
    ]
    torch.manual_seed(7)
    net = FloodSusceptibilityNet(in_features=len(feat_cols), hidden_dim=16)

    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "model").mkdir(parents=True)
    (bundle_dir / "preprocessing").mkdir(parents=True)
    (bundle_dir / "config").mkdir(parents=True)

    mean_vals = [float(ref_enriched[c].mean()) for c in feat_cols]
    scale_vals = [float(ref_enriched[c].std() + 1e-3) for c in feat_cols]
    medians = {c: float(ref_enriched[c].median()) for c in feat_cols}

    prep_cfg = {
        "feature_columns": feat_cols,
        "scaler_mean": mean_vals,
        "scaler_scale": scale_vals,
        "imputation_medians": medians,
        "basin_stats": basin_stats,
    }
    model_cfg = {
        "architecture": {"in_features": len(feat_cols), "hidden_dim": 16},
        "thresholds": {"top_5_pct": 0.6, "top_10_pct": 0.5, "top_20_pct": 0.4},
    }
    (bundle_dir / "preprocessing" / "preprocessing_config.json").write_text(
        json.dumps(prep_cfg), encoding="utf-8"
    )
    (bundle_dir / "config" / "model_config.json").write_text(
        json.dumps(model_cfg), encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "feature_columns": feat_cols,
            "scaler_mean": mean_vals,
            "scaler_scale": scale_vals,
            "imputation_medians": medians,
            "basin_stats": basin_stats,
        },
        bundle_dir / "model" / "best_model_a100.pt",
    )

    predictor = FloodSusceptibilityPredictor(bundle_dir, device="cpu")
    batch_res = predictor.predict_dataframe(df)
    single_res = predictor.predict_single(df.iloc[3].to_dict())

    assert abs(float(batch_res.iloc[3]["flood_susceptibility_prob"]) - single_res["flood_susceptibility_prob"]) < 1e-5


def test_audit_ranking_spatial_bias_strict_partition_and_spearman(tmp_path: Path) -> None:
    df = _sample_roads_df()
    processed_dir = tmp_path / "processed"
    run_out_dir = tmp_path / "run_out"
    processed_dir.mkdir(parents=True)
    run_out_dir.mkdir(parents=True)

    df.to_parquet(processed_dir / "road_features.parquet", index=False)
    df[["road_segment_id", "partition", "block_id"]].to_parquet(
        processed_dir / "road_partitions.parquet", index=False
    )
    mcda_df = pd.DataFrame(
        {
            "road_segment_id": df["road_segment_id"],
            "score": (100.0 - df["elevation_median_m"] * 5.0).clip(0.0, 100.0),
        }
    )
    mcda_df.to_parquet(processed_dir / "road_susceptibility.parquet", index=False)

    # 4 positives in reference (< 10) and 4 positives in holdout: must NOT leak holdout into reference training
    confirmed_ids = ["seg_000", "seg_001", "seg_002", "seg_003", "seg_040", "seg_041", "seg_042", "seg_043"]
    pd.DataFrame({"road_segment_id": confirmed_ids}).to_parquet(
        processed_dir / "confirmed_event_segments.parquet", index=False
    )
    pd.DataFrame({"road_segment_id": confirmed_ids, "match_distance_m": [5.0] * len(confirmed_ids)}).to_parquet(
        processed_dir / "road_flood_matches.parquet", index=False
    )
    pred_df = pd.DataFrame(
        {
            "road_segment_id": df["road_segment_id"],
            "flood_susceptibility_prob": np.linspace(0.9, 0.1, len(df), dtype=np.float32),
        }
    )
    pred_df.to_parquet(run_out_dir / "a100_road_flood_predictions.parquet", index=False)

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "audit_ranking_spatial_bias.py"
    spec = importlib.util.spec_from_file_location("audit_ranking_spatial_bias", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    report = mod.run_audit(processed_dir, run_out_dir, "test_run")
    assert report["methodological_notes"]["partition_isolation"]["strict_reference_only_training"] is True
    assert report["methodological_notes"]["partition_isolation"]["holdout_fallback_used"] is False
    assert len(report["spatial_validation_by_block"]) == 1
    blk = report["spatial_validation_by_block"][0]
    assert "Tương quan Spearman (MLP vs MCDA)" in blk
    assert "Tương quan Spearman (Ensemble vs MCDA)" in blk
