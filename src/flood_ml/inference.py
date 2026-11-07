"""Bộ nạp mô hình (Model + Preprocessing + Config) dự đoán nguy cơ ngập đường bộ."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional in CPU-only CI environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def fit_local_basin_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Đóng băng thống kê địa hình theo ô lưới 5km×5km và phân vị built-up từ tập Reference."""
    if df.empty:
        return {"global": {}, "blocks": {}}
    ref = (
        df.loc[df["partition"] == "reference"]
        if "partition" in df.columns and (df["partition"] == "reference").any()
        else df
    )
    elev_ref = pd.to_numeric(
        ref.get("elevation_median_m", pd.Series(0.0, index=ref.index)),
        errors="coerce",
    ).dropna()
    rel_ref = pd.to_numeric(
        ref.get("relative_elevation_m", pd.Series(0.0, index=ref.index)),
        errors="coerce",
    ).dropna()
    built_ref = pd.to_numeric(
        ref.get("builtup_ratio_250m", pd.Series(0.0, index=ref.index)),
        errors="coerce",
    ).dropna()

    q_grid = np.linspace(0.0, 1.0, 101)
    global_stats = {
        "elev_median": float(elev_ref.median()) if len(elev_ref) else 0.0,
        "elev_iqr": float(
            max(
                (elev_ref.quantile(0.75) - elev_ref.quantile(0.25))
                if len(elev_ref)
                else 0.5,
                0.5,
            )
        ),
        "neg_elev_quantiles": (
            [float(x) for x in np.quantile(-elev_ref.to_numpy(dtype=np.float64), q_grid)]
            if len(elev_ref)
            else [0.0] * 101
        ),
        "neg_rel_elev_quantiles": (
            [float(x) for x in np.quantile(-rel_ref.to_numpy(dtype=np.float64), q_grid)]
            if len(rel_ref)
            else [0.0] * 101
        ),
        "builtup_quantiles": (
            [float(x) for x in np.quantile(built_ref.to_numpy(dtype=np.float64), q_grid)]
            if len(built_ref)
            else [0.0] * 101
        ),
    }

    blocks: dict[str, dict[str, Any]] = {}
    if "block_id" in df.columns:
        for block_id, grp in df.groupby("block_id", dropna=True):
            b_elev = pd.to_numeric(grp.get("elevation_median_m"), errors="coerce").dropna()
            b_rel = pd.to_numeric(grp.get("relative_elevation_m"), errors="coerce").dropna()
            if len(b_elev) == 0:
                continue
            iqr = float(b_elev.quantile(0.75) - b_elev.quantile(0.25))
            blocks[str(block_id)] = {
                "elev_median": float(b_elev.median()),
                "elev_iqr": max(iqr, 0.5),
                "neg_elev_quantiles": [
                    float(x)
                    for x in np.quantile(-b_elev.to_numpy(dtype=np.float64), q_grid)
                ],
                "neg_rel_elev_quantiles": (
                    [
                        float(x)
                        for x in np.quantile(-b_rel.to_numpy(dtype=np.float64), q_grid)
                    ]
                    if len(b_rel)
                    else global_stats["neg_rel_elev_quantiles"]
                ),
            }
    return {"global": global_stats, "blocks": blocks}


def _empirical_pct(values: pd.Series, ref_quantiles: list[float] | np.ndarray) -> pd.Series:
    q_vals = np.asarray(ref_quantiles, dtype=np.float64)
    if len(q_vals) != 101 or np.allclose(q_vals[0], q_vals[-1]):
        return pd.Series(0.5, index=values.index, dtype=np.float32)
    q_grid = np.linspace(0.0, 1.0, 101)
    arr = values.to_numpy(dtype=np.float64)
    return pd.Series(
        np.interp(arr, q_vals, q_grid, left=0.0, right=1.0),
        index=values.index,
        dtype=np.float32,
    )


def compute_local_basin_features(
    df: pd.DataFrame, basin_stats: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Bổ sung đặc trưng địa hình cục bộ theo ô lưới 5km×5km (block_id) chống rò rỉ giữa các phân vùng."""
    out = df.copy()
    elev = pd.to_numeric(
        out.get("elevation_median_m", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    rel_elev = pd.to_numeric(
        out.get("relative_elevation_m", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    builtup = pd.to_numeric(
        out.get("builtup_ratio_250m", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)

    stats = basin_stats if isinstance(basin_stats, dict) and basin_stats.get("global") else None
    if stats is None and len(out) > 1:
        stats = fit_local_basin_stats(out)

    if stats is not None:
        g_stats = stats["global"]
        b_map = dict(stats.get("blocks", {}))
        # Với các block chưa có trong tập Reference (ví dụ block thuộc Holdout khi chạy theo lô), bổ sung thống kê nội khối
        if "block_id" in out.columns and out["block_id"].notna().any() and len(out) > 1:
            q_grid = np.linspace(0.0, 1.0, 101)
            for bid_val, grp_b in out.assign(_elev=elev, _rel_elev=rel_elev).groupby("block_id", dropna=True):
                bid_str = str(bid_val)
                if bid_str not in b_map and len(grp_b) > 1:
                    ev_b = grp_b["_elev"]
                    rev_b = grp_b["_rel_elev"]
                    b_map[bid_str] = {
                        "elev_median": float(ev_b.median()),
                        "elev_iqr": float(max(ev_b.quantile(0.75) - ev_b.quantile(0.25), 0.5)),
                        "neg_elev_quantiles": [float(x) for x in np.quantile(-ev_b, q_grid)],
                        "neg_rel_elev_quantiles": [float(x) for x in np.quantile(-rev_b, q_grid)],
                    }
        z_vals, pocket_vals = [], []
        built_pct_series = _empirical_pct(builtup, g_stats["builtup_quantiles"])
        for idx in out.index:
            bid = (
                str(out.at[idx, "block_id"])
                if "block_id" in out.columns and pd.notna(out.at[idx, "block_id"])
                else ""
            )
            b_info = b_map.get(bid, g_stats)
            med = float(b_info.get("elev_median", g_stats.get("elev_median", 0.0)))
            iqr = max(float(b_info.get("elev_iqr", g_stats.get("elev_iqr", 0.5))), 0.5)
            ev = float(elev.at[idx])
            rev = float(rel_elev.at[idx])
            z_vals.append((ev - med) / iqr)
            low_el_p = float(
                _empirical_pct(
                    pd.Series([-ev]),
                    b_info.get("neg_elev_quantiles", g_stats["neg_elev_quantiles"]),
                ).iloc[0]
            )
            dep_p = float(
                _empirical_pct(
                    pd.Series([-rev]),
                    b_info.get(
                        "neg_rel_elev_quantiles", g_stats["neg_rel_elev_quantiles"]
                    ),
                ).iloc[0]
            )
            bp = float(built_pct_series.at[idx])
            pocket_vals.append(0.45 * dep_p + 0.35 * low_el_p + 0.20 * bp)
        out["block_rel_elev_z"] = pd.Series(z_vals, index=out.index, dtype=np.float32)
        out["local_pocket_index"] = pd.Series(
            pocket_vals, index=out.index, dtype=np.float32
        )
    else:
        out["block_rel_elev_z"] = np.float32(0.0)
        out["local_pocket_index"] = np.clip(
            0.5 * (1.0 / (1.0 + np.exp(rel_elev))) + 0.5 * builtup,
            0.0,
            1.0,
        ).astype(np.float32)
    return out


class FloodSusceptibilityNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Kiến trúc Residual Tabular MLP đã chuẩn hóa chống Overfitting."""

    def __init__(self, in_features: int, hidden_dim: int = 64):
        if nn is None:
            raise ImportError("PyTorch is required to instantiate FloodSusceptibilityNet")
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.35),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: Any) -> Any:
        h = self.input_layer(x)
        h = h + 0.5 * self.res_block(h)
        return self.head(h).squeeze(-1)


class FloodSusceptibilityPredictor:
    """Nạp trọn bộ Model + Preprocessing Scaler + Cấu hình ngưỡng nguy cơ."""

    def __init__(self, bundle_dir: str | Path, device: str | None = None):
        if torch is None:
            raise ImportError(
                "PyTorch is required to run FloodSusceptibilityPredictor"
            )
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Đọc cấu hình tiền xử lý (Preprocessing) & Cấu hình mô hình (Model Config)
        prep_path = self.bundle_dir / "preprocessing" / "preprocessing_config.json"
        cfg_path = self.bundle_dir / "config" / "model_config.json"
        ckpt_path = self.bundle_dir / "model" / "best_model_a100.pt"
        if not ckpt_path.is_file():
            ckpt_path = self.bundle_dir / "best_model_a100.pt"

        try:
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=True
            )
        except (RuntimeError, ValueError, TypeError, pickle.UnpicklingError):
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )

        if prep_path.is_file():
            self.prep_config = json.loads(prep_path.read_text(encoding="utf-8"))
        else:
            self.prep_config = {
                "feature_columns": checkpoint["feature_columns"],
                "scaler_mean": checkpoint["scaler_mean"],
                "scaler_scale": checkpoint["scaler_scale"],
                "imputation_medians": checkpoint.get("imputation_medians", {}),
                "basin_stats": checkpoint.get("basin_stats"),
            }
        self.basin_stats: dict[str, Any] | None = self.prep_config.get(
            "basin_stats"
        ) or checkpoint.get("basin_stats")

        if cfg_path.is_file():
            self.model_config = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            self.model_config = {
                "architecture": {
                    "in_features": len(self.prep_config["feature_columns"]),
                    "hidden_dim": 64,
                },
                "thresholds": {"default": 0.5, "top_10_pct": 0.45, "top_5_pct": 0.52},
            }

        self.feature_columns: list[str] = list(self.prep_config["feature_columns"])
        self.mean = np.asarray(self.prep_config["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(self.prep_config["scaler_scale"], dtype=np.float32)
        self.medians: dict[str, float] = dict(
            self.prep_config.get("imputation_medians", {})
        )

        # 2. Khởi tạo và nạp trọng số mạng nơ-ron
        hidden_dim = int(
            self.model_config.get("architecture", {}).get("hidden_dim", 64)
        )
        self.model = FloodSusceptibilityNet(
            len(self.feature_columns), hidden_dim=hidden_dim
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Chuẩn hóa bảng đặc trưng đầu vào theo đúng thứ tự cột và StandardScaler."""
        needs_basin = bool(
            {"block_rel_elev_z", "local_pocket_index"} & set(self.feature_columns)
        ) and not {"block_rel_elev_z", "local_pocket_index"}.issubset(df.columns)
        enriched = (
            compute_local_basin_features(df, basin_stats=self.basin_stats)
            if needs_basin
            else df
        )
        work = pd.DataFrame(index=enriched.index)
        for col in self.feature_columns:
            if col in enriched.columns:
                series = pd.to_numeric(enriched[col], errors="coerce")
            else:
                series = pd.Series(np.nan, index=enriched.index, dtype=np.float32)
            fill_val = float(self.medians.get(col, 0.0))
            work[col] = series.fillna(fill_val).astype(np.float32)
        x_raw = work[self.feature_columns].to_numpy(dtype=np.float32)
        return (x_raw - self.mean) / np.where(self.scale == 0, 1.0, self.scale)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về bảng kết quả kèm xác suất nhạy cảm ngập và phân tầng nguy cơ."""
        enriched = (
            df
            if {"block_rel_elev_z", "local_pocket_index"}.issubset(df.columns)
            else compute_local_basin_features(df, basin_stats=self.basin_stats)
        )
        x_scaled = self.preprocess(enriched)
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_scaled).to(self.device)
            probs = torch.sigmoid(self.model(x_tensor)).cpu().numpy()

        thresholds = self.model_config.get("thresholds", {})
        t_high = float(thresholds.get("top_5_pct", 0.52))
        t_med = float(thresholds.get("top_10_pct", 0.45))
        t_low = float(thresholds.get("top_20_pct", 0.35))

        risk_levels = np.where(
            probs >= t_high,
            "Rất cao (Top 5%)",
            np.where(
                probs >= t_med,
                "Cao (Top 10%)",
                np.where(probs >= t_low, "Trung bình (Top 20%)", "Thấp"),
            ),
        )
        out = enriched.copy()
        out["flood_susceptibility_prob"] = probs
        out["risk_tier"] = risk_levels
        return out

    def predict_single(self, features: dict[str, Any]) -> dict[str, Any]:
        """Dự đoán cho 1 đoạn đường đơn lẻ từ dict các đặc trưng địa vật lý."""
        row_df = pd.DataFrame([features])
        res = self.predict_dataframe(row_df).iloc[0]
        return {
            "flood_susceptibility_prob": float(res["flood_susceptibility_prob"]),
            "risk_tier": str(res["risk_tier"]),
        }
