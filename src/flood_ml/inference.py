"""Bộ nạp mô hình (Model + Preprocessing + Config) dự đoán nguy cơ ngập đường bộ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional in CPU-only CI environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def compute_local_basin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung đặc trưng địa hình cục bộ theo ô lưới 5km×5km (block_id) để nhận diện túi ngập ở vùng địa hình cao."""
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

    if "block_id" in out.columns and out["block_id"].notna().any() and len(out) > 1:
        grp = out.assign(
            _elev=elev,
            _rel_elev=rel_elev,
        ).groupby("block_id", dropna=False)
        block_med = grp["_elev"].transform("median")
        block_iqr = (
            grp["_elev"]
            .transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
            .clip(lower=0.5)
        )
        out["block_rel_elev_z"] = ((elev - block_med) / block_iqr).astype(np.float32)
        block_low_elev_pct = grp["_elev"].transform(lambda s: (-s).rank(pct=True))
        block_dep_pct = grp["_rel_elev"].transform(lambda s: (-s).rank(pct=True))
        builtup_pct = builtup.rank(pct=True)
        out["local_pocket_index"] = (
            0.45 * block_dep_pct + 0.35 * block_low_elev_pct + 0.20 * builtup_pct
        ).astype(np.float32)
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
            }

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
        enriched = (
            compute_local_basin_features(df)
            if {"block_rel_elev_z", "local_pocket_index"} & set(self.feature_columns)
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
        enriched = compute_local_basin_features(df)
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
