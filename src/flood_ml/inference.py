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
        work = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
            else:
                series = pd.Series(np.nan, index=df.index, dtype=np.float32)
            fill_val = float(self.medians.get(col, 0.0))
            work[col] = series.fillna(fill_val).astype(np.float32)
        x_raw = work[self.feature_columns].to_numpy(dtype=np.float32)
        return (x_raw - self.mean) / np.where(self.scale == 0, 1.0, self.scale)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về bảng kết quả kèm xác suất nhạy cảm ngập và phân tầng nguy cơ."""
        x_scaled = self.preprocess(df)
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
        out = df.copy()
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
