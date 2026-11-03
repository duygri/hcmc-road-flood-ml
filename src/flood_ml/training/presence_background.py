"""Fit and score the exploratory presence-background road ranker."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flood_ml.baseline.mcda import _values_and_reasons
from flood_ml.config import PipelineError
from flood_ml.training.sampling import FEATURE_COLUMNS, BackgroundDraw

KEY = "road_segment_id"
SCORE_COLUMNS = [
    KEY,
    "model_id",
    "background_ratio",
    "seed",
    "relative_score",
    "partition",
]
FROZEN_PARTITIONS = {"reference", "holdout", "guard", "excluded"}


@dataclass(frozen=True)
class FittedTrackBModel:
    model_id: str
    background_ratio: int
    seed: int
    estimator: Pipeline
    manifest: dict


def fit_track_b(
    draw: BackgroundDraw,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
) -> FittedTrackBModel:
    """Fit a scaled logistic ranker using the single reference draw only."""
    if draw.X.columns.tolist() != FEATURE_COLUMNS:
        raise PipelineError("Background draw must contain the five approved features")
    values = draw.X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PipelineError("Background draw features must be finite")
    if len(draw.X) != len(draw.y) or set(np.asarray(draw.y).tolist()) != {0, 1}:
        raise PipelineError(
            "Background draw requires aligned presence and background rows"
        )
    if not np.isfinite(C) or C <= 0 or max_iter < 1:
        raise PipelineError("C must be positive and max_iter must be positive")

    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter),
            ),
        ]
    )
    estimator.fit(draw.X.loc[:, FEATURE_COLUMNS], draw.y)
    manifest = {
        **draw.manifest,
        "model_id": draw.model_id,
        "feature_columns": list(FEATURE_COLUMNS),
        "fit_method": "sampled_presence_background_logistic_regression",
        "fit_row_count": len(draw.X),
        "positive_count": int(np.count_nonzero(draw.y == 1)),
        "background_count": int(np.count_nonzero(draw.y == 0)),
        "C": float(C),
        "solver": "lbfgs",
        "max_iter": int(max_iter),
        "score_semantics": "relative_rank_score_not_probability",
    }
    return FittedTrackBModel(
        draw.model_id,
        draw.background_ratio,
        draw.seed,
        estimator,
        manifest,
    )


def score_track_b(
    models: Sequence[FittedTrackBModel], features: pd.DataFrame
) -> pd.DataFrame:
    """Return deterministic long-form decision scores for eligible roads."""
    if not isinstance(features, pd.DataFrame) or features.columns.duplicated().any():
        raise PipelineError("Features require a DataFrame with distinct columns")
    if KEY not in features:
        raise PipelineError("Features require road_segment_id")
    if "partition" not in features:
        raise PipelineError("Features require frozen partition")
    if (
        not features[KEY]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
        or not features[KEY].is_unique
    ):
        raise PipelineError("Features require unique nonblank road_segment_id")
    if not features.partition.isin(FROZEN_PARTITIONS).all():
        raise PipelineError("Features contain an invalid frozen partition")

    values, reasons = _values_and_reasons(features)
    eligible = (
        values.notna().all(axis=1)
        & reasons.eq("")
        & features.partition.isin(["reference", "holdout"])
    )
    selected = features.loc[eligible]
    matrix = values.loc[eligible, FEATURE_COLUMNS]
    rows = []
    for fitted in models:
        scores = fitted.estimator.decision_function(matrix)
        rows.extend(
            {
                KEY: road_id,
                "model_id": fitted.model_id,
                "background_ratio": fitted.background_ratio,
                "seed": fitted.seed,
                "relative_score": float(score),
                "partition": partition,
            }
            for road_id, partition, score in zip(
                selected[KEY], selected.partition, scores, strict=True
            )
        )
    return pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values(
        [KEY, "model_id"], kind="stable", ignore_index=True
    )
