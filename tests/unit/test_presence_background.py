from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from flood_ml.training.presence_background import (
    fit_track_b,
    score_track_b,
)
from flood_ml.training.sampling import FEATURE_COLUMNS, BackgroundDraw


def draw():
    values = np.array(
        [
            [3.0, 4.0, 0.5, 1.0, 0.2],
            [4.0, 5.0, 0.8, 2.0, 0.3],
            [5.0, 6.0, 1.0, 3.0, 0.4],
            [1.0, 1.0, 2.5, 7.0, 0.8],
            [2.0, 2.0, 2.0, 8.0, 0.7],
            [0.0, 3.0, 3.0, 9.0, 0.9],
        ]
    )
    return BackgroundDraw(
        model_id="track_b_r10_s42",
        background_ratio=10,
        seed=42,
        X=pd.DataFrame(values, columns=FEATURE_COLUMNS),
        y=np.array([1, 1, 1, 0, 0, 0], dtype=np.int8),
        road_ids=tuple(f"r{i}" for i in range(6)),
        manifest={"positive_count": 3, "background_count": 3},
    )


def score_frame():
    values = np.array(
        [
            [3.5, 4.5, 0.7, 2.0, 0.25],
            [1.5, 1.5, 2.3, 7.5, 0.75],
            [4.0, 5.0, 0.8, 2.0, 0.3],
            [1.0, 1.0, 2.5, 7.0, 0.8],
            [2.0, 2.0, 2.0, 8.0, 0.7],
            [3.0, 4.0, 0.5, 1.0, 0.2],
        ]
    )
    return pd.DataFrame(
        {
            "road_segment_id": [
                "h1",
                "h2",
                "bad_nan",
                "bad_inf",
                "bad_struct",
                "guard",
            ],
            **{
                column: values[:, index] for index, column in enumerate(FEATURE_COLUMNS)
            },
            "score_eligible": [True, True, True, True, False, True],
            "partition": [
                "holdout",
                "holdout",
                "holdout",
                "holdout",
                "holdout",
                "guard",
            ],
        }
    ).assign(
        elevation_median_m=lambda frame: frame.elevation_median_m.mask(
            frame.road_segment_id.eq("bad_nan"), np.nan
        ),
        dist_water_m=lambda frame: frame.dist_water_m.mask(
            frame.road_segment_id.eq("bad_inf"), np.inf
        ),
    )


def test_fit_track_b_uses_fixed_scaled_logistic_pipeline():
    fitted = fit_track_b(draw(), C=0.25, max_iter=321)

    assert list(fitted.estimator.named_steps) == ["scale", "model"]
    assert isinstance(fitted.estimator.named_steps["scale"], StandardScaler)
    classifier = fitted.estimator.named_steps["model"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.C == 0.25
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 321
    assert fitted.model_id == "track_b_r10_s42"
    assert fitted.background_ratio == 10
    assert fitted.seed == 42


def test_scaler_is_fit_only_on_background_draw():
    training = draw()
    fitted = fit_track_b(training)
    frame = score_frame()
    frame.loc[frame.road_segment_id.eq("h1"), FEATURE_COLUMNS] = 1e12

    score_track_b([fitted], frame)

    np.testing.assert_allclose(
        fitted.estimator.named_steps["scale"].mean_,
        training.X.mean().to_numpy(),
    )


def test_score_track_b_uses_decision_function(monkeypatch):
    fitted = fit_track_b(draw())
    classifier = fitted.estimator.named_steps["model"]
    monkeypatch.setattr(
        classifier,
        "predict_proba",
        lambda *_args, **_kwargs: pytest.fail("predict_proba must not be called"),
    )

    scores = score_track_b([fitted], score_frame())

    assert np.isfinite(scores.relative_score).all()
    assert scores.road_segment_id.tolist() == ["h1", "h2"]


def test_score_track_b_excludes_missing_nonfinite_or_ineligible_rows():
    fitted = fit_track_b(draw())

    scores = score_track_b([fitted], score_frame())

    assert set(scores.road_segment_id) == {"h1", "h2"}
    assert not {"bad_nan", "bad_inf", "bad_struct", "guard"}.intersection(
        scores.road_segment_id
    )


def test_score_track_b_returns_empty_schema_when_no_rows_are_eligible():
    fitted = fit_track_b(draw())
    features = score_frame().assign(score_eligible=False)

    scores = score_track_b([fitted], features)

    assert scores.empty
    assert scores.columns.tolist() == [
        "road_segment_id",
        "model_id",
        "background_ratio",
        "seed",
        "relative_score",
        "partition",
    ]


def test_scores_are_long_form_unique_and_repeatable():
    first_model = fit_track_b(draw())
    second_draw = replace(draw(), model_id="track_b_r10_s43", seed=43)
    second_model = fit_track_b(second_draw)
    features = score_frame()

    first = score_track_b([first_model, second_model], features)
    second = score_track_b([first_model, second_model], features)

    assert first.columns.tolist() == [
        "road_segment_id",
        "model_id",
        "background_ratio",
        "seed",
        "relative_score",
        "partition",
    ]
    assert first[["road_segment_id", "model_id"]].duplicated().sum() == 0
    assert set(first.road_segment_id) == {"h1", "h2"}
    assert set(first.partition) == {"holdout"}
    assert first.to_dict("records") == second.to_dict("records")
    assert first.road_segment_id.tolist() == ["h1", "h1", "h2", "h2"]
