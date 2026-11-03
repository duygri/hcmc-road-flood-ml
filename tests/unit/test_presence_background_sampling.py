import numpy as np
import pandas as pd

from flood_ml.training.sampling import FEATURE_COLUMNS, build_background_draws


def case(n_events=20, unknowns=40):
    positive = [f"p{i}" for i in range(n_events)]
    unknown = [f"u{i}" for i in range(unknowns)]
    negatives = ["verified"]
    ids = positive + unknown + negatives + ["hold", "guard", "excluded"]
    labels = pd.DataFrame(
        {
            "road_segment_id": ids,
            "label_status": ["positive"] * len(positive)
            + ["unknown"] * len(unknown)
            + ["verified_negative", "unknown", "unknown", "unknown"],
        }
    )
    features = pd.DataFrame(
        {
            "road_segment_id": ids,
            **{col: np.arange(len(ids), dtype=float) for col in FEATURE_COLUMNS},
        }
    )
    events = pd.DataFrame(
        {
            "road_segment_id": positive,
            "event_group_id": [f"e{i}" for i in range(n_events)],
            "location_group_id": [f"l{i}" for i in range(n_events)],
            "evidence": "review",
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-10-01T00:00:00Z",
            "event_evidence": "event review",
            "event_reviewed_by": "reviewer",
            "event_reviewed_at": "2026-10-01T00:00:00Z",
        }
    )
    partitions = pd.DataFrame(
        {
            "road_segment_id": ids,
            "block_id": [f"b{i % 4}" for i in range(len(ids))],
            "partition": ["reference"] * (len(ids) - 3)
            + ["holdout", "guard", "excluded"],
        }
    )
    config = {
        "primary_background_ratio": 1,
        "sensitivity_background_ratios": [2],
        "seeds": list(range(42, 62)),
    }
    return features, labels, events, partitions, config


def test_fit_gate_requires_twenty_feature_eligible_independent_reference_events():
    args = case(n_events=19)
    result = build_background_draws(*args)
    assert result.status == "pending_positive_review"
    assert result.gate["reviewed_reference_events"] == 19
    assert result.draws == ()


def test_background_draws_use_only_reviewed_reference_positives_and_unknowns():
    args = case()
    result = build_background_draws(*args)
    assert result.status == "ready"
    for draw in result.draws:
        positives = {draw.road_ids[i] for i, y in enumerate(draw.y) if y == 1}
        backgrounds = {draw.road_ids[i] for i, y in enumerate(draw.y) if y == 0}
        assert positives == {f"p{i}" for i in range(20)}
        assert backgrounds <= {f"u{i}" for i in range(40)}


def test_background_draws_exclude_verified_negative_and_nonreference_rows():
    result = build_background_draws(*case())
    for draw in result.draws:
        assert not {"verified", "hold", "guard", "excluded"}.intersection(draw.road_ids)


def test_structurally_ineligible_rows_are_excluded():
    args = case()
    args[0]["score_eligible"] = True
    args[0].loc[args[0].road_segment_id.eq("u0"), "score_eligible"] = False
    result = build_background_draws(*args)
    assert all("u0" not in draw.road_ids for draw in result.draws)
    assert {item["reason"] for item in result.gate["background_exclusions"]} >= {
        "structurally_ineligible"
    }


def test_incomplete_or_nonfinite_training_features_are_excluded():
    args = case()
    args[0].loc[args[0].road_segment_id.eq("p0"), FEATURE_COLUMNS[0]] = np.nan
    args[0].loc[args[0].road_segment_id.eq("u0"), FEATURE_COLUMNS[1]] = np.inf
    result = build_background_draws(*args)
    assert "p0" in {x["road_segment_id"] for x in result.gate["positive_exclusions"]}
    assert "u0" in {x["road_segment_id"] for x in result.gate["background_exclusions"]}


def test_draw_X_is_exact_allowlist_and_labels_are_unchanged():
    args = case()
    before = args[1].copy(deep=True)
    result = build_background_draws(*args)
    assert all(draw.X.columns.tolist() == FEATURE_COLUMNS for draw in result.draws)
    pd.testing.assert_frame_equal(args[1], before)


def test_background_draws_repeat_by_seed_without_replacement():
    args = case()
    first = build_background_draws(*args)
    second = build_background_draws(*args)
    assert [(d.model_id, d.road_ids) for d in first.draws] == [
        (d.model_id, d.road_ids) for d in second.draws
    ]
    for draw in first.draws:
        backgrounds = [rid for rid, y in zip(draw.road_ids, draw.y) if y == 0]
        assert len(backgrounds) == len(set(backgrounds))


def test_small_background_pool_is_audited():
    args = case(unknowns=10)
    result = build_background_draws(*args)
    assert result.status == "background_pool_insufficient"
    assert result.draws == ()
    assert result.failures
