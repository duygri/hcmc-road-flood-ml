"""Reviewed reference positives and deterministic unknown background draws."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from flood_ml.config import PipelineError
from flood_ml.evaluation.capture import evaluation_gate
from flood_ml.labeling.matching import require_ids

KEY = "road_segment_id"
FEATURE_COLUMNS = [
    "elevation_median_m",
    "relative_elevation_m",
    "slope_mean_deg",
    "dist_water_m",
    "builtup_ratio_250m",
]
DEFAULT_PRIMARY_RATIO = 10
DEFAULT_SENSITIVITY_RATIOS = (5, 20)
DEFAULT_SEEDS = tuple(range(42, 62))


@dataclass(frozen=True)
class BackgroundDraw:
    model_id: str
    background_ratio: int
    seed: int
    X: pd.DataFrame
    y: np.ndarray
    road_ids: tuple[str, ...]
    manifest: dict


@dataclass(frozen=True)
class SamplingResult:
    status: Literal["ready", "pending_positive_review", "background_pool_insufficient"]
    gate: dict
    draws: tuple[BackgroundDraw, ...]
    failures: tuple[dict, ...]


def _valid_text(value):
    return isinstance(value, str) and bool(value.strip())


def _settings(config):
    if not isinstance(config, dict):
        raise PipelineError("Sampling configuration must be a mapping")
    section = config.get("track_b", config.get("sampling", config))
    primary = section.get("primary_background_ratio", DEFAULT_PRIMARY_RATIO)
    sensitivities = section.get(
        "sensitivity_background_ratios", DEFAULT_SENSITIVITY_RATIOS
    )
    if "seeds" in section:
        seeds = section["seeds"]
    else:
        seed_start = int(section.get("seed_start", DEFAULT_SEEDS[0]))
        repetitions = int(section.get("seed_repetitions", len(DEFAULT_SEEDS)))
        seeds = range(seed_start, seed_start + repetitions)
    try:
        primary = int(primary)
        sensitivities = tuple(int(value) for value in sensitivities)
        seeds = tuple(int(value) for value in seeds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineError("Sampling ratios and seeds must be integers") from exc
    if (
        primary < 1
        or any(value < 1 for value in sensitivities)
        or len(set(seeds)) != len(seeds)
    ):
        raise PipelineError("Sampling ratios must be positive and seeds unique")
    if not seeds or len(seeds) != 20:
        raise PipelineError("Sampling requires exactly 20 deterministic seeds")
    return primary, sensitivities, seeds


def _validate_inputs(features, labels, events, partitions):
    for name, frame in (
        ("features", features),
        ("labels", labels),
        ("partitions", partitions),
    ):
        require_ids(frame, KEY)
        if frame.columns.duplicated().any():
            raise PipelineError(f"{name} columns must be unique")
    if not set(FEATURE_COLUMNS).issubset(features.columns):
        raise PipelineError("Feature table is missing approved Track B features")
    if (
        "score_eligible" in features
        and not features.score_eligible.map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all()
    ):
        raise PipelineError("score_eligible must contain booleans")
    if "label_status" not in labels:
        raise PipelineError("Labels require label_status")
    if not {"partition", "block_id"}.issubset(partitions.columns):
        raise PipelineError("Partitions require partition and block_id")
    if not partitions.partition.isin(
        ["reference", "holdout", "guard", "excluded"]
    ).all():
        raise PipelineError("Invalid frozen partition value")
    if not partitions.block_id.map(_valid_text).all():
        raise PipelineError("Partitions require nonblank block_id")
    if set(features[KEY]) != set(labels[KEY]) or set(features[KEY]) != set(
        partitions[KEY]
    ):
        raise PipelineError(
            "Features, labels and partitions must cover the same road population"
        )


def _allocations(counts, total):
    """Allocate a total proportionally using largest remainders, block ID ties."""
    available = sum(counts.values())
    if total > available:
        return None
    if not available:
        return {block: 0 for block in counts}
    quotas = {block: total * count / available for block, count in counts.items()}
    result = {block: int(np.floor(value)) for block, value in quotas.items()}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda block: (-(quotas[block] - result[block]), block))
    for block in order[:remaining]:
        result[block] += 1
    return result


def build_background_draws(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    confirmed_event_segments: pd.DataFrame,
    partitions: pd.DataFrame,
    config: dict,
) -> SamplingResult:
    """Build 20 seeded case-control draws; sampled zeroes remain in memory only."""
    _validate_inputs(features, labels, confirmed_event_segments, partitions)
    primary, sensitivities, seeds = _settings(config)
    gate = evaluation_gate(confirmed_event_segments, partitions)
    label_map = labels.set_index(KEY).label_status
    part_map = partitions.set_index(KEY)
    numeric = features.set_index(KEY)[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    complete_ids = set(numeric.index[np.isfinite(numeric.to_numpy()).all(axis=1)])
    structurally_eligible = (
        set(features.loc[features.score_eligible, KEY])
        if "score_eligible" in features
        else set(features[KEY])
    )
    complete_ids &= structurally_eligible

    # Gate assignments are independent, reviewed event groups. A group qualifies
    # only when it contributes at least one complete-feature positive road.
    eligible_events = []
    positive_exclusions = []
    for assignment in gate["reviewed_events"]:
        group = confirmed_event_segments.loc[
            confirmed_event_segments.event_group_id.eq(assignment["event_group_id"])
        ]
        candidates = sorted(set(group[KEY]) & complete_ids)
        candidates = [
            rid
            for rid in candidates
            if label_map.get(rid) == "positive"
            and part_map.loc[rid, "partition"] == "reference"
        ]
        if candidates:
            eligible_events.append(assignment["event_group_id"])
        else:
            group_positives = [
                rid
                for rid in set(group[KEY])
                if label_map.get(rid) == "positive"
                and part_map.loc[rid, "partition"] == "reference"
            ]
            positive_exclusions.extend(
                {
                    "event_group_id": assignment["event_group_id"],
                    "road_segment_id": rid,
                    "reason": "structurally_ineligible"
                    if rid not in structurally_eligible
                    else "incomplete_or_nonfinite_features"
                    if rid not in complete_ids
                    else "no_complete_feature_eligible_positive_road",
                }
                for rid in group_positives
            )
    gate = {
        **gate,
        "feature_eligible_reference_events": len(eligible_events),
        "positive_exclusions": positive_exclusions,
        "background_exclusions": [],
    }
    unknown_ids = [
        rid
        for rid in features[KEY]
        if label_map.get(rid) == "unknown"
        and part_map.loc[rid, "partition"] == "reference"
    ]
    finite_unknowns = [rid for rid in unknown_ids if rid in complete_ids]
    gate["background_exclusions"] = [
        {
            "road_segment_id": rid,
            "reason": "structurally_ineligible"
            if rid not in structurally_eligible
            else "incomplete_or_nonfinite_features",
        }
        for rid in unknown_ids
        if rid not in complete_ids
    ]
    positive_ids = sorted(
        {
            rid
            for event_id in eligible_events
            for rid in confirmed_event_segments.loc[
                confirmed_event_segments.event_group_id.eq(event_id), KEY
            ]
            if rid in complete_ids
            and label_map.get(rid) == "positive"
            and part_map.loc[rid, "partition"] == "reference"
        }
    )
    if len(eligible_events) < 20:
        return SamplingResult("pending_positive_review", gate, (), ())

    candidate_by_block = {}
    for rid in finite_unknowns:
        candidate_by_block.setdefault(part_map.loc[rid, "block_id"], []).append(rid)

    draws, failures = [], []
    for ratio in (primary, *sensitivities):
        for seed in seeds:
            requested = len(positive_ids) * ratio
            counts = {
                block: len(ids) for block, ids in sorted(candidate_by_block.items())
            }
            allocation = _allocations(counts, requested)
            model_id = f"track_b_r{ratio}_s{seed}"
            if allocation is None:
                failures.append(
                    {
                        "model_id": model_id,
                        "background_ratio": ratio,
                        "seed": seed,
                        "required_backgrounds": requested,
                        "available_backgrounds": sum(counts.values()),
                        "reason": "background_pool_insufficient",
                    }
                )
                continue
            chosen = []
            rng = np.random.default_rng(seed)
            for block in sorted(allocation):
                pool = sorted(candidate_by_block[block])
                chosen.extend(
                    rng.choice(pool, size=allocation[block], replace=False).tolist()
                )
            road_ids = (*positive_ids, *sorted(chosen))
            X = (
                numeric.loc[list(road_ids), FEATURE_COLUMNS]
                .reset_index(drop=True)
                .copy()
            )
            y = np.concatenate(
                (
                    np.ones(len(positive_ids), dtype=np.int8),
                    np.zeros(len(chosen), dtype=np.int8),
                )
            )
            draws.append(
                BackgroundDraw(
                    model_id,
                    ratio,
                    seed,
                    X,
                    y,
                    road_ids,
                    {
                        "positive_road_ids": positive_ids,
                        "background_road_ids": sorted(chosen),
                        "positive_event_group_ids": eligible_events,
                        "reference_block_quotas": allocation,
                        "sampling": "unknown_labels_reference_blocks_largest_remainder_without_replacement",
                        "score_semantics": "relative_rank_score_not_probability",
                    },
                )
            )
    if any(failure["background_ratio"] == primary for failure in failures):
        status = "background_pool_insufficient"
        draws = []
    else:
        status = "ready"
    return SamplingResult(status, gate, tuple(draws), tuple(failures))
