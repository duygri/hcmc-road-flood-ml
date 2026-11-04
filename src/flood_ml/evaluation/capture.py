"""Exploratory capture of explicitly reviewed independent flood events.

``events`` is Task 8's confirmed-link table with both match and event provenance.
Connected event/location/observation/road groups crossing partitions are excluded
in full. This module never edits partitions or selects replacement blocks.

Capture additionally requires EVERY event link to submit ``affected_geometry``
(reviewed LineString/MultiLineString subset of that road, in roads.crs),
``affected_geometry_reviewed=True`` and ``affected_geometry_complete=True``.
Completeness is a HUMAN assertion of the full affected extent, never inferred
from accepted candidates. Geometry accepts Shapely objects, WKB bytes or hex. A
GeoDataFrame's CRS must equal roads.crs. Missing assertions exclude the event.
Task 8's optional event CSV ``affected_geometry_complete`` field is false when
missing/blank. Explicit true attests that the accepted FULL road segments form
the complete affected extent; Task 8 then propagates the flag and road WKB with
the existing match/event review provenance. For partial extents use explicitly
reviewed WKB in the extended link table. Never copy fixture assertions into real
reviews; current real capture remains pending until human review is supplied.

An event touching several blocks in one partition belongs to the lexically first
block for quota and bootstrap counts. Ownership is disclosed; no event counts
twice towards block quotas. Lengths use geometric union, never source-row sums.
Only completely covered reviewed events enter the shared capture denominator.
"""

import math
from collections import Counter
from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkb, union_all
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import require_metric_crs
from flood_ml.labeling.matching import require_ids

KEY = "road_segment_id"
FRACTIONS = (0.01, 0.05, 0.1, 0.2)
REVIEW_FIELDS = (
    "evidence",
    "reviewed_by",
    "reviewed_at",
    "event_evidence",
    "event_reviewed_by",
    "event_reviewed_at",
)
RULES = {
    "minimum_reference_events": 20,
    "minimum_holdout_events": 20,
    "minimum_holdout_blocks": 3,
    "minimum_events_per_holdout_block": 5,
    "top_k_length_fractions": list(FRACTIONS),
    "primary_capture_fraction": 0.5,
    "sensitivity_capture": ["any", 1.0],
    "minimum_affected_score_coverage": 1.0,
    "random_seed_start": 42,
    "random_repetitions": 100,
    "bootstrap_seed": 42,
    "bootstrap_repetitions": 1000,
}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _partitions(partitions):
    require_ids(partitions, KEY)
    if (
        partitions.columns.duplicated().any()
        or "partition" not in partitions
        or not partitions.partition.isin(
            ["reference", "holdout", "guard", "excluded"]
        ).all()
        or "block_id" not in partitions
        or not partitions.block_id.map(_text).all()
    ):
        raise PipelineError("Frozen partitions require valid partition and block_id")


def _reviewed(row):
    if not all(_text(row.get(key)) for key in REVIEW_FIELDS):
        return False
    for key in ("reviewed_at", "event_reviewed_at"):
        try:
            stamp = pd.Timestamp(row[key])
            if pd.isna(stamp) or stamp.tzinfo is None:
                return False
        except (ValueError, TypeError, OverflowError):
            return False
    return True


def evaluation_gate(events: pd.DataFrame, partitions: pd.DataFrame) -> dict:
    """Return exact independent-event deficits and a nonmutating group audit.

    ``ready`` requires reference>=20, holdout>=20 and >=3 frozen holdout blocks
    owning >=5 events each. Missing provenance contributes no reviewed events.
    Malformed IDs/partitions raise PipelineError; unknown labels are never used.
    """
    _partitions(partitions)
    keys = [KEY, "event_group_id", "location_group_id"]
    if not isinstance(events, pd.DataFrame) or events.columns.duplicated().any():
        raise PipelineError("Confirmed events require a table with distinct columns")
    if any(key not in events or not events[key].map(_text).all() for key in keys):
        raise PipelineError("Confirmed events require nonblank road/event/location IDs")
    indexed = partitions.set_index(KEY)
    if not set(events[KEY]).issubset(indexed.index):
        raise PipelineError("Reviewed event references nonexistent partition road")
    parents = {}

    def find(token):
        parents.setdefault(token, token)
        if parents[token] != token:
            parents[token] = find(parents[token])
        return parents[token]

    records = events.to_dict("records")
    for row in records:
        tokens = [(key, row[key]) for key in keys]
        if _text(row.get("observation_id")):
            tokens.append(("observation_id", row["observation_id"]))
        roots = [find(token) for token in tokens]
        root = min(roots)
        for other in roots:
            parents[other] = root
    components = {}
    for row in records:
        components.setdefault(find((KEY, row[KEY])), set()).add(row[KEY])
    component_reasons = {}
    for root, road_ids in components.items():
        subset = indexed.loc[sorted(road_ids)]
        parts = set(subset.partition)
        anchor = set(subset.get("block_partition", subset.partition))
        reason = (
            "reviewed_group_crosses_partitions"
            if {"reference", "holdout"}.issubset(parts | anchor)
            else "reviewed_group_outside_evaluation_partitions"
            if not parts.issubset({"reference", "holdout"})
            else ""
        )
        component_reasons[root] = reason
    assignments, reviewed_assignments, excluded = [], [], []
    for event_id, links in events.groupby("event_group_id", sort=True):
        road_ids = sorted(set(links[KEY]))
        reason = component_reasons[find((KEY, road_ids[0]))]
        if not reason and not all(_reviewed(row) for row in links.to_dict("records")):
            reason = "unreviewed_event"
        blocks = sorted(set(indexed.loc[road_ids, "block_id"]))
        owner = {
            "event_group_id": event_id,
            "partition": indexed.loc[road_ids[0], "partition"],
            "block_id": blocks[0],
            "touched_block_ids": blocks,
        }
        geometry, geometry_reason = _reviewed_extent(links)
        length = float(geometry.length) if geometry is not None else None
        if not reason:
            reviewed_assignments.append(owner)
            reason = geometry_reason
        if reason:
            excluded.append(
                {
                    "event_group_id": event_id,
                    "reason": reason,
                    "block_ids": blocks,
                    "partition": owner["partition"],
                    "known_affected_length_m": length,
                }
            )
        else:
            assignments.append(owner)
    reviewed_quota = _quota(reviewed_assignments, partitions)
    geometry_quota = _quota(assignments, partitions)
    return {
        **geometry_quota,
        "reviewed_reference_events": reviewed_quota["reference_events"],
        "reviewed_holdout_events": reviewed_quota["holdout_events"],
        "label_review_shortfalls": reviewed_quota,
        "geometry_completeness_shortfalls": geometry_quota,
        "event_block_ownership": "lexically_first_touched_block_once_per_event",
        "eligible_events": assignments,
        "reviewed_events": reviewed_assignments,
        "excluded_events": excluded,
        "requirements": {
            key: RULES[key]
            for key in (
                "minimum_reference_events",
                "minimum_holdout_events",
                "minimum_holdout_blocks",
                "minimum_events_per_holdout_block",
            )
        },
    }


def _quota(assignments, partitions):
    holdout = [row for row in assignments if row["partition"] == "holdout"]
    reference = [row for row in assignments if row["partition"] == "reference"]
    frozen_blocks = sorted(
        set(partitions.loc[partitions.partition.eq("holdout"), "block_id"])
    )
    counts = Counter(row["block_id"] for row in holdout)
    counts = {block: counts[block] for block in frozen_blocks}
    qualifying = sum(value >= 5 for value in counts.values())
    shortfalls = {block: 5 - value for block, value in counts.items() if value < 5}
    missing_reference, missing_holdout = (
        max(20 - len(reference), 0),
        max(20 - len(holdout), 0),
    )
    missing_blocks = max(3 - qualifying, 0)
    return {
        "status": "pending_label_review"
        if any([missing_reference, missing_holdout, missing_blocks])
        else "ready",
        "reference_events": len(reference),
        "holdout_events": len(holdout),
        "missing_reference_events": missing_reference,
        "missing_holdout_events": missing_holdout,
        "missing_holdout_blocks": missing_blocks,
        "qualifying_holdout_blocks": qualifying,
        "holdout_events_by_block": counts,
        "holdout_block_shortfalls": shortfalls,
    }


def _rules(config):
    if not isinstance(config, dict):
        raise PipelineError("Evaluation configuration must be a mapping")
    experiments = config.get("experiments", config)
    supplied = experiments.get("evaluation", {})
    if not isinstance(supplied, dict) or any(
        key not in RULES or RULES[key] != value for key, value in supplied.items()
    ):
        raise PipelineError("Evaluation rules must match the predeclared protocol")
    return dict(RULES)


def _geometry(value):
    try:
        if isinstance(value, (bytes, memoryview, str)):
            value = from_wkb(bytes(value) if isinstance(value, memoryview) else value)
    except (ValueError, TypeError, GEOSException):
        return None
    if not isinstance(value, BaseGeometry):
        return None
    if (
        value.geom_type not in ("LineString", "MultiLineString")
        or value.is_empty
        or not value.is_valid
        or not np.isfinite(value.bounds).all()
        or value.length <= 0
    ):
        return None
    return value


def _asserted(links, key):
    return (
        key in links
        and links[key].map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v)).all()
    )


def _reviewed_extent(rows):
    geometries = [
        _geometry(row.get("affected_geometry")) for row in rows.to_dict("records")
    ]
    complete_geometry = all(geom is not None for geom in geometries)
    affected = union_all(geometries) if complete_geometry else None
    reason = (
        "missing_affected_geometry"
        if not complete_geometry
        else "unreviewed_affected_geometry"
        if not _asserted(rows, "affected_geometry_reviewed")
        else "incomplete_affected_geometry"
        if not _asserted(rows, "affected_geometry_complete")
        else ""
    )
    return affected, reason


def _exclusion_summary(exclusions):
    # Capture exclusions concern holdout; reference deficits remain in the gate.
    holdout = [row for row in exclusions if row["partition"] == "holdout"]
    counts = dict(sorted(Counter(row["reason"] for row in holdout).items()))
    lengths = {}
    for reason in counts:
        values = [
            row["known_affected_length_m"] for row in holdout if row["reason"] == reason
        ]
        lengths[reason] = (
            None if any(value is None for value in values) else math.fsum(values)
        )
    return {
        "excluded_events": exclusions,
        "exclusion_counts": counts,
        "excluded_affected_length_m": lengths,
    }


def _affected_events(links, gate, roads, eligible_ids):
    road_geometry = roads.set_index(KEY).geometry.to_dict()
    eligible, exclusions = [], []
    ownership = {row["event_group_id"]: row for row in gate["eligible_events"]}
    for event_id, owner in sorted(ownership.items()):
        rows = links.loc[links.event_group_id.eq(event_id)]
        geometries = [
            _geometry(row.get("affected_geometry")) for row in rows.to_dict("records")
        ]
        complete_geometry = all(geom is not None for geom in geometries)
        affected = union_all(geometries) if complete_geometry else None
        length = float(affected.length) if affected is not None else None
        reason = (
            "missing_affected_geometry"
            if not complete_geometry
            else "unreviewed_affected_geometry"
            if not _asserted(rows, "affected_geometry_reviewed")
            else "incomplete_affected_geometry"
            if not _asserted(rows, "affected_geometry_complete")
            else ""
        )
        if not reason:
            outside = any(
                row[KEY] not in road_geometry
                or not road_geometry[row[KEY]].covers(geom)
                for row, geom in zip(rows.to_dict("records"), geometries)
            )
            reason = "affected_geometry_outside_confirmed_road" if outside else ""
        if not reason and not set(rows[KEY]).issubset(eligible_ids):
            reason = "affected_geometry_outside_score_coverage"
        if reason:
            exclusions.append(
                {
                    "partition": owner["partition"],
                    "event_group_id": event_id,
                    "reason": reason,
                    "known_affected_length_m": length,
                }
            )
        else:
            eligible.append(
                {**owner, "geometry": affected, "affected_length_m": length}
            )
    return eligible, exclusions


def _select(ordered, fraction, denominator):
    target = fraction * denominator
    cumulative = ordered.road_length_m.cumsum().to_numpy()
    n = (
        min(int(np.searchsorted(cumulative, target, side="left")) + 1, len(ordered))
        if denominator > 0
        else 0
    )
    selected = ordered.iloc[:n]
    length = float(selected.road_length_m.sum())
    return {
        "selected_road_segment_ids": selected[KEY].tolist(),
        "target_length_m": target,
        "selected_length_m": length,
        "actual_selected_fraction": length / denominator if denominator else None,
        "overshoot_m": max(length - target, 0),
    }


def _rate(values):
    return float(np.mean(values)) if len(values) else None


def _ratio(numerator, denominator):
    return (
        numerator / denominator
        if numerator is not None and denominator is not None and denominator > 0
        else None
    )


def _ci(values):
    defined = [value for value in values if value is not None]
    return (
        [float(value) for value in np.quantile(defined, [0.025, 0.975])]
        if defined
        else None
    )


def _bootstrap_indices(gate, events):
    # Retain reviewed blocks even if all their events fail geometry coverage.
    # A draw with no eligible events has undefined capture, never zero capture.
    blocks = sorted(
        {
            row["block_id"]
            for row in gate["eligible_events"]
            if row["partition"] == "holdout"
        }
    )
    members = {
        block: [i for i, row in enumerate(events) if row["block_id"] == block]
        for block in blocks
    }
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        sampled = []
        for block in rng.choice(blocks, size=len(blocks), replace=True):
            candidates = members[block]
            if candidates:
                sampled.extend(
                    rng.choice(candidates, size=len(candidates), replace=True).tolist()
                )
        draws.append(np.array(sampled, dtype=int))
    return draws


def _track_b_table(scores, partitions):
    columns = {
        KEY,
        "model_id",
        "background_ratio",
        "seed",
        "relative_score",
        "partition",
    }
    if (
        not isinstance(scores, pd.DataFrame)
        or scores.columns.duplicated().any()
        or not columns.issubset(scores)
    ):
        raise PipelineError("Track B scores require the long-form score schema")
    table = scores[sorted(columns)].copy()
    if (
        not table[KEY].map(_text).all()
        or not table.model_id.map(_text).all()
        or table.duplicated([KEY, "model_id"]).any()
    ):
        raise PipelineError("Track B requires nonblank IDs and unique road/model pairs")
    for column, minimum in (("background_ratio", 1), ("seed", 0)):
        if (
            not table[column]
            .map(
                lambda value, lower=minimum: (
                    isinstance(value, (int, np.integer))
                    and not isinstance(value, (bool, np.bool_))
                    and value >= lower
                )
            )
            .all()
        ):
            raise PipelineError("Track B ratio and seed must be valid integers")
    if (
        table.groupby("model_id")[["background_ratio", "seed"]]
        .nunique()
        .ne(1)
        .any()
        .any()
    ):
        raise PipelineError("Track B model IDs require consistent ratio and seed")
    if (
        table.groupby("model_id")[["background_ratio", "seed"]]
        .first()
        .duplicated()
        .any()
    ):
        raise PipelineError("Track B ratio/seed draws must have distinct model IDs")
    table["relative_score"] = pd.to_numeric(table.relative_score, errors="coerce")
    if not np.isfinite(table.relative_score).all():
        raise PipelineError("Track B relative scores must be finite")
    partition_map = partitions.set_index(KEY).partition
    if (
        not table.partition.eq("holdout").all()
        or not table[KEY].map(partition_map).eq("holdout").all()
    ):
        raise PipelineError("Track B scores must agree with frozen holdout partitions")
    return table.sort_values(["model_id", KEY]).reset_index(drop=True)


def _track_b_coverage(scores, population):
    required = set(population[KEY])
    models = {}
    for model_id, rows in scores.groupby("model_id", sort=True):
        missing = sorted(required - set(rows[KEY]))
        models[model_id] = {
            "status": "pending" if missing else "ready",
            "background_ratio": int(rows.background_ratio.iloc[0]),
            "seed": int(rows.seed.iloc[0]),
            "eligible_road_count": len(required & set(rows[KEY])),
            "missing_road_segment_ids": missing,
        }
    return {
        "status": "ready"
        if models and all(model["status"] == "ready" for model in models.values())
        else "pending",
        "required_road_count": len(required),
        "required_road_length_m": float(population.road_length_m.sum()),
        "model_count": len(models),
        "models": models,
    }


def _distribution(values):
    defined = [value for value in values if value is not None]
    q25, median, q75 = (
        [float(value) for value in np.quantile(defined, [0.25, 0.5, 0.75])]
        if defined
        else (None, None, None)
    )
    return {
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25 if defined else None,
        "defined_draws": len(defined),
        "undefined_draws": len(values) - len(defined),
    }


def _track_b_stability(models):
    by_ratio = {}
    for ratio in sorted({model["background_ratio"] for model in models.values()}):
        draws = [
            model for model in models.values() if model["background_ratio"] == ratio
        ]
        top_k = {}
        for fraction in FRACTIONS:
            metrics = [model["top_k"][str(fraction)] for model in draws]
            selected = [set(metric["selected_road_segment_ids"]) for metric in metrics]
            overlaps = [
                len(first & second) / len(first | second) if first | second else None
                for first, second in combinations(selected, 2)
            ]
            top_k[str(fraction)] = {
                **{
                    field: _distribution([metric[field] for metric in metrics])
                    for field in ("capture", "capture_any", "capture_all", "lift")
                },
                "mean_pairwise_jaccard": _rate(
                    [value for value in overlaps if value is not None]
                ),
                "pair_count": len(overlaps),
            }
        by_ratio[str(ratio)] = {
            "model_count": len(draws),
            "seeds": sorted(model["seed"] for model in draws),
            "top_k": top_k,
        }
    return {"by_background_ratio": by_ratio}


def evaluate_capture(
    scores: pd.DataFrame,
    roads: gpd.GeoDataFrame,
    confirmed_event_segments: pd.DataFrame,
    partitions: pd.DataFrame,
    config: dict,
    *,
    track_b_scores: pd.DataFrame | None = None,
) -> dict:
    """Return pending without metrics, or conditional reviewed holdout capture.

    Scores require unique IDs, finite score/elevation and boolean mcda_eligible.
    All methods compare exactly the same score-eligible holdout ROAD length.
    Events use union AFFECTED length. Budgets select whole segments through the
    target, ties ascending by road ID. Random ranks use PCG64 seed42..141; 1000
    seed42 block-then-event bootstrap resamples hold the rankings fixed. Null
    reports undefined rates/lift/intervals, including empty event denominators.
    """
    rules = _rules(config)
    gate = evaluation_gate(confirmed_event_segments, partitions)
    track_b = (
        _track_b_table(track_b_scores, partitions)
        if track_b_scores is not None
        else None
    )
    report = {
        "status": gate["status"],
        "evaluation_performed": False,
        "gate": gate,
        "exploratory": True,
        "limitations": "Reviewed-source selection and reporting bias; incomplete coverage; few independent spatial blocks. Exploratory capture is not a population flood rate or calibrated probability.",
        **_exclusion_summary(gate["excluded_events"]),
    }
    if gate["status"] != "ready":
        return report
    require_ids(scores, KEY)
    require_ids(roads, KEY)
    require_metric_crs(roads)
    if scores.columns.duplicated().any() or not {
        "score",
        "elevation_median_m",
        "mcda_eligible",
    }.issubset(scores):
        raise PipelineError(
            "Scores require score, elevation_median_m and mcda_eligible"
        )
    if not scores.mcda_eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise PipelineError("mcda_eligible must contain booleans")
    if not set(scores[KEY]).issubset(set(partitions[KEY])) or not set(
        roads[KEY]
    ).issubset(set(partitions[KEY])):
        raise PipelineError("Scores/roads must belong to frozen partition population")
    if roads.geometry.map(_geometry).isna().any():
        raise PipelineError("Road geometry must be nonempty finite metric lines")
    if (
        isinstance(confirmed_event_segments, gpd.GeoDataFrame)
        and confirmed_event_segments.crs != roads.crs
    ):
        raise PipelineError("Affected geometry CRS must equal roads CRS")
    partition_map = partitions.set_index(KEY).partition
    if (
        "partition" in scores
        and not scores.partition.eq(scores[KEY].map(partition_map)).all()
    ):
        raise PipelineError("Scores disagree with frozen partitions")
    table = scores[[KEY, "score", "elevation_median_m", "mcda_eligible"]].copy()
    for column in ("score", "elevation_median_m"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    lengths = pd.DataFrame({KEY: roads[KEY], "road_length_m": roads.geometry.length})
    table = table.merge(lengths, on=KEY, how="inner", validate="one_to_one")
    table = (
        table.loc[
            table[KEY].map(partition_map).isin(["reference", "holdout"])
            & table.mcda_eligible
            & np.isfinite(table.score)
            & np.isfinite(table.elevation_median_m)
        ]
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    events, exclusions = _affected_events(
        confirmed_event_segments, gate, roads, set(table[KEY])
    )
    coverage_quota = _quota(events, partitions)
    all_exclusions = [*gate["excluded_events"], *exclusions]
    final_gate = {
        **gate,
        **coverage_quota,
        "score_coverage_shortfalls": coverage_quota,
        "eligible_events": [
            {
                key: value
                for key, value in event.items()
                if key not in {"geometry", "affected_length_m"}
            }
            for event in events
        ],
        "excluded_events": all_exclusions,
    }
    events = [event for event in events if event["partition"] == "holdout"]
    table = table.loc[table[KEY].map(partition_map).eq("holdout")]
    denominator = float(table.road_length_m.sum())
    eligible_ids = set(table[KEY])
    report.update(
        gate=final_gate,
        review_geometry_gate=gate,
        status=final_gate["status"],
        eligible_event_count=len(events),
        eligible_road_length_m=denominator,
        eligible_road_segment_ids=sorted(eligible_ids),
        **_exclusion_summary(all_exclusions),
    )
    if final_gate["status"] != "ready":
        return report
    if track_b is not None:
        coverage = _track_b_coverage(track_b, table)
        report["track_b_coverage"] = coverage
        if coverage["status"] != "ready":
            report["status"] = "pending_track_b_coverage"
            return report
    geometry = roads.set_index(KEY).geometry
    event_ids = [event["event_group_id"] for event in events]
    methods, captures = {}, {}

    def evaluate_order(ordered):
        budgets, outcomes = {}, {}
        for fraction in FRACTIONS:
            selected = _select(ordered, fraction, denominator)
            selected_union = union_all(
                geometry.loc[selected["selected_road_segment_ids"]].to_list()
            )
            fractions = np.array(
                [
                    min(
                        1.0,
                        max(
                            0.0,
                            event["geometry"].intersection(selected_union).length
                            / event["affected_length_m"],
                        ),
                    )
                    for event in events
                ],
                dtype=float,
            )
            outcomes[str(fraction)] = np.column_stack(
                (fractions >= 0.5, fractions > 0, fractions >= 1.0)
            ).astype(float)
            budgets[str(fraction)] = {
                **selected,
                "capture": _rate(fractions >= 0.5),
                "capture_any": _rate(fractions > 0),
                "capture_all": _rate(fractions >= 1.0),
                "event_affected_fraction": dict(zip(event_ids, fractions.tolist())),
            }
        return budgets, outcomes

    for name, column, ascending in [
        ("mcda", "score", False),
        ("elevation_only", "elevation_median_m", True),
    ]:
        budgets, outcomes = evaluate_order(
            table.sort_values([column, KEY], ascending=[ascending, True], kind="stable")
        )
        methods[name] = {"eligible_road_length_m": denominator, "top_k": budgets}
        captures[name] = outcomes
    track_b_methods, track_b_captures = {}, {}
    if track_b is not None:
        for model_id, rows in track_b.groupby("model_id", sort=True):
            ranked = table.merge(
                rows[[KEY, "relative_score"]], on=KEY, validate="one_to_one"
            ).sort_values(
                ["relative_score", KEY], ascending=[False, True], kind="stable"
            )
            budgets, outcomes = evaluate_order(ranked)
            track_b_methods[model_id] = {
                "background_ratio": int(rows.background_ratio.iloc[0]),
                "seed": int(rows.seed.iloc[0]),
                "eligible_road_length_m": denominator,
                "top_k": budgets,
            }
            track_b_captures[model_id] = outcomes
    random_runs, random_outcomes = [], []
    for seed in range(42, 142):
        ordered = table.iloc[np.random.default_rng(seed).permutation(len(table))]
        budgets, outcomes = evaluate_order(ordered)
        random_runs.append({"seed": seed, "top_k": budgets})
        random_outcomes.append(outcomes)
    captures["random"] = {
        str(fraction): np.mean(
            [outcomes[str(fraction)] for outcomes in random_outcomes], axis=0
        )
        for fraction in FRACTIONS
    }
    random_summary = {}
    for fraction in FRACTIONS:
        key = str(fraction)
        matrix = captures["random"][key]
        random_summary[key] = {
            field: _rate(matrix[:, i])
            for i, field in enumerate(("capture", "capture_any", "capture_all"))
        }
        random_summary[key]["actual_selected_fraction_mean"] = (
            _rate(
                [run["top_k"][key]["actual_selected_fraction"] for run in random_runs]
            )
            if denominator
            else None
        )
    methods["random"] = {
        "eligible_road_length_m": denominator,
        "seeds": list(range(42, 142)),
        "runs": random_runs,
        "top_k": random_summary,
    }
    draws = _bootstrap_indices(gate, events)
    ranked_methods = [
        (name, method, captures[name], False) for name, method in methods.items()
    ] + [
        (name, method, track_b_captures[name], True)
        for name, method in track_b_methods.items()
    ]
    for name, method, method_captures, is_track_b in ranked_methods:
        for fraction in FRACTIONS:
            key = str(fraction)
            metric = method["top_k"][key]
            metric["lift"] = _ratio(
                metric["capture"], methods["random"]["top_k"][key]["capture"]
            )
            if name == "mcda":
                metric["lift_vs_elevation_only"] = _ratio(
                    metric["capture"],
                    methods["elevation_only"]["top_k"][key]["capture"],
                )
            if is_track_b:
                for baseline in ("mcda", "elevation_only"):
                    baseline_rate = methods[baseline]["top_k"][key]["capture"]
                    metric[f"lift_vs_{baseline}"] = _ratio(
                        metric["capture"], baseline_rate
                    )
                    metric[f"capture_difference_vs_{baseline}"] = (
                        metric["capture"] - baseline_rate
                        if metric["capture"] is not None and baseline_rate is not None
                        else None
                    )
            rates, any_rates, all_rates, lifts = [], [], [], []
            for sample in draws:
                outcomes = method_captures[key][sample]
                random_sample = captures["random"][key][sample]
                rate = _rate(outcomes[:, 0])
                rates.append(rate)
                any_rates.append(_rate(outcomes[:, 1]))
                all_rates.append(_rate(outcomes[:, 2]))
                lifts.append(_ratio(rate, _rate(random_sample[:, 0])))
            metric["uncertainty"] = {
                "method": "spatial_block_then_event_percentile_bootstrap_fixed_rankings",
                "replicates": 1000,
                "valid_replicates": sum(value is not None for value in rates),
                "undefined_replicates": rates.count(None),
                "undefined_lift_replicates": lifts.count(None),
                "capture_ci95": _ci(rates),
                "capture_any_ci95": _ci(any_rates),
                "capture_all_ci95": _ci(all_rates),
                "lift_ci95": _ci(lifts),
            }
    if track_b is not None:
        methods["track_b"] = track_b_methods
        report["track_b_stability"] = _track_b_stability(track_b_methods)
    report.update(
        status="complete",
        evaluation_performed=True,
        rules=rules,
        scope="score_eligible_holdout_input_roads_only",
        partition_population_count=len(partitions),
        score_input_count=len(scores),
        road_input_count=len(roads),
        eligible_road_segment_ids=sorted(eligible_ids),
        eligible_road_length_m=denominator,
        eligible_event_count=len(events),
        eligible_event_ids=event_ids,
        event_affected_length_m={
            event["event_group_id"]: event["affected_length_m"] for event in events
        },
        methods=methods,
    )
    return report
