"""Explicit human review is the only source of positive road labels.

Event reviews define confirmed observation membership in reviewer-assigned event
and location groups; they never confirm automated candidate groups implicitly.
One decision per observation/segment and one event membership per observation
are supported. Duplicate/conflicting inputs fail rather than selecting a winner.
"""

import json

import geopandas as gpd
import pandas as pd

from flood_ml.config import PipelineError
from flood_ml.labeling.matching import require_ids

MATCH_COLUMNS = [
    "observation_id",
    "road_segment_id",
    "decision",
    "evidence",
    "reviewed_by",
    "reviewed_at",
]
EVENT_COLUMNS = [
    "observation_id",
    "event_group_id",
    "location_group_id",
    "evidence",
    "reviewed_by",
    "reviewed_at",
]
LINK_COLUMNS = [
    "observation_id",
    "source_id",
    "road_segment_id",
    "event_group_id",
    "location_group_id",
    "evidence",
    "reviewed_by",
    "reviewed_at",
    "event_evidence",
    "event_reviewed_by",
    "event_reviewed_at",
    "match_distance_m",
    "match_method",
]


def _table(frame, columns, keys):
    if not set(columns).issubset(frame.columns) or frame.columns.duplicated().any():
        raise PipelineError(f"Review columns required: {columns}")
    result = frame.copy()
    for column in columns:
        if (
            result[column].isna().any()
            or not result[column]
            .map(lambda x: isinstance(x, str) and bool(x.strip()))
            .all()
        ):
            raise PipelineError(f"Review {column} must be nonblank text")
        # Keep submitted evidence/history intact; normalize only surrounding whitespace.
        result[column] = result[column].str.strip()
    if result.duplicated(keys).any():
        raise PipelineError(f"Duplicate or conflicting review keys: {keys}")
    for timestamp in result.reviewed_at:
        try:
            parsed = pd.Timestamp(timestamp)
            if pd.isna(parsed) or parsed.tzinfo is None:
                raise ValueError("timezone required")
        except (ValueError, TypeError, OverflowError) as exc:
            raise PipelineError(
                "Invalid reviewed_at; timestamp with timezone required"
            ) from exc
    return result


def read_reviews(path, columns):
    """Read text CSV without coercing IDs or treating source strings as NA."""
    try:
        result = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Cannot read review CSV: {exc}") from exc
    keys = (
        ["observation_id", "road_segment_id"]
        if "decision" in columns
        else ["observation_id"]
    )
    return _table(result, columns, keys)


def apply_reviews(
    roads: gpd.GeoDataFrame,
    observations: gpd.GeoDataFrame,
    candidates: pd.DataFrame,
    match_reviews: pd.DataFrame,
    event_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return nullable road labels and confirmed source/event/segment links.

    verified_negative is deliberately unsupported for current IRD-only evidence.
    Rejected/needs_info/no_report decisions remain audit history, never negatives.
    """
    require_ids(roads, "road_segment_id")
    require_ids(observations, "observation_id")
    if (
        "source_id" not in observations
        or observations.source_id.isna().any()
        or not observations.source_id.map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all()
    ):
        raise PipelineError("Observation source_id required")
    mr = _table(match_reviews, MATCH_COLUMNS, ["observation_id", "road_segment_id"])
    er = _table(event_reviews, EVENT_COLUMNS, ["observation_id"])
    if not mr.decision.isin(["accepted", "rejected", "needs_info", "no_report"]).all():
        raise PipelineError(
            "Unsupported review decision; verified_negative requires independent data"
        )
    obs = observations.set_index("observation_id")
    road_ids = set(roads.road_segment_id)
    for frame in [candidates, mr, er]:
        if "observation_id" not in frame or not set(frame.observation_id).issubset(
            obs.index
        ):
            raise PipelineError("Nonexistent observation_id")
    if not set(mr.road_segment_id).issubset(road_ids) or not set(
        candidates.road_segment_id.dropna()
    ).issubset(road_ids):
        raise PipelineError("Nonexistent road_segment_id")
    pairs = candidates.dropna(subset=["road_segment_id"])
    if pairs.duplicated(["observation_id", "road_segment_id"]).any():
        raise PipelineError("Duplicate candidate pair")
    pair_index = pairs.set_index(["observation_id", "road_segment_id"])
    events = er.set_index("observation_id")
    if (
        not er.empty
        and er.groupby("event_group_id").location_group_id.nunique().gt(1).any()
    ):
        raise PipelineError("Inconsistent location_group_id within event_group_id")
    for _, event in er.iterrows():
        for name in ["event_group_id", "location_group_id"]:
            expected = obs.loc[event.observation_id].get(name)
            if pd.notna(expected) and expected != event[name]:
                raise PipelineError(f"Inconsistent or nonexistent {name}")
    links = []
    for _, review in mr.iterrows():
        key = (review.observation_id, review.road_segment_id)
        if key not in pair_index.index:
            raise PipelineError("Review references nonexistent candidate pair")
        observation = obs.loc[review.observation_id]
        if bool(observation.get("outside_aoi", True)):
            raise PipelineError("Outside-AOI observation cannot match AOI road")
        if review.decision != "accepted":
            continue
        if review.observation_id not in events.index:
            raise PipelineError("Accepted match requires explicit event review")
        event = events.loc[review.observation_id]
        candidate = pair_index.loc[key]
        links.append(
            {
                "observation_id": review.observation_id,
                "source_id": observation.source_id,
                "road_segment_id": review.road_segment_id,
                "event_group_id": event.event_group_id,
                "location_group_id": event.location_group_id,
                "evidence": review.evidence,
                "reviewed_by": review.reviewed_by,
                "reviewed_at": review.reviewed_at,
                "event_evidence": event.evidence,
                "event_reviewed_by": event.reviewed_by,
                "event_reviewed_at": event.reviewed_at,
                "match_distance_m": candidate.match_distance_m,
                "match_method": candidate.match_method,
            }
        )
    confirmed = pd.DataFrame(links, columns=LINK_COLUMNS)
    labels = roads[["road_segment_id"]].copy().reset_index(drop=True)
    labels["label_status"] = "unknown"
    labels["label_binary"] = pd.array([None] * len(labels), dtype="Int64")
    labels["label_confidence"] = None
    labels["label_evidence"] = "[]"
    labels["decision_history"] = "[]"
    for i, road in labels.loc[
        labels.road_segment_id.isin(mr.road_segment_id)
    ].iterrows():
        history = mr.loc[mr.road_segment_id.eq(road.road_segment_id)].to_dict("records")
        labels.at[i, "decision_history"] = json.dumps(history)
        evidence = confirmed.loc[confirmed.road_segment_id.eq(road.road_segment_id)]
        if not evidence.empty:
            labels.at[i, "label_status"] = "positive"
            labels.at[i, "label_binary"] = 1
            labels.at[i, "label_evidence"] = evidence.to_json(orient="records")
    return labels, confirmed
