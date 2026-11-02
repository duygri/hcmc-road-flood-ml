"""Provenance-preserving observation inputs and unconfirmed event candidates.

No I/O or review decisions occur here. Pass raw source attributes before a reader
coerces date strings (e.g. restore GPKG Date text by Obs_id from read-only SQLite).
Raw columns are copied to raw_* before normalization; geometry is projected to
the AOI CRS, and original geometry WKB/CRS are retained for audit.
"""

import hashlib
import json
import math
import re
from datetime import date, datetime

import geopandas as gpd
import pandas as pd

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import require_metric_crs, select_aoi


def _missing(value):
    return value is None or pd.isna(value) or str(value).strip() == ""


def _date(value, year):
    flags = []
    observed = pd.NaT
    precision = "unknown"
    observed_year = None
    if not _missing(year):
        try:
            number = float(year)
            if number.is_integer() and 1 <= number <= 9999:
                observed_year = int(number)
            else:
                flags.append("invalid_year")
        except (ValueError, TypeError):
            flags.append("invalid_year")
    if _missing(value):
        flags.append("missing_date")
    elif re.fullmatch(r"\d{4}", str(value).strip()):
        value_year = int(str(value).strip())
        if 1 <= value_year <= 9999:
            if observed_year is not None and observed_year != value_year:
                flags.append("date_year_conflict")
            observed_year = value_year
            flags.append("year_only_date")
        else:
            flags.append("unparseable_date")
    elif isinstance(value, (date, datetime, pd.Timestamp)) or re.fullmatch(
        r"\d{4}-\d{1,2}-\d{1,2}(?:[T ].+)?", str(value).strip()
    ):
        try:
            # Preserve the source's calendar day; never shift it through UTC.
            parsed = pd.Timestamp(value)
            observed = pd.Timestamp(parsed.date())
            precision = "day"
            if observed_year is not None and observed_year != parsed.year:
                flags.append("date_year_conflict")
            observed_year = parsed.year
            if not isinstance(value, str) or value != observed.strftime("%Y-%m-%d"):
                flags.append("date_normalized")
        except (ValueError, TypeError, OverflowError):
            flags.append("unparseable_date")
    else:
        flags.append("unparseable_date")
    if precision != "day" and observed_year is not None:
        precision = "year"
    return observed, observed_year, precision, flags


def _ids(frame):
    for name in ("observation_id", "source_id"):
        if name not in frame or frame[name].map(_missing).any():
            raise PipelineError(f"Missing {name}")
    if frame.observation_id.duplicated().any():
        raise PipelineError("Duplicate observation_id")


def normalize_observations(
    raw: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Keep every point, project to metric AOI, and add auditable normalized fields.

    Native Obs_id/Source_id map verbatim when canonical IDs are absent. Callers
    may supply stable record keys (e.g. GPKG fid) if native Obs_id repeats; native
    values remain untouched and duplicate native IDs are flagged for review.
    Source precision codes express description quality, never numeric accuracy.
    Unknown codes remain explicit and flagged. Flags are a JSON string array.
    """
    if raw.crs is None:
        raise PipelineError("Observations require a declared CRS")
    require_metric_crs(aoi)
    # Reuse the AOI contract: exactly one valid polygon, without implicit repair.
    checked = select_aoi(
        aoi.assign(_aoi="selected"), field="_aoi", value="selected", target_crs=aoi.crs
    )
    if (
        raw.geometry.isna().any()
        or raw.geometry.is_empty.any()
        or not raw.geometry.is_valid.all()
        or not raw.geom_type.eq("Point").all()
    ):
        raise PipelineError("Observations require valid nonempty Point geometries")
    result = raw.copy(deep=True)
    for column in raw.columns:
        if column != raw.geometry.name:
            if f"raw_{column}" in raw.columns:
                raise PipelineError("Reserved raw_* audit column collision")
            result[f"raw_{column}"] = raw[column]
    result["raw_geometry_wkb"] = raw.geometry.to_wkb(hex=True)
    result["raw_geometry_crs"] = str(raw.crs)
    for canonical, native in [("observation_id", "Obs_id"), ("source_id", "Source_id")]:
        if canonical not in result and native in result:
            result[canonical] = result[native]
    _ids(result)
    result = result.to_crs(checked.crs).reset_index(drop=True)
    polygon = checked.geometry.iloc[0]
    result["outside_aoi"] = ~result.geometry.map(polygon.covers).astype(bool)
    dates, years, precisions, all_flags = [], [], [], []
    normalized = {
        key: []
        for key in (
            "spatial_precision",
            "geocoding_method",
            "source_type",
            "cause",
            "location_name",
        )
    }
    duplicate_native = (
        result.Obs_id.duplicated(keep=False)
        if "Obs_id" in result
        else pd.Series(False, index=result.index)
    )
    for index, row in result.iterrows():
        observed, year, precision, flags = _date(
            row.get("Date", row.get("observed_date")), row.get("Year")
        )
        if duplicate_native.iloc[index]:
            flags.append("duplicate_native_observation_id")
        dates.append(observed)
        years.append(year)
        precisions.append(precision)
        for target, native in [
            ("spatial_precision", "Spatial_precision"),
            ("geocoding_method", "Geocoding_method"),
            ("source_type", "Source_type"),
            ("cause", "Cause"),
            ("location_name", "Location_name"),
        ]:
            source = row.get(native, row.get(target))
            value = None if _missing(source) else str(source).strip()
            if value is not None and target != "location_name":
                value = re.sub(r"\s+", "_", value.lower().replace("_", " "))
            if target == "spatial_precision":
                value = {"hgh": "high", "hugh": "high"}.get(value, value)
                if value not in {"high", "medium", "low"}:
                    flags.append("unknown_spatial_precision")
            if not _missing(source) and value != source:
                flags.append(f"{target}_normalized")
            normalized[target].append(value)
        if row.outside_aoi:
            flags.append("outside_aoi")
        all_flags.append(json.dumps(sorted(set(flags))))
    result["observed_date"] = pd.to_datetime(pd.Series(dates, dtype="object"))
    result["observed_year"] = pd.array(years, dtype="Int64")
    result["date_precision"] = pd.Series(precisions, dtype="str")
    for name, values in normalized.items():
        result[name] = pd.Series(values, dtype="object")
    if "location_accuracy_m" not in result:
        result["location_accuracy_m"] = float("nan")
    result["quality_flags"] = pd.Series(all_flags, dtype="str")
    result["quality_flag"] = "review"
    result["review_required"] = True
    return result


def propose_event_groups(observations: gpd.GeoDataFrame, config: dict) -> pd.DataFrame:
    """Return one unconfirmed membership per observation, retaining every source.

    Edges need day precision AND both configured thresholds. Missing-day records
    stay singleton review candidates. Components can contain transitive/ambiguous
    chains; candidate_edges records direct evidence and is not event confirmation.
    IDs hash sorted observation IDs (stable for fixed inputs); names never merge.
    source_ids and candidate_edges are JSON for portable review-table storage.
    """
    require_metric_crs(observations)
    _ids(observations)
    thresholds = []
    for name in ("candidate_distance_m", "candidate_time_days"):
        value = config.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise PipelineError(f"{name} must be finite and nonnegative")
        thresholds.append(value)
    distance, days = thresholds
    obs = observations.sort_values("observation_id").reset_index(drop=True)
    parent = list(range(len(obs)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    edges = []
    for i in range(len(obs)):
        left = obs.iloc[i]
        if left.date_precision != "day" or pd.isna(left.observed_date):
            continue
        for j in range(i + 1, len(obs)):
            right = obs.iloc[j]
            if right.date_precision != "day" or pd.isna(right.observed_date):
                continue
            delta = (
                abs((left.observed_date - right.observed_date).total_seconds()) / 86400
            )
            separation = left.geometry.distance(right.geometry)
            if delta <= days and separation <= distance:
                parent[root(j)] = root(i)
                edges.append(
                    {
                        "observation_id_a": left.observation_id,
                        "observation_id_b": right.observation_id,
                        "distance_m": separation,
                        "time_delta_days": delta,
                    }
                )
    records = []
    for i, row in obs.iterrows():
        members = obs.iloc[[j for j in range(len(obs)) if root(j) == root(i)]]
        ids = members.observation_id.tolist()
        group_id = (
            "candidate_" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:24]
        )
        records.append(
            {
                "observation_id": row.observation_id,
                "source_id": row.source_id,
                "candidate_group_id": group_id,
                "candidate_status": "unconfirmed",
                "review_required": True,
                "review_reason": "missing_day"
                if row.date_precision != "day"
                else "candidate_requires_review",
                "source_ids": json.dumps(sorted(set(members.source_id))),
                "candidate_edges": json.dumps(
                    [edge for edge in edges if edge["observation_id_a"] in ids]
                ),
                "spatial_precision": row.spatial_precision,
                "location_accuracy_m": row.location_accuracy_m,
                "observed_date": row.observed_date,
                "date_precision": row.date_precision,
                "outside_aoi": row.outside_aoi,
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "observation_id",
            "source_id",
            "candidate_group_id",
            "candidate_status",
            "review_required",
            "review_reason",
            "source_ids",
            "candidate_edges",
            "spatial_precision",
            "location_accuracy_m",
            "observed_date",
            "date_precision",
            "outside_aoi",
        ],
    )
