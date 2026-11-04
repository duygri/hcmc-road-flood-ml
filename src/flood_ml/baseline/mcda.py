"""Reference-only MCDA hypothesis; scores are relative indices, not probabilities.

The five inputs/directions are deliberately fixed. Fit receives reference rows
only and uses complete, structurally eligible rows. Each missing input prevents
scoring even when its weight is zero. Quantile clipping is linear; a collapsed
reference range contributes 0.5. Diagnostic p10 is never scored.
"""

import hashlib
import itertools
import math

import numpy as np
import pandas as pd

from flood_ml.config import PipelineError
from flood_ml.gis.roads import _active
from flood_ml.utils.run import _hash_json

KEY = "road_segment_id"
FEATURE_SPEC = {
    "elevation_median_m": "lower_is_higher_risk",
    "relative_elevation_m": "lower_is_higher_risk",
    "slope_mean_deg": "lower_is_higher_risk",
    "dist_water_m": "lower_is_higher_risk",
    "builtup_ratio_250m": "higher_is_higher_risk",
}
TOP_K_FRACTIONS = (0.01, 0.05, 0.1, 0.2)


def _table(frame):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise PipelineError("Features require a DataFrame with distinct columns")
    if (
        KEY not in frame
        or not frame[KEY].map(lambda v: isinstance(v, str) and bool(v.strip())).all()
        or not frame[KEY].is_unique
    ):
        raise PipelineError("Features require unique nonblank road_segment_id")
    forbidden = [
        c
        for c in frame
        if c.startswith(("label_", "flood_"))
        or c in {"source_count", "source_id", "observation_id", "event_group_id"}
    ]
    if forbidden:
        raise PipelineError(f"Label/evidence columns are forbidden: {forbidden}")


def _values_and_reasons(frame):
    _table(frame)
    values = pd.DataFrame(index=frame.index)
    for feature in FEATURE_SPEC:
        column = frame.get(feature, pd.Series(np.nan, index=frame.index))
        values[feature] = pd.to_numeric(column, errors="coerce").astype(float)
    values = values.where(np.isfinite(values))
    if "water_distance_incomplete" in frame:
        if not frame.water_distance_incomplete.map(
            lambda v: isinstance(v, (bool, np.bool_))
        ).all():
            raise PipelineError("water_distance_incomplete must contain booleans")
        values.loc[frame.water_distance_incomplete, "dist_water_m"] = np.nan
    reasons = pd.Series("", index=frame.index, dtype=object)
    for structure in ("bridge", "tunnel"):
        if structure in frame:
            active = frame[structure].map(_active)
            reasons.loc[active] = reasons.loc[active].map(
                lambda prior, structure=structure: ";".join(
                    filter(None, [prior, structure])
                )
            )
    if "score_eligible" in frame:
        if not frame.score_eligible.map(
            lambda v: isinstance(v, (bool, np.bool_))
        ).all():
            raise PipelineError("score_eligible must contain booleans")
        excluded = ~frame.score_eligible & reasons.eq("")
        supplied = frame.get("score_exclusion_reason", pd.Series("", index=frame.index))
        reasons.loc[excluded] = (
            supplied.loc[excluded].fillna("").replace("", "score_policy_excluded")
        )
    return values, reasons


def fit_normalizer(reference_features: pd.DataFrame, feature_spec: dict) -> dict:
    """Fit q05/q95 on complete eligible reference rows, never on held-out rows.

    A supplied partition column must contain only 'reference'. The caller must
    select that frozen partition before calling. Hashes cover the actual given
    reference inputs and the fixed rules; CLI adds file/config provenance.
    """
    if feature_spec != FEATURE_SPEC:
        raise PipelineError(
            "MCDA feature_spec must contain exactly the five approved directions"
        )
    if (
        "partition" in reference_features
        and not reference_features.partition.eq("reference").all()
    ):
        raise PipelineError("Normalizer may receive reference partition rows only")
    values, reasons = _values_and_reasons(reference_features)
    eligible = values.notna().all(axis=1) & reasons.eq("")
    if (
        "reference_fit_eligible" in reference_features
        and not reference_features.reference_fit_eligible.eq(True).all()
    ):
        raise PipelineError("Reference fit mask conflicts with reference partition")
    fit = values.loc[eligible]
    if fit.empty:
        raise PipelineError("Reference partition has no complete eligible rows")
    audit = values.copy()
    audit[KEY] = reference_features[KEY]
    audit["structure_exclusion"] = reasons
    digest = hashlib.sha256(
        audit.sort_values(KEY).to_json(orient="records", double_precision=15).encode()
    ).hexdigest()
    thresholds = {}
    for feature, direction in FEATURE_SPEC.items():
        lower, upper = fit[feature].quantile([0.05, 0.95]).tolist()
        thresholds[feature] = {
            "q05": lower,
            "q95": upper,
            "direction": direction,
            "constant_reference": lower == upper,
        }
    return {
        "schema_version": 1,
        "method": "q05_q95_linear_clip",
        "quantiles": [0.05, 0.95],
        "feature_spec": dict(FEATURE_SPEC),
        "feature_spec_sha256": _hash_json(FEATURE_SPEC),
        "reference_input_sha256": digest,
        "reference_row_count": len(values),
        "fit_row_count": len(fit),
        "fit_road_segment_ids": sorted(reference_features.loc[eligible, KEY].tolist()),
        "features": thresholds,
    }


def _validate_normalizer(normalizer):
    if (
        not isinstance(normalizer, dict)
        or normalizer.get("feature_spec") != FEATURE_SPEC
        or normalizer.get("method") != "q05_q95_linear_clip"
        or normalizer.get("quantiles") != [0.05, 0.95]
    ):
        raise PipelineError("Invalid MCDA normalizer rules")
    if set(normalizer.get("features", {})) != set(FEATURE_SPEC):
        raise PipelineError("Normalizer must contain exactly five features")
    for feature, direction in FEATURE_SPEC.items():
        entry = normalizer["features"][feature]
        lo, hi = entry.get("q05"), entry.get("q95")
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in [lo, hi])
            or lo > hi
            or entry.get("direction") != direction
            or type(entry.get("constant_reference")) is not bool
            or entry["constant_reference"] != (lo == hi)
        ):
            raise PipelineError(f"Invalid normalizer thresholds for {feature}")


def _validate_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(FEATURE_SPEC):
        raise PipelineError("Weights require exactly the five MCDA inputs")
    if any(
        type(w) not in (int, float) or not math.isfinite(w) or w < 0
        for w in weights.values()
    ) or not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise PipelineError("Weights must be finite, nonnegative and sum to one")


def score_mcda(
    features: pd.DataFrame, normalizer: dict, weights: dict[str, float]
) -> pd.DataFrame:
    """Preserve every road and QC; null incomplete/excluded scores, no imputation.

    rank=1 is highest score, ties ascending road_segment_id. percentile_rank
    spans 100 (first) to 0 (last); a singleton receives 100. These ordinal
    percentiles describe only this invocation's eligible roads. Spearman in
    sensitivity instead uses average score-tie ranks.
    """
    _validate_normalizer(normalizer)
    _validate_weights(weights)
    values, structure = _values_and_reasons(features)
    result = features.copy()
    risks = pd.DataFrame(index=features.index)
    constants = []
    for feature in FEATURE_SPEC:
        rule = normalizer["features"][feature]
        if rule["constant_reference"]:
            risk = pd.Series(0.5, index=features.index).where(values[feature].notna())
            constants.append(feature)
        else:
            risk = ((values[feature] - rule["q05"]) / (rule["q95"] - rule["q05"])).clip(
                0, 1
            )
            if rule["direction"] == "lower_is_higher_risk":
                risk = 1 - risk
        risks[feature] = risk
        result[f"risk_{feature}"] = risk
    missing = values.isna()
    result["missing_required_features"] = pd.Series(
        [
            ";".join(
                feature
                for feature, absent in zip(FEATURE_SPEC, row, strict=True)
                if absent
            )
            for row in missing.to_numpy()
        ],
        index=features.index,
        dtype=object,
    )
    result["constant_reference_features"] = ";".join(constants)
    complete = ~missing.any(axis=1)
    result["score_reason"] = np.where(complete, "scored", "missing_required_feature")
    result.loc[structure.ne(""), "score_reason"] = structure.loc[structure.ne("")]
    eligible = complete & structure.eq("")
    result["mcda_eligible"] = eligible
    result["score"] = (
        (risks.mul(pd.Series(weights)).sum(axis=1, min_count=5) * 100)
        .round(12)
        .where(eligible)
    )
    ordered = result.loc[eligible].sort_values(
        ["score", KEY], ascending=[False, True], kind="stable"
    )
    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["percentile_rank"] = np.nan
    if len(ordered):
        result.loc[ordered.index, "rank"] = np.arange(1, len(ordered) + 1)
        result.loc[ordered.index, "percentile_rank"] = (
            100 if len(ordered) == 1 else np.linspace(100, 0, len(ordered))
        )
    return result


def _top_length(scored, fraction):
    ordered = scored.loc[scored.mcda_eligible].sort_values("rank")
    total = float(ordered.road_length_m.sum())
    target = total * fraction
    # Include the whole boundary segment; publish its actual budget overshoot.
    count = (
        min(
            len(ordered),
            int(
                np.searchsorted(
                    ordered.road_length_m.cumsum().to_numpy(), target, side="left"
                )
            )
            + 1,
        )
        if len(ordered)
        else 0
    )
    selected = ordered.iloc[:count]
    length = float(selected.road_length_m.sum())
    return {
        "selected_road_segment_ids": selected[KEY].tolist(),
        "target_length_m": target,
        "selected_length_m": length,
        "overshoot_m": max(0.0, length - target),
        "actual_length_fraction": length / total if total else None,
    }


def compare_weight_sets(
    features: pd.DataFrame, normalizer: dict, weight_sets: dict
) -> dict:
    """Compare fixed schemes on identical eligible roads and length denominators.

    Top-k overlap is shared selected length / min(selected lengths), with the
    length-weighted Jaccard also reported. No fractional segments or labels.
    """
    if (
        not isinstance(weight_sets, dict)
        or not weight_sets
        or not all(isinstance(k, str) and k for k in weight_sets)
    ):
        raise PipelineError("weight_sets must be a nonempty named mapping")
    if (
        "road_length_m" not in features
        or not pd.api.types.is_numeric_dtype(features.road_length_m)
        or not (
            np.isfinite(features.road_length_m) & features.road_length_m.gt(0)
        ).all()
    ):
        raise PipelineError(
            "Sensitivity requires positive finite road_length_m for every road"
        )
    scores = {
        name: score_mcda(features, normalizer, weights)
        for name, weights in weight_sets.items()
    }
    first = next(iter(scores.values()))
    eligible_ids = sorted(first.loc[first.mcda_eligible, KEY].tolist())
    schemes = {}
    for name, scored in scores.items():
        if sorted(scored.loc[scored.mcda_eligible, KEY].tolist()) != eligible_ids:
            raise PipelineError("Sensitivity eligible sets differ")
        schemes[name] = {
            "weights": weight_sets[name],
            "eligible_count": len(eligible_ids),
            "top_k": {str(k): _top_length(scored, k) for k in TOP_K_FRACTIONS},
        }
    lengths = features.set_index(KEY).road_length_m
    pairs = {}
    for left, right in itertools.combinations(scores, 2):
        a = (
            scores[left]
            .set_index(KEY)
            .loc[eligible_ids, "score"]
            .rank(method="average")
        )
        b = (
            scores[right]
            .set_index(KEY)
            .loc[eligible_ids, "score"]
            .rank(method="average")
        )
        correlation = (
            float(a.corr(b))
            if len(a) > 1 and a.nunique() > 1 and b.nunique() > 1
            else None
        )
        top = {}
        for fraction in TOP_K_FRACTIONS:
            x, y = (schemes[name]["top_k"][str(fraction)] for name in (left, right))
            shared = set(x["selected_road_segment_ids"]) & set(
                y["selected_road_segment_ids"]
            )
            common = float(lengths.loc[sorted(shared)].sum())
            denominator = min(x["selected_length_m"], y["selected_length_m"])
            union = x["selected_length_m"] + y["selected_length_m"] - common
            top[str(fraction)] = {
                "intersection_length_m": common,
                "length_overlap_fraction": common / denominator
                if denominator
                else None,
                "length_jaccard": common / union if union else None,
            }
        pairs[f"{left}__{right}"] = {"spearman": correlation, "top_k": top}
    return {
        "eligible_road_segment_ids": eligible_ids,
        "eligible_count": len(eligible_ids),
        "eligible_length_m": float(lengths.loc[eligible_ids].sum()),
        "top_k_length_fractions": list(TOP_K_FRACTIONS),
        "schemes": schemes,
        "comparisons": pairs,
    }
