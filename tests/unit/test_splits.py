"""Spatial split behavior, with hand-checked grid and reviewed-link fixtures."""

from copy import deepcopy

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from flood_ml.config import PipelineError
from flood_ml.evaluation.splits import assign_spatial_partitions


def roads_case(lines):
    return gpd.GeoDataFrame(
        {"road_segment_id": list(lines)}, geometry=list(lines.values()), crs="EPSG:9210"
    )


def confirmed(*rows):
    return pd.DataFrame(
        [
            {
                "observation_id": f"o{i}",
                "road_segment_id": road,
                "event_group_id": event,
                "location_group_id": location,
                "source_id": f"synthetic-source-{i}",
                "evidence": "Synthetic reviewed fixture",
            }
            for i, (road, event, location) in enumerate(rows)
        ],
        columns=[
            "observation_id",
            "road_segment_id",
            "event_group_id",
            "location_group_id",
            "source_id",
            "evidence",
        ],
    )


@pytest.fixture
def split_case():
    return {
        "roads": roads_case(
            {
                f"r{i}": LineString([(2500 + 5000 * i, 2500), (2600 + 5000 * i, 2500)])
                for i in range(12)
            }
        ),
        "confirmed_event_segments": confirmed(),
        "config": {
            "experiments": {
                "seed": 42,
                "spatial_block_m": 5000,
                "holdout_fraction": 0.2,
                "guard_m": 1000,
            },
            "features": {
                "neighborhood_radius_m": 500,
                "dem_buffer_m": 30,
                "landcover_buffer_m": 250,
                "water_buffer_m": 250,
            },
        },
    }


def test_hash_selected_grid_is_frozen_without_quota_adjustment(split_case):
    result = assign_spatial_partitions(**split_case).set_index("road_segment_id")
    # SHA-256 threshold selects these blocks, rather than selecting exactly 20%.
    assert set(result.index[result.partition.eq("holdout")]) == {"r2", "r3", "r6", "r7"}
    assert result.loc["r0", "block_id"] == "0:0"
    assert result.loc["r4", "partition"] == "reference"
    assert result.reference_fit_eligible.equals(result.partition.eq("reference"))
    assert result.evaluation_eligible.all()


def test_spatial_split_is_deterministic(split_case):
    first = assign_spatial_partitions(**split_case).sort_values("road_segment_id")
    split_case["roads"] = split_case["roads"].iloc[::-1]
    second = assign_spatial_partitions(**split_case).sort_values("road_segment_id")
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
    assert first.attrs == second.attrs


def test_labels_scores_and_review_counts_cannot_select_holdout_blocks(split_case):
    first = assign_spatial_partitions(**split_case)
    split_case["roads"]["label_status"] = "positive"
    split_case["roads"]["score"] = list(range(12))
    split_case["confirmed_event_segments"] = confirmed(
        ("r0", "e0", "l0"), ("r2", "e0", "l0")
    )
    second = assign_spatial_partitions(**split_case)
    pd.testing.assert_frame_equal(
        first[["road_segment_id", "block_id", "block_partition"]],
        second[["road_segment_id", "block_id", "block_partition"]],
    )
    assert first.attrs["block_assignments"] == second.attrs["block_assignments"]
    assert second.set_index("road_segment_id").loc["r2", "partition"] == "excluded"


def test_full_line_contact_is_guarded_without_promoting_reference_anchor(split_case):
    split_case["roads"] = roads_case(
        {
            "crossing": LineString([(6500, 2500), (10100, 2500)]),
            "near": LineString([(9400, 3000), (9500, 3000)]),
            "far": LineString([(7000, 3500), (7100, 3500)]),
            "hold": LineString([(12500, 2500), (12600, 2500)]),
        }
    )
    result = assign_spatial_partitions(**split_case).set_index("road_segment_id")
    assert result.loc["crossing", "block_id"] == "1:0"
    assert result.loc["crossing", "block_partition"] == "reference"
    assert result.loc["crossing", "partition"] == "guard"
    assert "holdout_block_contact" in result.loc["crossing", "exclusion_reason"]
    assert result.loc["near", "partition"] == "guard"
    assert result.loc["far", "partition"] == "reference"
    assert not result.loc[["crossing", "near"], "reference_fit_eligible"].any()
    assert not result.loc[["crossing", "near"], "evaluation_eligible"].any()


def test_holdout_full_line_extension_cannot_share_reference_feature_support(split_case):
    split_case["roads"] = roads_case(
        {
            "hold": LineString([(8000, 2500), (17000, 2500)]),
            "close": LineString([(7500, 2600), (7600, 2600)]),
            "touch": LineString([(6900, 2500), (7000, 2500)]),
            "safe": LineString([(6700, 2500), (6800, 2500)]),
        }
    )
    result = assign_spatial_partitions(**split_case).set_index("road_segment_id")
    assert result.loc["hold", "partition"] == "holdout"
    assert result.loc["close", "partition"] == "guard"
    assert result.loc["touch", "partition"] == "guard"
    assert result.loc["safe", "partition"] == "reference"
    geometry = split_case["roads"].set_index("road_segment_id").geometry
    reference = geometry.loc[result.partition.eq("reference")].buffer(500).union_all()
    holdout = geometry.loc[result.partition.eq("holdout")].buffer(500).union_all()
    assert reference.disjoint(holdout)


@pytest.mark.parametrize(
    "radius_key",
    ["neighborhood_radius_m", "landcover_buffer_m", "water_buffer_m", "dem_buffer_m"],
)
def test_larger_feature_radius_grows_guard_even_if_guard_config_is_small(
    split_case, radius_key
):
    split_case["roads"] = roads_case(
        {
            "reference": LineString([(8300, 2500), (8400, 2500)]),
            "hold": LineString([(12500, 2500), (12600, 2500)]),
        }
    )
    first = assign_spatial_partitions(**split_case).set_index("road_segment_id")
    assert first.loc["reference", "partition"] == "reference"
    split_case["config"]["features"][radius_key] = 1000
    split_case["config"]["experiments"]["guard_m"] = 10
    second = assign_spatial_partitions(**split_case).set_index("road_segment_id")
    assert second.loc["reference", "partition"] == "guard"
    assert second.attrs["effective_guard_m"] == 2000
    assert second.attrs["feature_support_radius_m"] == 1000


@pytest.mark.parametrize(
    "links",
    [
        (("r0", "same-event", "place"), ("r2", "same-event", "place")),
        (("r0", "event-a", "same-place"), ("r2", "event-b", "same-place")),
        (
            ("r0", "event-a", "place"),
            ("r4", "event-b", "place"),
            ("r2", "event-b", "place"),
        ),
    ],
)
def test_connected_reviewed_groups_crossing_partitions_are_excluded_and_audited(
    split_case, links
):
    split_case["confirmed_event_segments"] = confirmed(*links)
    result = assign_spatial_partitions(**split_case).set_index("road_segment_id")
    affected = {row[0] for row in links}
    assert set(result.index) == set(split_case["roads"].road_segment_id)
    assert result.loc[list(affected), "partition"].eq("excluded").all()
    assert (
        result.loc[list(affected), "exclusion_reason"]
        .str.contains("reviewed_group_crosses_partitions")
        .all()
    )
    assert not result.loc[list(affected), "reference_fit_eligible"].any()
    assert not result.loc[list(affected), "evaluation_eligible"].any()
    audit = pd.DataFrame(result.attrs["confirmed_event_audit"])
    assert len(audit) == len(links)
    assert set(audit.observation_id) == set(
        split_case["confirmed_event_segments"].observation_id
    )
    assert audit.evidence.eq("Synthetic reviewed fixture").all()
    assert audit.group_component_id.nunique() == 1
    assert audit.partition.eq("excluded").all()
    reverse = dict(
        split_case,
        roads=split_case["roads"].iloc[::-1],
        confirmed_event_segments=split_case["confirmed_event_segments"].iloc[::-1],
    )
    other = assign_spatial_partitions(**reverse).set_index("road_segment_id")
    pd.testing.assert_frame_equal(result.sort_index(), other.sort_index())
    assert result.attrs == other.attrs


def test_retained_reference_and_holdout_event_and_location_sets_are_disjoint(
    split_case,
):
    split_case["confirmed_event_segments"] = confirmed(
        ("r0", "e0", "l0"),
        ("r1", "e1", "l1"),
        ("r2", "e2", "l2"),
        ("r3", "e3", "l3"),
        ("r4", "cross", "cross"),
        ("r6", "cross", "cross"),
    )
    result = assign_spatial_partitions(**split_case)
    audit = pd.DataFrame(result.attrs["confirmed_event_audit"])
    for key in ["event_group_id", "location_group_id"]:
        ref = set(audit.loc[audit.partition.eq("reference"), key])
        hold = set(audit.loc[audit.partition.eq("holdout"), key])
        assert ref and hold and ref.isdisjoint(hold)


def test_guard_member_excludes_entire_event_from_evaluation(split_case):
    split_case["roads"] = roads_case(
        {
            "guard": LineString([(9400, 2500), (9500, 2500)]),
            "far": LineString([(2500, 2500), (2600, 2500)]),
        }
    )
    split_case["confirmed_event_segments"] = confirmed(
        ("guard", "event", "place"), ("far", "event", "place")
    )
    result = assign_spatial_partitions(**split_case).set_index("road_segment_id")
    assert result.partition.eq("excluded").all()
    assert result.exclusion_reason.str.contains("reviewed_group_intersects_guard").all()


def test_empty_input_preserves_schema_and_has_no_confirmed_audit(split_case):
    split_case["roads"] = split_case["roads"].iloc[:0]
    result = assign_spatial_partitions(**split_case)
    assert result.empty
    assert {"road_segment_id", "block_id", "partition", "exclusion_reason"}.issubset(
        result
    )
    assert result.attrs["confirmed_event_audit"] == []


@pytest.mark.parametrize(
    "change",
    ["duplicate_id", "missing_id", "geographic", "point", "empty", "missing_geometry"],
)
def test_invalid_road_inputs_fail_before_spatial_assignment(split_case, change):
    roads = split_case["roads"]
    if change == "duplicate_id":
        roads.loc[1, "road_segment_id"] = roads.loc[0, "road_segment_id"]
    elif change == "missing_id":
        roads.loc[0, "road_segment_id"] = None
    elif change == "geographic":
        split_case["roads"] = roads.to_crs(4326)
    elif change == "point":
        roads.loc[0, "geometry"] = Point(2500, 2500)
    elif change == "empty":
        roads.loc[0, "geometry"] = LineString()
    else:
        roads.loc[0, "geometry"] = None
    with pytest.raises(PipelineError):
        assign_spatial_partitions(**split_case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("spatial_block_m", 0),
        ("spatial_block_m", float("nan")),
        ("holdout_fraction", -0.1),
        ("holdout_fraction", 1.1),
        ("guard_m", -1),
        ("seed", True),
    ],
)
def test_invalid_split_config_rejected(split_case, key, value):
    split_case["config"]["experiments"][key] = value
    with pytest.raises(PipelineError):
        assign_spatial_partitions(**split_case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("road_segment_id", "absent"),
        ("event_group_id", None),
        ("location_group_id", " "),
        ("observation_id", None),
    ],
)
def test_invalid_confirmed_links_rejected(split_case, key, value):
    links = confirmed(("r0", "e0", "l0"))
    links.loc[0, key] = value
    split_case["confirmed_event_segments"] = links
    with pytest.raises(PipelineError):
        assign_spatial_partitions(**split_case)


def test_inputs_not_mutated(split_case):
    roads = split_case["roads"].copy(deep=True)
    links = split_case["confirmed_event_segments"].copy(deep=True)
    config = deepcopy(split_case["config"])
    assign_spatial_partitions(**split_case)
    pd.testing.assert_frame_equal(roads, split_case["roads"])
    pd.testing.assert_frame_equal(links, split_case["confirmed_event_segments"])
    assert config == split_case["config"]


@pytest.mark.parametrize("fraction,expected", [(0, "reference"), (1, "holdout")])
def test_fraction_endpoints_do_not_create_spurious_guards(
    split_case, fraction, expected
):
    split_case["config"]["experiments"]["holdout_fraction"] = fraction
    result = assign_spatial_partitions(**split_case)
    assert result.partition.eq(expected).all()


def test_negative_grid_coordinates_use_floor_and_flat_config_is_supported(split_case):
    split_case["roads"] = roads_case({"negative": LineString([(-90, -90), (-10, -10)])})
    split_case["config"] = {
        "seed": 42,
        "spatial_block_m": 5000,
        "holdout_fraction": 0,
        "guard_m": 1000,
    }
    result = assign_spatial_partitions(**split_case)
    assert result.block_id.tolist() == ["-1:-1"]


def test_missing_feature_config_does_not_silently_drop_default_support(split_case):
    split_case["config"]["features"] = {}
    split_case["config"]["experiments"]["guard_m"] = 0
    result = assign_spatial_partitions(**split_case)
    assert result.attrs["effective_guard_m"] == 1000


@pytest.mark.parametrize("radius", [-1, float("nan"), True])
def test_invalid_feature_support_radius_rejected(split_case, radius):
    split_case["config"]["features"]["neighborhood_radius_m"] = radius
    with pytest.raises(PipelineError):
        assign_spatial_partitions(**split_case)


def test_exact_guard_contact_with_empty_left_holdout_cell_is_included(split_case):
    # Holdout cell 3:0 ends at x=20000. The reference line is exactly 1000 m
    # away; the closed guard must include the cell even with no holdout roads.
    split_case["roads"] = roads_case(
        {"touch": LineString([(21000, 2500), (21100, 2500)])}
    )
    result = assign_spatial_partitions(**split_case)
    assert result.block_id.tolist() == ["4:0"]
    assert result.partition.tolist() == ["guard"]
