"""OSM-node topology, polygon clipping and deterministic metric segmentation."""

import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import date

import geopandas as gpd
from shapely import prepare
from shapely.geometry import LineString, Point
from shapely.ops import substring

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import _require_polygon, require_metric_crs

SEGMENT_COLUMNS = [
    "road_segment_id",
    "osm_way_id",
    "road_name",
    "highway_type",
    "length_m",
    "geometry",
    "geometry_crs",
    "osm_snapshot_date",
    "study_area_id",
    "node_ids",
    "tags",
    "bridge",
    "tunnel",
    "layer",
    "chain_id",
    "chainage_start_m",
    "chainage_end_m",
]
CANONICAL_DECIMALS = (
    6  # 1 micrometre in the metric CRS; hash only, never snap geometry.
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _coordinates(line):
    return tuple(
        tuple(round(float(v), CANONICAL_DECIMALS) for v in xy[:2]) for xy in line.coords
    )


def _canonical(line):
    coords = list(line.coords)
    if coords[0] == coords[-1]:
        # Closed ways have no natural start. Rotate each direction to its
        # lexicographically smallest vertex before selecting an orientation.
        ring = coords[:-1]
        candidates = []
        for direction in (ring, ring[::-1]):
            minimum = min(direction)
            for i, xy in enumerate(direction):
                if xy == minimum:
                    rotated = direction[i:] + direction[:i]
                    candidates.append(rotated + rotated[:1])
        coords = min(candidates)
    elif _coordinates(line)[::-1] < _coordinates(line):
        coords.reverse()
    return LineString(coords)


def _active(value):
    return value is not None and str(value).strip().lower() not in {
        "",
        "no",
        "false",
        "0",
        "nan",
    }


def _level(row):
    layer = row.get("layer")
    layer = "0" if layer is None or str(layer).lower() == "nan" else str(layer).strip()
    # Explicit layer=0 and absent layer describe the same ordinary level.
    return (_active(row.get("bridge")), _active(row.get("tunnel")), layer)


def _lines(geometry):
    if geometry.geom_type == "LineString":
        if geometry.length > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _lines(part)


def _clip_traversal(chain, polygon):
    """Clip single source edges and join only consecutive source intervals.

    Whole-LineString intersection nodes self-crossings and dissolves retraced
    portions. Each edge has no such ambiguity. Source chainage distinguishes
    genuine consecutive fragments from a later return to the same coordinate.
    The caller canonicalizes source direction before geometric calculations,
    so reversing the source cannot introduce different clipping roundoff.
    """
    if polygon.covers(chain):
        yield chain, 0.0, chain.length
        return
    pending = []
    pending_start = None
    previous_end = None
    traversed = 0.0
    for start, end in itertools.pairwise(chain.coords):
        edge = LineString([start, end])
        if edge.length == 0:
            continue
        fragments = []
        for part in _lines(edge.intersection(polygon)):
            coords = list(part.coords)
            lo = edge.project(Point(coords[0]))
            hi = edge.project(Point(coords[-1]))
            if lo > hi:
                lo, hi = hi, lo
                coords.reverse()
            fragments.append((lo, hi, coords))
        for lo, hi, coords in sorted(fragments):
            source_start, source_end = traversed + lo, traversed + hi
            if pending and previous_end == source_start and pending[-1] == coords[0]:
                pending.extend(coords[1:])
            else:
                if pending:
                    yield LineString(pending), pending_start, previous_end
                pending = coords
                pending_start = source_start
            previous_end = source_end
        traversed += edge.length
    if pending:
        yield LineString(pending), pending_start, previous_end


def segment_roads(
    ways: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    *,
    snapshot_date: str,
    target_m: float,
    maximum_m: float,
) -> gpd.GeoDataFrame:
    """Split shared same-level OSM nodes, clip exact AOI, then divide evenly.

    Length and chainage use unrounded projected geometry. Each clipped topology
    chain gets ceil(length/target_m) equal pieces; target <= maximum is required.
    Boundary clipping preserves source edge traversal; geometric self-crossings
    never introduce junctions. Short chains/fragments survive. Node refs/tags
    retain the *whole source way*.
    Chainage starts at zero on each clipped, canonically oriented chain.

    IDs are SHA-256 of snapshot, source way ID, canonical chain coordinate
    sequence rounded to 6 decimals, segment chainage rounded to 6 decimals, and
    a config digest (algorithm version, precision, metric CRS, target, maximum).
    Structure level and the clipped interval's chainage along the canonically
    directed full source way are also part of chain identity. This source
    interval distinguishes repeated traversal of otherwise identical geometry.
    Source direction compares rounded full coordinate sequences, with original
    node IDs breaking symmetric-geometry ties. Hash rounding is not a
    promise of stability across precision-cell boundaries or changed snapshots.
    Canonical orientation and sorted output remove input order/direction effects.
    """
    require_metric_crs(aoi)
    require_metric_crs(ways)
    _require_polygon(aoi)
    if ways.crs != aoi.crs:
        raise PipelineError("Ways and AOI must share the same projected CRS")
    for value in (target_m, maximum_m):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise PipelineError("Segmentation lengths must be finite and positive")
    if target_m > maximum_m:
        raise PipelineError("target_m must not exceed maximum_m")
    try:
        if (
            not isinstance(snapshot_date, str)
            or date.fromisoformat(snapshot_date).isoformat() != snapshot_date
        ):
            raise ValueError
    except ValueError as exc:
        raise PipelineError("snapshot_date must be an ISO YYYY-MM-DD date") from exc
    if not {"osm_way_id", "node_ids", "highway_type"} <= set(ways):
        raise PipelineError(
            "OSM ways require source IDs, node references and highway type"
        )
    if ways.osm_way_id.isna().any() or ways.osm_way_id.astype(str).duplicated().any():
        raise PipelineError("OSM way IDs must be nonnull and unique")
    config_digest = _hash(
        {
            "version": 3,
            "precision": CANONICAL_DECIMALS,
            "crs": aoi.crs.to_string(),
            "target_m": float(target_m),
            "maximum_m": float(maximum_m),
        }
    )
    records = ways.to_dict(
        "records"
    )  # AOI/context subset only, never the national source.
    shared = Counter()
    for row in records:
        line, nodes = row["geometry"], row["node_ids"]
        if (
            line is None
            or line.is_empty
            or not line.is_valid
            or line.geom_type != "LineString"
            or not isinstance(nodes, (list, tuple))
            or len(nodes) != len(line.coords)
        ):
            raise PipelineError(
                "Each OSM LineString requires one source node ID per vertex"
            )
        level = _level(row)
        shared.update((node, level) for node in nodes)
    polygon = aoi.geometry.iloc[0]
    prepare(polygon)
    area_id = aoi.iloc[0].get("study_area_id", aoi.attrs.get("study_area_id"))
    if not isinstance(area_id, str) or not area_id:
        raise PipelineError(
            "AOI requires an explicit study_area_id column or attribute"
        )
    output = []
    for row in records:
        coords = list(row["geometry"].coords)
        nodes = list(row["node_ids"])
        source_key = (_coordinates(row["geometry"]), tuple(map(str, nodes)))
        if (source_key[0][::-1], source_key[1][::-1]) < source_key:
            coords.reverse()
            nodes.reverse()
        source_distances = [0.0]
        for a, b in itertools.pairwise(coords):
            source_distances.append(
                source_distances[-1] + math.hypot(b[0] - a[0], b[1] - a[1])
            )
        level = _level(row)
        cuts = (
            [0]
            + [i for i, node in enumerate(nodes[1:-1], 1) if shared[node, level] > 1]
            + [len(coords) - 1]
        )
        for start, end in itertools.pairwise(cuts):
            topology_chain = LineString(coords[start : end + 1])
            if not polygon.intersects(topology_chain):
                continue
            for clipped, source_lo, source_hi in _clip_traversal(
                topology_chain, polygon
            ):
                chain = _canonical(clipped)
                chain_key = _coordinates(chain)
                source_interval = (
                    round(source_distances[start] + source_lo, CANONICAL_DECIMALS),
                    round(source_distances[start] + source_hi, CANONICAL_DECIMALS),
                )
                chain_id = _hash(
                    [
                        snapshot_date,
                        str(row["osm_way_id"]),
                        chain_key,
                        source_interval,
                        level,
                        config_digest,
                    ]
                )
                count = math.ceil(chain.length / target_m)
                for i in range(count):
                    lo, hi = chain.length * i / count, chain.length * (i + 1) / count
                    piece = substring(chain, lo, hi)
                    road_id = _hash(
                        [
                            chain_id,
                            round(lo, CANONICAL_DECIMALS),
                            round(hi, CANONICAL_DECIMALS),
                        ]
                    )
                    output.append(
                        {
                            "road_segment_id": road_id,
                            "osm_way_id": str(row["osm_way_id"]),
                            "road_name": row.get("name"),
                            "highway_type": row["highway_type"],
                            "length_m": piece.length,
                            "geometry": piece,
                            "geometry_crs": aoi.crs.to_string(),
                            "osm_snapshot_date": snapshot_date,
                            "study_area_id": area_id,
                            "node_ids": list(row["node_ids"]),
                            "tags": row.get("tags", {}),
                            "bridge": row.get("bridge"),
                            "tunnel": row.get("tunnel"),
                            "layer": row.get("layer"),
                            "chain_id": chain_id,
                            "chainage_start_m": lo,
                            "chainage_end_m": hi,
                            "_chain_key": chain_key,
                            "_source_interval": source_interval,
                        }
                    )
    output.sort(
        key=lambda row: (
            row["osm_way_id"],
            row["_chain_key"],
            row["_source_interval"],
            row["chainage_start_m"],
        )
    )
    result = gpd.GeoDataFrame(
        output, columns=SEGMENT_COLUMNS, geometry="geometry", crs=aoi.crs
    )
    if not result.road_segment_id.is_unique:
        raise PipelineError(
            "Canonical segment ID collision; inspect overlapping source chains"
        )
    result.attrs["segmentation_config_digest"] = config_digest
    return result
