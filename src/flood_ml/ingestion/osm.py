"""Stream OSM ways with their original node topology and tags."""

import time
from pathlib import Path

import geopandas as gpd
import osmium
from pyproj import Transformer
from shapely import prepare
from shapely.geometry import LineString

from flood_ml.config import PipelineError
from flood_ml.gis.study_area import _require_polygon, require_metric_crs

WAY_COLUMNS = [
    "osm_way_id",
    "node_ids",
    "highway_type",
    "name",
    "bridge",
    "tunnel",
    "layer",
    "tags",
    "geometry",
]


def read_osm_roads(
    path: Path, context: gpd.GeoDataFrame, allowed_highways: list[str]
) -> gpd.GeoDataFrame:
    """Retain complete ways intersecting metric context; never clip node lists.

    A compact C++ sparse location index stores country-wide node locations,
    avoiding a Python node dictionary or a country-wide GeoDataFrame. Only
    allowlisted ways whose envelope overlaps the context are constructed in
    Python, and only actual context intersections are retained. Complete node
    locations ensure crossing ways with both endpoints outside are included.
    No country-wide feature table or Python node dictionary is built. The
    sparse_mem_array index avoids libosmium's Windows disk-index handle leak.
    Input files are opened only by the read-only OSM reader.
    """
    require_metric_crs(context)
    _require_polygon(context)
    if (
        not isinstance(allowed_highways, list)
        or not allowed_highways
        or any(not isinstance(value, str) or not value for value in allowed_highways)
    ):
        raise PipelineError("allowed_highways must be a nonempty list of strings")
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"OSM source missing: {path}")
    allowed = set(allowed_highways)
    xmin, ymin, xmax, ymax = context.to_crs(4326).total_bounds
    polygon = context.geometry.iloc[0]
    prepare(polygon)
    project = Transformer.from_crs(4326, context.crs, always_xy=True)
    rows = []
    stats = {
        "ways_seen": 0,
        "allowlisted_ways": 0,
        "envelope_candidates": 0,
        "context_ways": 0,
        "degenerate_ways": 0,
        "location_index": "sparse_mem_array",
    }

    class Roads(osmium.SimpleHandler):
        def way(self, way):
            stats["ways_seen"] += 1
            highway = way.tags.get("highway")
            if highway not in allowed:
                return
            stats["allowlisted_ways"] += 1
            coords = [(node.lon, node.lat) for node in way.nodes]
            if len(coords) < 2:
                stats["degenerate_ways"] += 1
                return
            xs, ys = zip(*coords)
            if max(xs) < xmin or min(xs) > xmax or max(ys) < ymin or min(ys) > ymax:
                return
            stats["envelope_candidates"] += 1
            x, y = project.transform(xs, ys)
            line = LineString(zip(x, y))
            if line.is_empty or line.length == 0:
                stats["degenerate_ways"] += 1
                return
            if not line.is_valid:
                raise PipelineError(f"Invalid OSM geometry for way {way.id}")
            if line.intersects(polygon):
                tags = dict(way.tags)
                rows.append(
                    {
                        "osm_way_id": str(way.id),
                        "node_ids": [node.ref for node in way.nodes],
                        "highway_type": highway,
                        "name": tags.get("name"),
                        "bridge": tags.get("bridge"),
                        "tunnel": tags.get("tunnel"),
                        "layer": tags.get("layer"),
                        "tags": tags,
                        "geometry": line,
                    }
                )
                stats["context_ways"] += 1

    started = time.perf_counter()
    try:
        Roads().apply_file(str(path), locations=True, idx="sparse_mem_array")
    except (OSError, RuntimeError, osmium.InvalidLocationError) as exc:
        raise PipelineError(
            f"Cannot read OSM ways with complete node locations: {exc}"
        ) from exc
    stats["elapsed_seconds"] = time.perf_counter() - started
    result = gpd.GeoDataFrame(
        rows, columns=WAY_COLUMNS, geometry="geometry", crs=context.crs
    )
    result.attrs["read_stats"] = stats
    return result
