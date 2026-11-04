"""Read the IRD GeoPackage with stable feature keys and uncoerced date text."""

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd

from flood_ml.config import PipelineError


def read_ird(path: Path) -> gpd.GeoDataFrame:
    """Join raw attributes to geometry through GeoPackage OBJECTID/fid.

    Native ``Obs_id`` is not a record key in the current snapshot; it repeats.
    The stable key uses the one-to-one OBJECTID/fid pair while retaining all
    original fields and the raw date text for provenance.
    """
    path = Path(path).resolve()
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            attrs = pd.read_sql_query("SELECT * FROM Flood_event_locations", connection)
            row = connection.execute(
                "SELECT column_name FROM gpkg_geometry_columns "
                "WHERE table_name = 'Flood_event_locations'"
            ).fetchone()
        if row is None:
            raise PipelineError("IRD layer has no registered geometry column")
        attrs = attrs.drop(columns=[row[0]])
        geometry = gpd.read_file(
            path, layer="Flood_event_locations", columns=[], fid_as_index=True
        ).reset_index()
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise PipelineError(f"Cannot read IRD GeoPackage {path}: {exc}") from exc
    if (
        "OBJECTID" not in attrs
        or attrs.OBJECTID.isna().any()
        or attrs.OBJECTID.duplicated().any()
        or set(attrs.OBJECTID) != set(geometry.fid)
    ):
        raise PipelineError(
            "IRD OBJECTID/fid requires complete one-to-one geometry join"
        )
    raw = gpd.GeoDataFrame(
        attrs.merge(
            geometry, left_on="OBJECTID", right_on="fid", validate="one_to_one"
        ),
        crs=geometry.crs,
    )
    raw["observation_id"] = "ird:flood_observations:fid:" + raw.OBJECTID.astype(str)
    return raw
