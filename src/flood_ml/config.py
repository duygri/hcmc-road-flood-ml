"""Validated shared settings and bounded paths, independent of process CWD."""

import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import yaml


class PipelineError(ValueError):
    """A configuration, input, or pipeline operation cannot safely proceed."""


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    sources: dict
    study_area: dict
    features: dict
    labeling: dict
    experiments: dict
    seed: int
    configuration_paths: tuple[Path, ...] = ()


def _relative_path(root: Path, value: str) -> Path:
    if (
        not isinstance(value, str)
        or not value.strip()
        or PureWindowsPath(value).drive
        or value.startswith(("/", "\\"))
        or ":" in value
        or "\x00" in value
    ):
        raise PipelineError(f"Expected a relative path: {value!r}")
    root = root.resolve()
    # Accept either separator on every OS; resolve follows existing symlinks/junctions.
    candidate = (root / value.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(root):
        raise PipelineError(f"Path escapes root {root}: {value!r}")
    return candidate


def resolve_input(raw_root: Path, asset_path: str) -> Path:
    """Resolve a read-only asset location inside the explicitly configured raw root.

    This resolver never creates files or opens a writable handle. Callers must
    use resolve_output for every output rather than writing to returned inputs.
    """
    return _relative_path(raw_root, asset_path)


def resolve_output(repo_root: Path, relative_path: str) -> Path:
    """Resolve an output within the worktree; reject absolute paths and traversal."""
    return _relative_path(repo_root, relative_path)


def _read_section(path: Path, name: str) -> dict:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get(name), dict):
        raise PipelineError(f"Config {path} requires mapping section {name!r}")
    return document[name]


def _validate_values(value, name: str) -> None:
    """Reject ambiguous YAML values and check generic numeric settings."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PipelineError(f"{name}: config keys must be strings")
            _validate_values(item, f"{name}.{key}")
    elif isinstance(value, list):
        for item in value:
            _validate_values(item, name)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise PipelineError(f"{name}: unsupported YAML value type")
    key = name.rsplit(".", 1)[-1]
    # A background count ratio is validated separately, and may exceed one.
    if name == "experiments.track_b.primary_background_ratio":
        return
    if key.endswith(("_m", "_fraction", "_ratio")):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise PipelineError(f"{name}: expected a finite nonnegative number")
        if key.endswith(("_fraction", "_ratio")) and value > 1:
            raise PipelineError(f"{name}: expected a number in [0, 1]")
    if isinstance(value, float) and not math.isfinite(value):
        raise PipelineError(f"{name}: nonfinite numbers are not allowed")


def _validate_track_b(config) -> None:
    """Keep the approved pilot settings fixed before any run is initialized."""
    expected = {
        "feature_columns": [
            "elevation_median_m",
            "relative_elevation_m",
            "slope_mean_deg",
            "dist_water_m",
            "builtup_ratio_250m",
        ],
        "primary_background_ratio": 10,
        "sensitivity_background_ratios": [5, 20],
        "seed_start": 42,
        "seed_repetitions": 20,
        "minimum_reference_events": 20,
        "logistic_regression": {"C": 1.0, "max_iter": 1000},
    }
    if not isinstance(config, dict) or set(config) != set(expected):
        raise PipelineError("experiments.track_b requires the predeclared settings")
    for key, value in expected.items():
        actual = config[key]
        valid_type = (
            type(actual) is int
            if type(value) is int
            else isinstance(actual, type(value))
        )
        if isinstance(value, list) and key == "sensitivity_background_ratios":
            valid_type = valid_type and all(type(item) is int for item in actual)
        if key == "logistic_regression":
            valid_type = (
                valid_type
                and set(actual) == {"C", "max_iter"}
                and type(actual["C"]) in (int, float)
                and type(actual["max_iter"]) is int
            )
        if not valid_type or actual != value:
            raise PipelineError(
                f"experiments.track_b.{key} must match the predeclared value"
            )


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id
    ):
        raise PipelineError(
            "run_id must contain 1–128 letters, digits, underscores or hyphens"
        )
    if run_id.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise PipelineError("run_id is a reserved Windows filename")


def load_settings(config_path: Path, *, repo_root: Path) -> Settings:
    """Load the shared bundle, optionally selecting a different section file.

    experiments.yaml owns run_id, seed, and links relative to repo_root. A CLI
    can enter through any of the five named YAML files. Local sources.yaml is
    a shallow source override, used for explicit external raw_root only.
    """
    repo_root = repo_root.resolve()
    config_path = Path(config_path)
    if config_path.is_absolute():
        selected = config_path.resolve()
        if not selected.is_relative_to(repo_root):
            raise PipelineError("Config must be inside repo_root")
    else:
        selected = resolve_output(repo_root, str(config_path))
    name = selected.stem
    sections = ("sources", "study_area", "features", "labeling")
    if name not in (*sections, "experiments"):
        raise PipelineError(f"Unsupported configuration entry point: {name}")
    experiments_path = (
        selected
        if name == "experiments"
        else resolve_output(repo_root, "configs/experiments.yaml")
    )
    experiments = _read_section(experiments_path, "experiments")
    seed = experiments.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise PipelineError("experiments.seed must be an integer in [0, 2**32)")
    _validate_run_id(experiments.get("run_id"))
    links = experiments.get("configs")
    if not isinstance(links, dict) or set(links) != set(sections):
        raise PipelineError("experiments.configs must link all four shared sections")
    loaded = {}
    for section in sections:
        path = resolve_output(repo_root, links[section])
        loaded[section] = _read_section(selected if name == section else path, section)
    sources = loaded["sources"]
    # Validate the tracked default before applying an explicit local override.
    resolve_output(repo_root, sources.get("raw_root"))
    local = resolve_output(repo_root, "configs/local/sources.yaml")
    if local.exists():
        sources.update(_read_section(local, "sources"))
    labeling = loaded["labeling"]
    for review_key, template_key in (
        ("match_reviews", "match_reviews_template"),
        ("event_reviews", "event_reviews_template"),
    ):
        template = labeling.get(template_key)
        if (
            not labeling.get(review_key)
            and isinstance(template, str)
            and template.strip()
        ):
            candidate = resolve_output(repo_root, template)
            if candidate.is_file():
                # The local header-only files are the editable review inputs.
                # Their content hash then participates in the run fingerprint.
                labeling[review_key] = template
    raw_value = sources.get("raw_root")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise PipelineError("sources.raw_root must be a nonempty path string")
    raw_root = Path(raw_value)
    raw_root = (
        raw_root.resolve()
        if raw_root.is_absolute()
        else resolve_output(repo_root, raw_value)
    )
    assets = sources.get("assets")
    if not isinstance(assets, list):
        raise PipelineError("sources.assets must be a list")
    paths = []
    for asset in assets:
        if not isinstance(asset, dict):
            raise PipelineError("Every source asset must be a mapping")
        paths.append(resolve_input(raw_root, asset.get("path")))
    if len(set(paths)) != len(paths):
        raise PipelineError("Duplicate source asset path")
    outputs = experiments.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "interim",
        "processed",
        "reports",
        "artifacts",
    }:
        raise PipelineError(
            "experiments.outputs requires interim, processed, reports and artifacts"
        )
    for value in outputs.values():
        output = resolve_output(repo_root, value)
        if output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
            raise PipelineError("Output directories must not overlap raw_root")
    if "track_b" in experiments:
        _validate_track_b(experiments["track_b"])
    for section, values in {**loaded, "experiments": experiments}.items():
        _validate_values(values, section)
        if "seed" in values and (
            type(values["seed"]) is not int or values["seed"] != seed
        ):
            raise PipelineError(f"{section}.seed must match experiments.seed")
        if "run_id" in values and values["run_id"] != experiments["run_id"]:
            raise PipelineError(f"{section}.run_id must match experiments.run_id")
    crs = loaded["study_area"].get("target_crs")
    if not isinstance(crs, str) or not crs.strip():
        raise PipelineError("study_area.target_crs must be a nonempty string")
    if loaded["labeling"].get("positive_requires_review") is not True:
        raise PipelineError("labeling.positive_requires_review must be boolean true")
    sources["raw_root"] = raw_root
    configuration_paths = {
        selected,
        experiments_path,
        *(resolve_output(repo_root, value) for value in links.values()),
    }
    if local.exists():
        configuration_paths.add(local)
    return Settings(
        repo_root=repo_root,
        **loaded,
        experiments=experiments,
        seed=seed,
        configuration_paths=tuple(sorted(configuration_paths)),
    )
