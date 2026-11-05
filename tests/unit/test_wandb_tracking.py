"""Synthetic-only tests of the optional W&B reporting contract."""

import builtins
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def logger():
    return importlib.import_module("flood_ml.utils.wandb_tracking").log_stage_report


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-key")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    run = SimpleNamespace(summary={}, log=Mock(), finish=Mock())
    sdk = SimpleNamespace(init=Mock(return_value=run), Settings=Mock(side_effect=dict))
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk, run


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_key_skips_without_import(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
    else:
        monkeypatch.setenv("WANDB_API_KEY", key)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            pytest.fail("W&B must not be imported without credentials")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert (
        logger()(run_id="synthetic", stage="track-b", config={}, summary={}, metrics={})
        == "skipped_no_api_key"
    )


@pytest.mark.parametrize("stage", ["track-b", "evaluation"])
def test_logs_named_group_config_summary_and_metrics(service, stage):
    sdk, run = service
    status = logger()(
        run_id="synthetic_001",
        stage=stage,
        config={"seed": 42, "rules": {"draws": 3}},
        summary={"status": "pending_positive_review", "evaluation_performed": False},
        metrics={"rows": 12, "capture": {"top_10": 0.75}},
    )
    assert status == "logged"
    options = sdk.init.call_args.kwargs
    assert {
        key: options[key]
        for key in ("entity", "project", "name", "group", "job_type", "config")
    } == {
        "entity": "duygri-saigon-university-sgu-",
        "project": "hcmc-road-flood-ml",
        "name": f"synthetic_001-{stage}",
        "group": "synthetic_001",
        "job_type": stage,
        "config": {"seed": 42, "rules": {"draws": 3}},
    }
    assert run.summary == {
        "status": "pending_positive_review",
        "evaluation_performed": False,
    }
    run.log.assert_called_once_with({"rows": 12, "capture/top_10": 0.75})
    run.finish.assert_called_once_with()
    assert options["settings"]["disable_code"] is True
    assert options["settings"]["disable_git"] is True
    assert options["settings"]["console"] == "off"
    assert options["settings"]["x_disable_meta"] is True
    assert options["settings"]["x_disable_stats"] is True


def test_environment_overrides(service, monkeypatch):
    sdk, _ = service
    monkeypatch.setenv("WANDB_ENTITY", "synthetic-team")
    monkeypatch.setenv("WANDB_PROJECT", "synthetic-project")
    assert (
        logger()(
            run_id="synthetic", stage="evaluation", config={}, summary={}, metrics={}
        )
        == "logged"
    )
    assert sdk.init.call_args.kwargs["entity"] == "synthetic-team"
    assert sdk.init.call_args.kwargs["project"] == "synthetic-project"


def test_filters_non_numeric_nonfinite_and_list_metrics(service):
    _, run = service
    metrics = {
        "zero": 0,
        "negative": -2,
        "fraction": 0.5,
        "large": 10**400,
        "true": True,
        "false": False,
        "none": None,
        "string": "private/input.csv",
        "nan": float("nan"),
        "positive_inf": float("inf"),
        "negative_inf": -float("inf"),
        "list": [1, {"raw_row": 2}],
        "nested": {"good": 3, "bad": None, "deeper": {"value": 4.5}},
    }
    assert (
        logger()(
            run_id="synthetic",
            stage="evaluation",
            config={},
            summary={},
            metrics=metrics,
        )
        == "logged"
    )
    run.log.assert_called_once_with(
        {
            "zero": 0,
            "negative": -2,
            "fraction": 0.5,
            "large": 10**400,
            "nested/good": 3,
            "nested/deeper/value": 4.5,
        }
    )
    assert metrics["true"] is True


def test_empty_metrics_do_not_fabricate_measurements(service):
    _, run = service
    assert (
        logger()(
            run_id="synthetic",
            stage="track-b",
            config={},
            summary={"status": "pending"},
            metrics={},
        )
        == "logged"
    )
    run.log.assert_not_called()
    assert run.summary == {"status": "pending"}
    run.finish.assert_called_once_with()


def test_finish_timeout_returns_failed_instead_of_waiting_for_uploads(service):
    sdk, run = service

    def finish():
        settings = sdk.init.call_args.kwargs["settings"]
        assert settings.get("finish_timeout") == 5
        if settings.get("finish_timeout_raises"):
            raise TimeoutError("synthetic upload timeout")

    run.finish.side_effect = finish
    assert (
        logger()(run_id="synthetic", stage="track-b", config={}, summary={}, metrics={})
        == "failed"
    )
    settings = sdk.init.call_args.kwargs["settings"]
    assert settings.get("finish_timeout") == 5
    assert settings.get("finish_timeout_raises") is True
    run.finish.assert_called_once_with()


@pytest.mark.parametrize("failure", ["init", "log", "summary", "finish"])
def test_remote_failures_return_failed_and_finalize(service, failure):
    sdk, run = service
    if failure == "summary":
        run.summary = Mock()
        run.summary.update.side_effect = RuntimeError("synthetic service failure")
    else:
        getattr(sdk if failure == "init" else run, failure).side_effect = RuntimeError(
            "synthetic service failure"
        )
    assert (
        logger()(
            run_id="synthetic",
            stage="evaluation",
            config={},
            summary={"status": "ready"},
            metrics={"rows": 3},
        )
        == "failed"
    )
    assert run.finish.call_count == (0 if failure == "init" else 1)


def test_missing_sdk_is_nonblocking(service, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert (
        logger()(run_id="synthetic", stage="track-b", config={}, summary={}, metrics={})
        == "failed"
    )


def test_unsupported_stage_does_not_start_remote_run(service):
    sdk, _ = service
    assert (
        logger()(run_id="synthetic", stage="unknown", config={}, summary={}, metrics={})
        == "failed"
    )
    sdk.init.assert_not_called()
