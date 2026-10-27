from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from typer.testing import CliRunner

from reformatters.google.weathernext3.forecast_virtual.template_config import STATISTICS
from reformatters.noaa.refs.forecast_hourly_virtual.template_config import (
    NoaaRefsForecastHourlyVirtualTemplateConfig,
)
from scripts.validation import (
    availability,
    compare_spatial,
    compare_timeseries,
    plots,
    value_timeseries,
)
from scripts.validation.utils import RunContext, choose_statistic, select_var_slice
from scripts.validation.value_timeseries import _point_arrays

WN3_DATASET_IDS = [
    f"google-weathernext3-forecast-{horizon}-{resolution}-degree-virtual"
    for horizon in ("48-hour", "15-day")
    for resolution in ("0-05", "0-1")
]


@pytest.fixture
def wn3_context(tmp_path: Path) -> RunContext:
    init = pd.date_range("2026-10-01", periods=2, freq="6h")
    lead = pd.to_timedelta([1, 2, 3], unit="h")
    values = np.broadcast_to(
        np.arange(1, len(STATISTICS) + 1)[None, :, None, None, None],
        (2, len(STATISTICS), 3, 2, 2),
    ).astype(float)
    ds = xr.Dataset(
        {
            "temperature_2m": (
                ("init_time", "statistic", "lead_time", "latitude", "longitude"),
                values,
                {"units": "K", "long_name": "2 metre temperature"},
            ),
        },
        coords={
            "init_time": init,
            "statistic": np.asarray(STATISTICS, dtype=object),
            "lead_time": lead,
            "latitude": [40.0, 30.0],
            "longitude": [-100.0, -90.0],
            "valid_time": (
                ("init_time", "lead_time"),
                init.values[:, None] + lead.values[None, :],
            ),
        },
        attrs={"dataset_id": WN3_DATASET_IDS[0], "name": "Validation"},
    )
    ds["deterministic"] = ds.temperature_2m.sel(statistic="mean", drop=True) * 10
    reference = (
        ds.sel(statistic="mean", drop=True)
        .isel(init_time=0, drop=True)
        .rename({"lead_time": "time"})
        .assign_coords(time=init[0] + lead)
        .drop_vars("valid_time")
    )
    reference.attrs["name"] = "Reference"
    return RunContext(
        output_dir=tmp_path,
        validation_url="s3://test/wn3/v1.icechunk",
        reference_url="s3://test/reference/v1.icechunk",
        validation_ds=ds,
        reference_ds=reference,
        started_at=pd.Timestamp.now(tz="UTC"),
        point1_sel={"latitude": 0, "longitude": 0},
        point2_sel={"latitude": 1, "longitude": 1},
        point1_lat=40.0,
        point1_lon=-100.0,
        point2_lat=30.0,
        point2_lon=-90.0,
        ensemble_member=None,
        variables=["temperature_2m"],
        init_time="2026-10-01",
        lead_time="2",
    )


@pytest.mark.parametrize("dataset_id", WN3_DATASET_IDS)
@pytest.mark.parametrize("dtype", [object, str])
def test_wn3_default_statistic(
    wn3_context: RunContext, dataset_id: str, dtype: type
) -> None:
    ds = wn3_context.validation_ds.assign_coords(
        statistic=np.asarray(STATISTICS, dtype=dtype)
    )
    ds.attrs["dataset_id"] = dataset_id
    assert choose_statistic(ds, "temperature_2m") == {"statistic": "mean"}


@pytest.mark.parametrize("label", STATISTICS)
def test_wn3_explicit_statistic(wn3_context: RunContext, label: str) -> None:
    ctx = wn3_context
    ctx.statistic_override = label
    stats = ctx.stats_for("temperature_2m")
    selection = select_var_slice(ctx, "temperature_2m", stats)
    assert selection == {"statistic": label}
    assert stats.label_value == label
    assert (
        ctx.validation_ds.temperature_2m.sel(selection) == STATISTICS.index(label) + 1
    ).all()


@pytest.mark.parametrize("override", [None, "mean"])
def test_wn3_all_nan_mean_stays_selected(
    wn3_context: RunContext, override: str | None
) -> None:
    ds = wn3_context.validation_ds
    ds["temperature_2m"].loc[{"statistic": "mean"}] = np.nan
    selection = choose_statistic(ds, "temperature_2m", override)
    assert selection == {"statistic": "mean"}
    assert ds.temperature_2m.sel(selection).isnull().all()


@pytest.mark.parametrize(
    ("labels", "override", "error"),
    [
        (["p10", "p25"], None, "missing"),
        (["mean", "p10"], "p90", "missing"),
        (["mean", "unknown"], None, "unknown statistic labels"),
        (["mean", "p10"], "standard_deviation", "not declared"),
    ],
)
def test_wn3_invalid_statistic(
    wn3_context: RunContext, labels: list[str], override: str | None, error: str
) -> None:
    ds = wn3_context.validation_ds.sel(statistic=list(STATISTICS[: len(labels)]))
    ds = ds.assign_coords(statistic=labels)
    with pytest.raises(ValueError, match=error):
        choose_statistic(ds, "temperature_2m", override)


def test_wn3_undeclared_variable_and_dataset_fail(wn3_context: RunContext) -> None:
    ds = wn3_context.validation_ds
    ds["undeclared"] = ds.temperature_2m
    with pytest.raises(ValueError, match="no registered source declaration"):
        choose_statistic(ds, "undeclared")
    ds.attrs["dataset_id"] = "unregistered"
    with pytest.raises(ValueError, match="no registered source declaration"):
        choose_statistic(ds, "temperature_2m")


def test_wn3_numeric_statistic_labels_fail(wn3_context: RunContext) -> None:
    ctx = wn3_context
    ctx.validation_ds = ctx.validation_ds.assign_coords(
        statistic=np.arange(len(STATISTICS))
    )
    with pytest.raises(ValueError, match="unknown statistic labels"):
        select_var_slice(ctx, "temperature_2m", ctx.stats_for("temperature_2m"))


def test_wn3_unknown_categorical_axis_fails(wn3_context: RunContext) -> None:
    ds = wn3_context.validation_ds.rename(statistic="category")
    with pytest.raises(ValueError, match="unsupported categorical axes"):
        choose_statistic(ds, "temperature_2m")


@pytest.mark.parametrize("label", STATISTICS)
def test_wn3_virtual_value_sample_selects_one_statistic(
    wn3_context: RunContext, label: str
) -> None:
    ctx = wn3_context
    ctx.is_virtual = True
    ctx.statistic_override = label
    stats = ctx.stats_for("temperature_2m")
    for point in _point_arrays(ctx, "temperature_2m", stats):
        assert point.dims == ("init_time",)
        assert point.statistic.item() == label
        assert (point == STATISTICS.index(label) + 1).all()


@pytest.mark.parametrize("label", [None, "p10"])
def test_value_availability_scans_only_selected_statistic(
    wn3_context: RunContext, label: str | None
) -> None:
    ctx = wn3_context
    ctx.statistic_override = label
    ctx.validation_ds["temperature_2m"].loc[{"statistic": "mean"}] = np.nan
    availability.run_value_availability(ctx)
    expected = label or "mean"
    stats = ctx.stats["temperature_2m"]
    assert stats.label_value == expected
    assert stats.null_count_p1 == (0 if label else stats.total_count_p1)
    np.testing.assert_array_equal(
        ctx.availability["temperature_2m"].fraction, [1 if label else 0] * 2
    )


@pytest.mark.parametrize(
    ("var", "override", "valid"),
    [
        ("wind_u_10m", "mean", True),
        ("wind_u_10m", "standard_deviation", True),
        ("temperature_2m", "mean", True),
        ("temperature_2m", "standard_deviation", False),
        ("derived_radar_reflectivity_1000m", "standard_deviation", True),
        ("derived_radar_reflectivity_1000m", "mean", False),
        ("wind_u_10m", "p50", False),
    ],
)
def test_refs_explicit_statistic(
    statistic_context: RunContext, var: str, override: str, valid: bool
) -> None:
    ds = statistic_context.validation_ds
    if valid:
        assert choose_statistic(ds, var, override) == {"statistic": override}
    else:
        with pytest.raises(ValueError, match="not declared"):
            choose_statistic(ds, var, override)


def test_statistic_override_ignores_standalone_products(
    statistic_context: RunContext,
) -> None:
    ctx = statistic_context
    ctx.statistic_override = "p90"
    for var in ctx.variables:
        if "statistic" not in ctx.validation_ds[var].dims:
            stats = ctx.stats_for(var)
            assert select_var_slice(ctx, var, stats) == {}
            assert stats.label_value is None


def test_refs_requested_statistic_missing_from_coordinates(
    statistic_context: RunContext,
) -> None:
    ds = statistic_context.validation_ds.sel(statistic=["mean"])
    with pytest.raises(ValueError, match="missing"):
        choose_statistic(ds, "wind_u_10m", "standard_deviation")


def test_refs_unsupported_statistic_source_family_fails(
    statistic_context: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    var = next(
        v
        for v in NoaaRefsForecastHourlyVirtualTemplateConfig().data_vars
        if v.name == "wind_u_10m"
    )
    var = var.model_copy(
        update={
            "internal_attrs": var.internal_attrs.model_copy(
                update={"source_families": ("prob",)}
            )
        }
    )
    monkeypatch.setattr(
        "scripts.validation.scan_common.find_registered_dataset",
        Mock(return_value=Mock(template_config=Mock(data_vars=[var]))),
    )
    with pytest.raises(ValueError, match="unsupported statistic source families"):
        choose_statistic(statistic_context.validation_ds, var.path)


@pytest.mark.parametrize("label", [None, *STATISTICS])
def test_run_all_statistic_cli_smoke(
    wn3_context: RunContext, monkeypatch: pytest.MonkeyPatch, label: str | None
) -> None:
    ctx = wn3_context
    monkeypatch.setattr(
        plots,
        "load_zarr_dataset",
        Mock(side_effect=[ctx.validation_ds, ctx.reference_ds]),
    )
    monkeypatch.setattr(plots, "is_virtual_store", Mock(return_value=False))
    scan = Mock(wraps=plots.run_value_availability)
    monkeypatch.setattr(plots, "run_value_availability", scan)
    trace = Mock(wraps=availability._plot_var_availability)
    monkeypatch.setattr(availability, "_plot_var_availability", trace)
    args = [
        "run-all",
        ctx.validation_url,
        "--reference-url",
        str(ctx.reference_url),
        "--init-time",
        str(ctx.init_time),
        "--lead-time",
        str(ctx.lead_time),
        "--output-dir",
        str(ctx.output_dir),
    ]
    if label is not None:
        args.extend(["--statistic", label])
    result = CliRunner().invoke(plots.app, args)
    assert result.exit_code == 0, result.output or repr(result.exception)
    actual_ctx = scan.call_args.args[0]
    stats = actual_ctx.stats["temperature_2m"]
    expected = label or "mean"
    value = STATISTICS.index(expected) + 1
    assert actual_ctx.statistic_override == label
    assert stats.label_value == expected
    assert stats.value_mean_p1 == value
    assert stats.val_spatial_mean == value
    assert stats.val_temporal_mean_p1 == value
    assert stats.ref_available_spatial == (expected == "mean")
    assert stats.ref_available_temporal == (expected == "mean")
    deterministic = actual_ctx.stats["deterministic"]
    assert deterministic.label_value is None
    assert deterministic.value_mean_p1 == 10
    assert deterministic.ref_available_spatial
    assert deterministic.ref_available_temporal
    assert (
        actual_ctx.loaded_point_data["temperature_2m"][0].statistic.item() == expected
    )
    summary = (ctx.output_dir / "validation_summary.md").read_text()
    assert f"### `temperature_2m` [statistic={expected}]" in summary
    assert f"| sampled statistic | statistic={expected} |" in summary
    assert any(
        f"temperature_2m [statistic={expected}]" == call.args[2]
        for call in trace.call_args_list
    )
    assert (
        f"| Statistic | override {label} |"
        if label is not None
        else "| Statistic | mean when declared, otherwise standard_deviation |"
    ) in summary
    for filename in (stats.value_ts_plot, stats.spatial_plot, stats.temporal_plot):
        assert filename is not None
        assert (ctx.output_dir / filename).stat().st_size > 0


@pytest.mark.parametrize(
    ("command", "label"),
    [
        ("availability", "p10"),
        ("value-timeseries", "p25"),
        ("compare-timeseries", "p75"),
        ("compare-spatial", "p90"),
    ],
)
def test_standalone_statistic_cli_smoke(
    wn3_context: RunContext, monkeypatch: pytest.MonkeyPatch, command: str, label: str
) -> None:
    ctx = wn3_context
    module, function_name = {
        "availability": (availability, "run_value_availability"),
        "value-timeseries": (value_timeseries, "run_value_timeseries"),
        "compare-timeseries": (compare_timeseries, "run_compare_timeseries"),
        "compare-spatial": (compare_spatial, "run_compare_spatial"),
    }[command]
    monkeypatch.setattr(
        module,
        "load_zarr_dataset",
        Mock(side_effect=[ctx.validation_ds, ctx.reference_ds]),
    )
    monkeypatch.setattr(module, "is_virtual_store", Mock(return_value=False))
    run = Mock(wraps=getattr(module, function_name))
    monkeypatch.setattr(module, function_name, run)
    args = [
        command,
        ctx.validation_url,
        "--statistic",
        label,
        "--output-dir",
        str(ctx.output_dir),
    ]
    if command.startswith("compare-"):
        args.extend(
            [
                "--reference-url",
                str(ctx.reference_url),
                "--init-time",
                str(ctx.init_time),
            ]
        )
    if command == "compare-spatial":
        args.extend(["--lead-time", str(ctx.lead_time)])
    result = CliRunner().invoke(plots.app, args)
    assert result.exit_code == 0, result.output or repr(result.exception)
    actual_ctx = run.call_args.args[0]
    stats = actual_ctx.stats["temperature_2m"]
    assert actual_ctx.statistic_override == label
    assert stats.label_value == label
    assert not stats.ref_available_spatial
    assert not stats.ref_available_temporal
    if command == "availability":
        assert (
            actual_ctx.loaded_point_data["temperature_2m"][0].statistic.item() == label
        )
    else:
        assert STATISTICS.index(label) + 1 in (
            stats.value_mean_p1,
            stats.val_temporal_mean_p1,
            stats.val_spatial_mean,
        )
