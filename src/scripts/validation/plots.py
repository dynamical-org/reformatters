from pathlib import Path

import pandas as pd
import typer

from reformatters.common.logging import get_logger
from scripts.validation.compare_spatial import (
    compare_spatial,
    run_compare_spatial,
)
from scripts.validation.compare_timeseries import (
    compare_timeseries,
    run_compare_timeseries,
)
from scripts.validation.render import render_report_command
from scripts.validation.report_nulls import report_nulls, run_report_nulls
from scripts.validation.summary import write_summary_md
from scripts.validation.upload import upload_command
from scripts.validation.utils import (
    RunContext,
    create_run_output_dir,
    end_date_option,
    get_two_random_points,
    is_forecast_dataset,
    is_virtual_store,
    level_option,
    load_zarr_dataset,
    output_dir_option,
    reference_url_option,
    resolve_reference_url,
    scope_time_period,
    select_variables_for_plotting,
    start_date_option,
    variables_option,
)
from scripts.validation.value_timeseries import (
    run_value_timeseries,
    value_timeseries,
)

log = get_logger(__name__)

app = typer.Typer(
    help="Dataset validation plotting tools", pretty_exceptions_show_locals=False
)

app.command("compare-spatial", help="Spatial comparison, one PNG per variable")(
    compare_spatial
)
app.command("compare-timeseries", help="Timeseries comparison, one PNG per variable")(
    compare_timeseries
)
app.command("report-nulls", help="Null analysis, one PNG per variable")(report_nulls)
app.command(
    "value-timeseries",
    help="Full-period value time series (mean ± std), one PNG per variable",
)(value_timeseries)
app.command(
    "render-report",
    help="Render validation_summary.md to a static HTML report in the run directory.",
)(render_report_command)
app.command(
    "upload",
    help="Re-render and upload the run directory to the validation-reports R2 bucket.",
)(upload_command)


@app.command(
    "run-all",
    help="Run all validation plots into a single run directory, with a validation_summary.md index.",
)
def run_all(
    dataset_url: str,
    reference_url: str | None = reference_url_option,
    variables: list[str] | None = variables_option,
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    init_time: str | None = typer.Option(
        None,
        "--init-time",
        help="Forecast init_time for spatial plots (default: random)",
    ),
    lead_time: str | None = typer.Option(
        None,
        "--lead-time",
        help="Forecast lead_time for spatial plots (default: random)",
    ),
    time: str | None = typer.Option(
        None, "--time", help="Analysis time for spatial plots (default: random)"
    ),
    level: float | None = level_option,
    output_dir: Path | None = output_dir_option,
) -> None:
    """Produce nulls / spatial / temporal plots, one per variable, in one directory + validation_summary.md."""
    started_at = pd.Timestamp.now(tz="UTC")

    log.info(f"Loading validation dataset: {dataset_url}")
    validation_ds = load_zarr_dataset(dataset_url)
    if start_date or end_date:
        validation_ds = scope_time_period(validation_ds, start_date, end_date)

    reference_url = resolve_reference_url(reference_url)
    log.info(f"Loading reference dataset:  {reference_url}")
    reference_ds = load_zarr_dataset(reference_url)

    is_forecast = is_forecast_dataset(validation_ds)
    log.info(f"Validation dataset type: {'forecast' if is_forecast else 'analysis'}")

    is_virtual = is_virtual_store(dataset_url)
    if is_virtual:
        log.info(
            "Virtual store: availability is covered whole-archive by manifest_scan.py "
            "(value-based null scan skipped); value time series are sampled."
        )

    selected_vars = select_variables_for_plotting(validation_ds, variables)
    log.info(f"Variables ({len(selected_vars)}): {', '.join(selected_vars)}")

    point1_sel, point2_sel, (lat1, lon1), (lat2, lon2) = get_two_random_points(
        validation_ds
    )

    out = (
        Path(output_dir)
        if output_dir
        else create_run_output_dir(dataset_url, started_at)
    )
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {out}")

    # ctx.validation_ds keeps the full ensemble dim; run_compare_spatial and
    # run_compare_timeseries reduce it to a random member lazily so the null
    # analysis sees every member.
    ctx = RunContext(
        output_dir=out,
        validation_url=dataset_url,
        reference_url=reference_url,
        validation_ds=validation_ds,
        reference_ds=reference_ds,
        started_at=started_at,
        point1_sel=point1_sel,
        point2_sel=point2_sel,
        point1_lat=lat1,
        point1_lon=lon1,
        point2_lat=lat2,
        point2_lon=lon2,
        ensemble_member=None,
        variables=selected_vars,
        start_date=start_date,
        end_date=end_date,
        is_virtual=is_virtual,
        level_override=level,
    )

    run_report_nulls(ctx)
    run_value_timeseries(ctx)
    run_compare_timeseries(ctx)
    run_compare_spatial(ctx, init_time=init_time, lead_time=lead_time, time=time)

    summary_path = write_summary_md(ctx)
    log.info(f"Done. Summary: {summary_path}")
    typer.echo(str(summary_path))


if __name__ == "__main__":
    app()
