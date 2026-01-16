import typer

from scripts.validation.compare_spatial import compare_spatial
from scripts.validation.compare_timeseries import GEFS_ANALYSIS_URL, compare_timeseries
from scripts.validation.report_nulls import report_nulls
from scripts.validation.utils import (
    end_date_option,
    start_date_option,
    variables_option,
)

app = typer.Typer(
    help="Dataset validation plotting tools", pretty_exceptions_show_locals=False
)

app.command("compare-spatial", help="Compare two datasets")(compare_spatial)
app.command("compare-timeseries", help="Compare timeseries between datasets")(
    compare_timeseries
)
app.command("report-nulls", help="Analyze null values")(report_nulls)


@app.command("run-all", help="Run all validation plots")
def run_all(
    dataset_url: str,
    reference_url: str = typer.Option(
        GEFS_ANALYSIS_URL, "--reference-url", help="Reference dataset URL"
    ),
    variables: list[str] | None = variables_option,
    show_plot: bool = typer.Option(False, "--show-plot", help="Display plots"),
    start_date: str | None = start_date_option,
    end_date: str | None = end_date_option,
    init_time: str | None = typer.Option(
        None, "--init-time", help="Forecast init_time (for spatial plots)"
    ),
    lead_time: str | None = typer.Option(
        None, "--lead-time", help="Forecast lead_time (for spatial plots)"
    ),
    time: str | None = typer.Option(
        None, "--time", help="Analysis time (for spatial plots)"
    ),
) -> None:
    """Run all three validation plots: report-nulls, compare-timeseries, and compare-spatial."""
    typer.echo("Running report-nulls...")
    report_nulls(
        dataset_url=dataset_url,
        variables=variables,
        show_plot=show_plot,
        start_date=start_date,
        end_date=end_date,
    )

    typer.echo("Running compare-timeseries...")
    compare_timeseries(
        validation_url=dataset_url,
        reference_url=reference_url,
        variables=variables,
        show_plot=show_plot,
        start_date=start_date,
        end_date=end_date,
    )

    typer.echo("Running compare-spatial...")
    compare_spatial(
        validation_url=dataset_url,
        reference_url=reference_url,
        variables=variables,
        show_plot=show_plot,
        init_time=init_time,
        lead_time=lead_time,
        time=time,
        start_date=start_date,
        end_date=end_date,
    )

    typer.echo("All validation plots completed!")


if __name__ == "__main__":
    app()
