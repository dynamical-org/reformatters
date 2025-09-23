import typer

from scripts.validation.compare_spatial import compare_spatial
from scripts.validation.compare_timeseries import compare_timeseries
from scripts.validation.report_nulls import report_nulls

app = typer.Typer(help="Dataset validation plotting tools")

app.command("compare-spatial", help="Compare two datasets")(compare_spatial)
app.command("compare-timeseries", help="Compare timeseries between datasets")(
    compare_timeseries
)
app.command("report-nulls", help="Analyze null values")(report_nulls)

if __name__ == "__main__":
    app()
