import typer

from scripts.validation.compare_vars import compare_vars
from scripts.validation.nulls_over_time import plot_nulls

app = typer.Typer(help="Dataset validation plotting tools")

app.command("compare-vars", help="Compare two datasets")(compare_vars)
app.command("plot-nulls", help="Analyze null values")(plot_nulls)

if __name__ == "__main__":
    app()
