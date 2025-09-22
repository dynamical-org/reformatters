import typer

# Common constants
OUTPUT_DIR = "data/output"

# Common typer options
variables_option = typer.Option(
    None,
    "--variable",
    "-v",
    help="Variable to plot (can be used multiple times). "
    "If not provided, will plot all common variables.",
)
