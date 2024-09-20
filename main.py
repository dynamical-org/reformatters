import typer

import noaa.gefs.forecast.cli as noaa_gefs_forecast

app = typer.Typer()
app.add_typer(noaa_gefs_forecast.app, name="noaa-gefs-forecast")

if __name__ == "__main__":
    app()
