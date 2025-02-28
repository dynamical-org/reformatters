import sentry_sdk
import typer
from sentry_sdk.integrations.typer import TyperIntegration

import reformatters.noaa.gefs.forecast_35_day.cli as noaa_gefs_forecast_35_day
from reformatters.common import deploy
from reformatters.common.config import Config

if Config.is_sentry_enabled:
    sentry_sdk.init(
        dsn=Config.sentry_dsn,
        environment=Config.env.value,
        integrations=[TyperIntegration()],
    )


app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(noaa_gefs_forecast_35_day.app, name="noaa-gefs-forecast-35-day")


@app.command()
def deploy_operational_updates(
    docker_image: str | None = None,
) -> None:
    deploy.deploy_operational_updates(docker_image=docker_image)


if __name__ == "__main__":
    app()
