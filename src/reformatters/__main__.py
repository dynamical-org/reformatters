import sentry_sdk
import typer
from sentry_sdk.integrations.typer import TyperIntegration

import reformatters.noaa.gefs.analysis.cli as noaa_gefs_analysis
import reformatters.noaa.gefs.forecast_35_day.cli as noaa_gefs_forecast_35_day
from reformatters.common import deploy
from reformatters.common.config import Config

if Config.is_sentry_enabled:
    sentry_sdk.init(
        dsn=Config.sentry_dsn,
        environment=Config.env.value,
        project_root="src/",
        in_app_include=["reformatters"],
        default_integrations=True,
        integrations=[
            TyperIntegration(),
        ],
    )


app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(noaa_gefs_forecast_35_day.app, name=noaa_gefs_forecast_35_day.DATASET_ID)
app.add_typer(noaa_gefs_analysis.app, name=noaa_gefs_analysis.DATASET_ID)


@app.command()
def deploy_operational_updates(
    docker_image: str | None = None,
) -> None:
    deploy.deploy_operational_updates(docker_image=docker_image)


if __name__ == "__main__":
    app()
