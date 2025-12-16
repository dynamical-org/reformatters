from collections.abc import Sequence
from datetime import timedelta
from typing import Literal

import typer

from reformatters.common import validation
from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.kubernetes import ArchiveGribFilesCronJob, CronJob

from .region_job import DwdIconEuForecastRegionJob, DwdIconEuForecastSourceFileCoord
from .template_config import DwdIconEuDataVar, DwdIconEuForecastTemplateConfig


class DwdIconEuForecastDataset(
    DynamicalDataset[DwdIconEuDataVar, DwdIconEuForecastSourceFileCoord]
):
    template_config: DwdIconEuForecastTemplateConfig = DwdIconEuForecastTemplateConfig()
    region_job_class: type[DwdIconEuForecastRegionJob] = DwdIconEuForecastRegionJob

    def archive_grib_files(
        self, nwp_run_to_archive: Literal["all", "00z", "06z", "12z", "18z"] = "all"
    ) -> None:
        pass

    def get_cli(
        self,
    ) -> typer.Typer:
        """Create a CLI app with dataset commands."""
        app = super().get_cli()
        app.command()(self.archive_grib_files)
        return app

    def validators(self) -> Sequence[validation.DataValidator]:
        """Return a sequence of DataValidators to run on this dataset."""
        # return (
        #     validation.check_analysis_current_data,
        #     validation.check_analysis_recent_nans,
        # )
        return ()  # TODO(Jack): Return appropriate `validation.check_*`

    def operational_kubernetes_resources(self, image_tag: str) -> Sequence[CronJob]:
        """Return the kubernetes cron job definitions to operationally update
        and validate this dataset."""
        suspend = True  # Defaults to False, remove when you're ready to run operational updates and validation
        ftp_to_obstore_cron_job = ArchiveGribFilesCronJob(
            name=f"{self.dataset_id}-archive-grib-files",
            # The files for each NWP init are ready 4 hours after the NWP init time. For example,
            # the 00Z NWP init is ready at 4am. So let's run 4 times per day, starting at 4am.
            schedule="0 4,10,16,22 * * *",
            pod_active_deadline=timedelta(
                minutes=120
            ),  # TODO(Jack): Check how long it *actually* takes!
            image=image_tag,
            dataset_id=self.dataset_id,
            cpu="14",
            memory="30G",
            shared_memory="12G",
            ephemeral_storage="30G",
            secret_names=self.store_factory.k8s_secret_names(),
            suspend=suspend,
        )

        # TODO(Jack): Implement ValidationCronJob, ReformatCronJob, and return these

        return [ftp_to_obstore_cron_job]
