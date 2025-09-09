from collections.abc import Mapping
from pathlib import Path

from reformatters.common.config_models import EnsembleStatistic
from reformatters.common.region_job import CoordinateValueOrRange, SourceFileCoord
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar, GEFSFileType
from reformatters.noaa.gefs.read_data import (
    SourceFileCoords,
)
from reformatters.noaa.gefs.read_data import (
    download_file as _download_file,
)


class GefsForecast35DayEnsembleSourceFileCoord(SourceFileCoord):
    """Source file coordinate for GEFS 35-day forecast ensemble data."""

    init_time: Timestamp
    ensemble_member: int
    lead_time: Timedelta

    def get_url(self) -> str:
        """Return the URL for this source file.

        Note: This is a simplified implementation. The actual URL construction
        in the original system is complex and depends on the data variables and
        file type, which are handled in the download_file function.
        """
        # The URL construction is complex and handled by the existing download_file function
        # For now, we return a placeholder URL - the actual URL will be constructed
        # by the download_file function when called from the RegionJob
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        lead_hours = self.lead_time.total_seconds() / 3600
        prefix = "c" if self.ensemble_member == 0 else "p"
        ensemble_str = f"{prefix}{self.ensemble_member:02}"

        return (
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{init_date_str}/"
            f"{init_hour_str}/atmos/pgrb2s25/ge{ensemble_str}.t{init_hour_str}z."
            f"pgrb2s.25.f{lead_hours:03.0f}"
        )

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        """Use existing coordinate structure directly for forecast dataset."""
        return {
            "init_time": self.init_time,
            "ensemble_member": self.ensemble_member,
            "lead_time": self.lead_time,
        }


class GefsForecast35DayStatisticSourceFileCoord(SourceFileCoord):
    """Source file coordinate for GEFS 35-day forecast statistic data."""

    init_time: Timestamp
    statistic: EnsembleStatistic
    lead_time: Timedelta

    def get_url(self) -> str:
        """Return the URL for this source file.

        Note: This is a simplified implementation. The actual URL construction
        in the original system is complex and depends on the data variables and
        file type, which are handled in the download_file function.
        """
        # Similar to ensemble, URL construction is handled by download_file
        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        lead_hours = self.lead_time.total_seconds() / 3600

        return (
            f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{init_date_str}/"
            f"{init_hour_str}/atmos/pgrb2s25/ge{self.statistic}.t{init_hour_str}z."
            f"pgrb2s.25.f{lead_hours:03.0f}"
        )

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        """Use existing coordinate structure directly for forecast dataset.

        Note: Statistic variables in forecast dataset don't have a 'statistic' dimension.
        They are separate data variables that don't include the ensemble_member dimension.
        The statistic coordinate is used for file selection during download/read,
        but the output location only uses init_time and lead_time.
        """
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
        }


# Union type for forecast 35-day source file coordinates
type GefsForecast35DaySourceFileCoord = (
    GefsForecast35DayEnsembleSourceFileCoord | GefsForecast35DayStatisticSourceFileCoord
)


def download_source_file(
    coord: GefsForecast35DaySourceFileCoord,
    gefs_file_type: GEFSFileType,
    gefs_idx_data_vars: list[GEFSDataVar],
) -> tuple[GefsForecast35DaySourceFileCoord, Path | None]:
    """Download a source file using the existing download_file function.

    This wraps the existing download_file function to work with our new
    SourceFileCoord classes while preserving all the complex logic for
    URL construction, file type handling, and error handling.
    """
    # Convert our SourceFileCoord back to the legacy TypedDict format
    # that the existing download_file function expects
    if isinstance(coord, GefsForecast35DayEnsembleSourceFileCoord):
        legacy_coords: SourceFileCoords = {
            "init_time": coord.init_time,
            "ensemble_member": coord.ensemble_member,
            "lead_time": coord.lead_time,
        }
    elif isinstance(coord, GefsForecast35DayStatisticSourceFileCoord):
        legacy_coords = {
            "init_time": coord.init_time,
            "statistic": coord.statistic,
            "lead_time": coord.lead_time,
        }
    else:
        raise TypeError(f"Unexpected coordinate type: {type(coord)}")

    # Call the existing download function
    returned_coords, path = _download_file(
        legacy_coords, gefs_file_type, gefs_idx_data_vars
    )

    # The returned coords should be identical to what we passed in
    assert returned_coords == legacy_coords

    return coord, path
