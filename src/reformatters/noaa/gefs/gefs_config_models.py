from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, assert_never
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from reformatters.common.config_models import DataVar, EnsembleStatistic
from reformatters.common.region_job import CoordinateValueOrRange, SourceFileCoord
from reformatters.common.types import Dim, Timedelta, Timestamp
from reformatters.noaa.models import NoaaInternalAttrs
from reformatters.noaa.noaa_utils import has_hour_0_values

# We pull data from 3 types of source files: `a`, `b` and `s`.
# Selected variables are available in `s` at higher resolution (0.25 vs 0.5 deg)
# but `s` stops after forecast lead time 240h at which point the variable is still in `a` or `b`.
# `s+b-b22` is the same as `s+b` when init time >= 2022-10-18T12 and `b` before.
type GEFSFileType = Literal["a", "b", "s+a", "s+b", "s+b-b22"]
GEFS_S_FILE_MAX = pd.Timedelta(hours=240)
GEFS_B22_TRANSITION_DATE = pd.Timestamp("2022-10-18T12:00")


class GEFSInternalAttrs(NoaaInternalAttrs):
    gefs_file_type: GEFSFileType


class GEFSDataVar(DataVar[GEFSInternalAttrs]):
    pass


# We pull data from three different periods of GEFS.
#
# 1. The current configuration archive, which is 0.25 degree data from 2020-09-23T12 the present.
# 2. The pre GEFS v12 archive, which is 1.0 degree data that we use from 2020-01-01T00 to 2020-09-23T06.
# 3. The GEFS v12 retrospective (reforecast) archive, which is 0.25 degree data from 2000-01-01T03 to 2019-12-31T21.
#
GEFS_CURRENT_ARCHIVE_START = pd.Timestamp("2020-09-23T12:00")
GEFS_REFORECAST_END = pd.Timestamp("2020-01-01T00:00")  # exclusive end point
GEFS_REFORECAST_START = pd.Timestamp("2000-01-01T00:00")

GEFS_REFORECAST_INIT_TIME_FREQUENCY = pd.Timedelta("24h")
GEFS_INIT_TIME_FREQUENCY: Final[pd.Timedelta] = pd.Timedelta("6h")

# Accumulations are reset every 6 hours in all periods of GEFS data
GEFS_ACCUMULATION_RESET_FREQUENCY: Final[pd.Timedelta] = pd.Timedelta("6h")
GEFS_ACCUMULATION_RESET_HOURS: Final[int] = int(
    GEFS_ACCUMULATION_RESET_FREQUENCY.total_seconds() / (60 * 60)
)
assert GEFS_ACCUMULATION_RESET_FREQUENCY == pd.Timedelta(
    hours=GEFS_ACCUMULATION_RESET_HOURS
)

# Short names are used in the file names of the GEFS v12 reforecast
GEFS_REFORECAST_LEVELS_SHORT = {
    "entire atmosphere": "eatm",
    "entire atmosphere (considered as a single layer)": "eatm",
    "cloud ceiling": "ceiling",
    "surface": "sfc",
    "mean sea level": "msl",
    "2 m above ground": "2m",
    "10 m above ground": "hgt",
    "100 m above ground": "hgt",
}
GEFS_REFORECAST_GRIB_ELEMENT_RENAME = {
    "PRMSL": "PRES",  # In the reforecast, PRMSL is PRES with level "mean sea level"
}

FILE_RESOLUTIONS = {
    "s": "0p25",
    "a": "0p50",
    "b": "0p50",
}


def is_v12(init_time: pd.Timestamp) -> bool:
    return init_time < GEFS_REFORECAST_END or GEFS_CURRENT_ARCHIVE_START <= init_time


def is_v12_index(times: pd.DatetimeIndex) -> np.ndarray[Any, np.dtype[np.bool]]:
    return (times < GEFS_REFORECAST_END) | (GEFS_CURRENT_ARCHIVE_START <= times)


def get_grib_element(var_info: GEFSDataVar, init_time: pd.Timestamp) -> str:
    grib_element = var_info.internal_attrs.grib_element
    if init_time < GEFS_REFORECAST_END:
        return GEFS_REFORECAST_GRIB_ELEMENT_RENAME.get(grib_element, grib_element)
    else:
        return grib_element


class GefsSourceFileCoord(SourceFileCoord):
    """Source file coordinate for GEFS forecast data."""

    init_time: Timestamp
    lead_time: Timedelta
    data_vars: Sequence[GEFSDataVar]

    primary_base_url: str = "noaa-gefs-pds.s3.amazonaws.com"
    fallback_base_url: str = "nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"

    @property
    def gefs_file_type(self) -> Literal["a", "b", "s", "reforecast"]:  # noqa: PLR0912 PLR0911
        # See `GEFSFileType` for details on the different types of files.

        gefs_file_types = {
            data_var.internal_attrs.gefs_file_type for data_var in self.data_vars
        }
        assert len(gefs_file_types) == 1, (
            f"All data vars must have the same gefs file type. Received {gefs_file_types}"
        )
        gefs_file_type = gefs_file_types.pop()

        if self.init_time >= GEFS_CURRENT_ARCHIVE_START:
            if gefs_file_type == "s+a":
                if self.lead_time <= GEFS_S_FILE_MAX:
                    return "s"
                else:
                    return "a"
            elif gefs_file_type == "s+b":
                if self.lead_time <= GEFS_S_FILE_MAX:
                    return "s"
                else:
                    return "b"
            elif gefs_file_type == "s+b-b22":
                if self.init_time >= GEFS_B22_TRANSITION_DATE:
                    if self.lead_time <= GEFS_S_FILE_MAX:
                        return "s"
                    else:
                        return "b"
                else:
                    return "b"
            else:
                return gefs_file_type

        elif self.init_time >= GEFS_REFORECAST_END:
            match gefs_file_type:
                case "s+a" | "a":
                    return "a"
                case "s+b" | "s+b-b22" | "b":
                    return "b"
                case _ as unreachable:
                    assert_never(unreachable)
        elif self.init_time >= GEFS_REFORECAST_START:
            return "reforecast"
        else:
            raise ValueError(f"Unexpected init time: {self.init_time}")

    def get_url(self) -> str:
        lead_time_hours = self.lead_time.total_seconds() / (60 * 60)
        if lead_time_hours != round(lead_time_hours):
            raise ValueError(
                f"Lead time {self.lead_time} must be a whole number of hours"
            )

        init_date_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")

        if isinstance(ensemble_member := getattr(self, "ensemble_member", None), int):
            # control (c) or perterbed (p) ensemble member
            prefix = "c" if ensemble_member == 0 else "p"
            ensemble_or_statistic_str = f"{prefix}{ensemble_member:02}"
        elif isinstance(statistic := getattr(self, "statistic", None), str):
            ensemble_or_statistic_str = statistic
        else:
            raise ValueError(
                f"coords must be ensemble or statistic coord, found {self}."
            )

        data_vars = self.data_vars

        # Accumulated and last N hour avg values don't exist in the 0-hour forecast.
        if lead_time_hours == 0:
            data_vars = [
                data_var for data_var in data_vars if has_hour_0_values(data_var)
            ]
            assert len(data_vars) > 0, "No data variables with hour 0 values"

        true_gefs_file_type = self.gefs_file_type

        if self.init_time >= GEFS_CURRENT_ARCHIVE_START:
            resolution_str = FILE_RESOLUTIONS[true_gefs_file_type]
            url = (
                f"https://{self.primary_base_url}/"
                f"gefs.{init_date_str}/{init_hour_str}/atmos/pgrb2{true_gefs_file_type}{resolution_str.strip('0')}/"
                f"ge{ensemble_or_statistic_str}.t{init_hour_str}z.pgrb2{true_gefs_file_type}.{resolution_str}.f{lead_time_hours:03.0f}"
            )
            return url
        elif self.init_time >= GEFS_REFORECAST_END:
            url = (
                f"https://{self.primary_base_url}/"
                f"gefs.{init_date_str}/{init_hour_str}/pgrb2{true_gefs_file_type}/"
                f"ge{ensemble_or_statistic_str}.t{init_hour_str}z.pgrb2{true_gefs_file_type}f{lead_time_hours:02.0f}"
            )
            return url
        else:
            assert len(data_vars) == 1, "Only one data variable per file in GEFS v12 reforecast"  # fmt: skip
            data_var = data_vars[0]
            days_str = (
                "Days:1-10"
                if self.lead_time <= pd.Timedelta(hours=240)
                else "Days:10-16"
            )
            grib_element = get_grib_element(data_var, self.init_time)
            level_str = GEFS_REFORECAST_LEVELS_SHORT[data_var.internal_attrs.grib_index_level]  # fmt: skip
            url = (
                "https://noaa-gefs-retrospective.s3.amazonaws.com/"
                f"GEFSv12/reforecast/{self.init_time.year}/{init_date_str}{init_hour_str}/"
                f"{ensemble_or_statistic_str}/{days_str}/"
                f"{grib_element.lower()}_{level_str}_"
                f"{init_date_str}{init_hour_str}_{ensemble_or_statistic_str}.grib2"
            )

            return url

    def get_fallback_url(self) -> str:
        url = self.get_url()
        url_parsed = urlparse(url)
        url_parsed = url_parsed._replace(netloc=self.fallback_base_url)
        return url_parsed.geturl()

    def get_index_url(self, fallback: bool = False) -> str:
        if fallback:
            return self.get_fallback_url() + ".idx"
        else:
            return self.get_url() + ".idx"


class GefsEnsembleSourceFileCoord(GefsSourceFileCoord):
    """Source file coordinate for GEFS forecast ensemble data."""

    ensemble_member: int

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


class GefsStatisticSourceFileCoord(GefsSourceFileCoord):
    """Source file coordinate for GEFS forecast statistic data."""

    statistic: EnsembleStatistic

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "statistic": self.statistic,
        }
