from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from gribberish.zarr import GribberishCodec
from pydantic import computed_field

from reformatters.common.config_models import (
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.pydantic import replace
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsForecast15Day025DegreeTemplateConfig,
)

EXPECTED_FORECAST_LENGTH = pd.Timedelta(hours=360)

# Reuse the materialized dataset's variable and CRS definitions so metadata stays
# in sync; the virtual dataset overrides only encoding (one chunk per message) and
# the few attrs where raw GRIB values differ from the materialized pipeline output.
_MATERIALIZED = EcmwfIfsEnsForecast15Day025DegreeTemplateConfig()

# Accumulated radiation needs a bespoke "time-integrated" name/units in its raw
# (un-deaccumulated) form; left out of this throwaway -dev dataset.
_DROP_VARS = frozenset(
    {
        "downward_long_wave_radiation_flux_surface",
        "downward_short_wave_radiation_flux_surface",
    }
)

# Virtual chunks serve raw GRIB values, so attrs diverge from the materialized
# dataset where its pipeline transforms values: temperatures stay Kelvin (no
# K -> degree_Celsius), total cloud cover stays a 0-1 fraction (not scaled to
# percent), and precipitation is the raw window accumulation in metres rather
# than a deaccumulated rate, so it gets a distinct name like the GEFS spatial dataset.
_RAW_VALUE_NAME_OVERRIDES: dict[str, str] = {
    "precipitation_surface": "total_precipitation_surface",
}
_RAW_VALUE_ATTRS_OVERRIDES: dict[str, dict[str, Any]] = {
    "temperature_2m": {"units": "K"},
    "dew_point_temperature_2m": {"units": "K"},
    "temperature_850hpa": {"units": "K"},
    "temperature_925hpa": {"units": "K"},
    "total_cloud_cover_atmosphere": {"units": "1"},
    "precipitation_surface": {
        "short_name": "tp",
        "long_name": "Total precipitation",
        "standard_name": "lwe_thickness_of_precipitation_amount",
        "units": "m",
        "step_type": "accum",
        "comment": "Accumulated from the start of the forecast; raw ECMWF total precipitation depth in metres.",
    },
}


class EcmwfIfsEnsForecast15DaySpatialTemplateConfig(TemplateConfig[EcmwfDataVar]):
    """Template configuration for the ECMWF IFS ENS 15-day spatial (virtual) forecast dataset.

    Virtual icechunk dataset: chunks are references to GRIB messages in ECMWF's
    open-data archive, decoded at read time, so the grid is the native 0.25 degree
    grid (latitude 90 to -90, longitude 180 to 179.75 wrapping through 0) with one
    chunk per message. See docs/virtual_datasets.md.
    """

    dims: tuple[Dim, ...] = (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    append_dim: AppendDim = "init_time"
    # IFS Cycle 50r1 (2026-05-12 06z) merged the ENS control into oper-fc; starting
    # at the first 00z init fully in that regime keeps every init's control member
    # in oper-fc and perturbed members in enfo-ef, with no historical special cases.
    append_dim_start: Timestamp = pd.Timestamp("2026-05-13T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("24h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            # -dev suffix: this dataset is a throwaway operational test of the
            # virtual dataset machinery; its structure is not settled.
            dataset_id="ecmwf-ifs-ens-forecast-15-day-spatial-dev",
            dataset_version="0.1.0",
            name="Development: ECMWF IFS ENS forecast, 15 day, spatial",
            description="Ensemble weather forecasts from the ECMWF Integrated Forecasting System (IFS), optimized for spatial (map) access patterns.",
            attribution="ECMWF IFS ENS forecast data processed by dynamical.org from ECMWF Open Data.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="Forecast step 0-144 hours: 3 hourly, 144-360 hours: 6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "145h", freq="3h").union(
                pd.timedelta_range("150h", "361h", freq="6h")
            ),
            # single control member (0) + 50 perturbed members (1-50)
            "ensemble_member": np.arange(0, 51),
            "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
            # Native ECMWF open-data grid: longitude starts at 180 and wraps through
            # 0 (180..359.75, then 0..179.75). Virtual chunks serve the raw message,
            # so this order must match the GRIB grid exactly.
            "longitude": np.concatenate(
                [np.arange(180, 360, 0.25), np.arange(0, 180, 0.25)]
            ),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                ("init_time",),
                np.full(
                    ds["init_time"].size, EXPECTED_FORECAST_LENGTH.to_timedelta64()
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()
        spatial_ref = next(c for c in _MATERIALIZED.coords if c.name == "spatial_ref")

        return (
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["latitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Latitude",
                    standard_name="latitude",
                    units="degree_north",
                    axis="Y",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["latitude"].min()),
                        max=float(dim_coords["latitude"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="longitude",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["longitude"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Longitude",
                    standard_name="longitude",
                    units="degree_east",
                    axis="X",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["longitude"].min()),
                        max=float(dim_coords["longitude"].max()),
                    ),
                ),
            ),
            spatial_ref,
            Coordinate(
                name="init_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast initialization time",
                    standard_name="forecast_reference_time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="ensemble_member",
                encoding=Encoding(
                    dtype="int16",
                    fill_value=-1,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["ensemble_member"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Ensemble member",
                    standard_name="realization",
                    units="1",
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["ensemble_member"].min()),
                        max=int(dim_coords["ensemble_member"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="lead_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=len(dim_coords["lead_time"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Forecast lead time",
                    standard_name="forecast_period",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="valid_time",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    calendar="proleptic_gregorian",
                    units="seconds since 1970-01-01 00:00:00",
                    chunks=(
                        append_dim_coordinate_chunk_size,
                        len(dim_coords["lead_time"]),
                    ),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Valid time",
                    standard_name="time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(), max="Present"
                    ),
                ),
            ),
            Coordinate(
                name="expected_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Expected forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(EXPECTED_FORECAST_LENGTH),
                        max=str(EXPECTED_FORECAST_LENGTH),
                    ),
                ),
            ),
        )

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcmwfDataVar]:
        dim_coords = self.dimension_coordinates()
        # One chunk per GRIB message: see "Encoding rules" in docs/virtual_datasets.md.
        message_chunks = (
            1,  # init_time
            1,  # lead_time
            1,  # ensemble_member
            len(dim_coords["latitude"]),
            len(dim_coords["longitude"]),
        )
        return [
            replace(
                var,
                name=_RAW_VALUE_NAME_OVERRIDES.get(var.name, var.name),
                encoding=_virtual_encoding(var, message_chunks),
                attrs=(
                    replace(var.attrs, **_RAW_VALUE_ATTRS_OVERRIDES[var.name])
                    if var.name in _RAW_VALUE_ATTRS_OVERRIDES
                    else var.attrs
                ),
            )
            for var in _MATERIALIZED.data_vars
            if var.name not in _DROP_VARS
        ]


def _virtual_encoding(var: EcmwfDataVar, message_chunks: tuple[int, ...]) -> Encoding:
    return Encoding(
        # GribberishCodec decodes to float64 natively; declaring float64 avoids any cast.
        dtype="float64",
        fill_value=np.nan,
        chunks=message_chunks,
        shards=None,
        compressors=(),
        filters=(),
        serializer=GribberishCodec(var=var.internal_attrs.grib_element).to_dict(),
    )
