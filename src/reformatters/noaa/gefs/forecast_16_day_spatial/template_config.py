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
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_coordinate_configs,
    get_shared_data_var_configs,
)
from reformatters.noaa.gefs.gefs_config_models import GEFSDataVar

EXPECTED_FORECAST_LENGTH = pd.Timedelta(hours=384)


class GefsForecast16DaySpatialTemplateConfig(TemplateConfig[GEFSDataVar]):
    """Template configuration for the GEFS 16-day spatial (virtual) forecast dataset.

    Virtual icechunk dataset: chunks are references to GRIB messages in NOAA's
    archive, decoded at read time, so the grid is the native 0.5 degree a/b file
    grid (latitude 90 to -90, longitude 0 to 359.5) with one chunk per message.
    See docs/virtual_datasets.md.
    """

    dims: tuple[Dim, ...] = (
        "init_time",
        "ensemble_member",
        "lead_time",
        "latitude",
        "longitude",
    )
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2020-10-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-gefs-forecast-16-day-spatial",
            dataset_version="0.1.0",
            name="NOAA GEFS forecast, 16 day, spatial",
            description="Weather forecasts from the Global Ensemble Forecast System (GEFS) operated by NOAA NWS NCEP, optimized for spatial (map) access patterns.",
            attribution="NOAA NWS NCEP GEFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.5 degrees (~40km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution="Forecasts initialized every 6 hours",
            forecast_domain="Forecast lead time 0-384 hours (0-16 days) ahead",
            forecast_resolution="Forecast step 0-240 hours: 3 hourly, 246-384 hours: 6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "ensemble_member": np.arange(31),
            "lead_time": pd.timedelta_range("0h", "240h", freq="3h").union(
                pd.timedelta_range("246h", "384h", freq="6h")
            ),
            # Native GEFS 0.5 degree grid: latitude descends, longitude is 0-360.
            "latitude": np.flip(np.arange(-90, 90.5, 0.5)),
            "longitude": np.arange(0, 360, 0.5),
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
        spatial_ref = next(
            c for c in get_shared_coordinate_configs() if c.name == "spatial_ref"
        )

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
                        min=dim_coords["latitude"].min(),
                        max=dim_coords["latitude"].max(),
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
                        min=dim_coords["longitude"].min(),
                        max=dim_coords["longitude"].max(),
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
                        min=self.append_dim_start.isoformat(), max="Present + 16 days"
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
    def data_vars(self) -> Sequence[GEFSDataVar]:
        dim_coords = self.dimension_coordinates()
        # One chunk per GRIB message: see "Encoding rules" in docs/virtual_datasets.md.
        message_chunks = (
            1,  # init_time
            1,  # ensemble_member
            1,  # lead_time
            len(dim_coords["latitude"]),
            len(dim_coords["longitude"]),
        )
        return [
            replace(var, encoding=_virtual_encoding(var, message_chunks))
            for var in get_shared_data_var_configs(message_chunks, message_chunks)
        ]


def _virtual_encoding(var: GEFSDataVar, message_chunks: tuple[int, ...]) -> Encoding:
    return Encoding(
        # GribberishCodec decodes to float64 natively; declaring float64 avoids any cast.
        dtype="float64",
        fill_value=np.nan,
        chunks=message_chunks,
        shards=None,
        # Empty, not None: exclude_none serialization would drop a None and zarr
        # would then apply its default compressor on top of the GRIB bytes.
        compressors=(),
        filters=(),
        serializer=GribberishCodec(var=var.internal_attrs.grib_element).to_dict(),
    )
