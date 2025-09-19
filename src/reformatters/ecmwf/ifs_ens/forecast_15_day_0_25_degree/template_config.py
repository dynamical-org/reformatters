from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import AppendDim, Dim, Timedelta, Timestamp
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)

"""
xr.open_dataset("data/2025-03-01T00:00:00.000000000.grib2", engine="cfgrib"), filter_by_keys={"dataType": "pf"})

=== GRIB2 File Information ===
File size: ~5.9 GB
Grid: 721 x 1440 (lat x lon)
Resolution: ~0.25°
Ensemble members: 50
Pressure levels: 13 levels
Time: 2025-03-01T00:00:00.000000000
Forecast step: 0 nanoseconds hours

=== Available Variables ===
asn        - Snow albedo [(0 - 1)]
d          - Divergence [s**-1]
ewss       - Time-integrated eastward turbulent surface stress [N m**-2 s]
gh         - Geopotential height [gpm]
lsm        - Land-sea mask [(0 - 1)]
msl        - Mean sea level pressure [Pa]
mucape     - Most-unstable CAPE [J kg**-1]
nsss       - Time-integrated northward turbulent surface stress [N m**-2 s]
ptype      - Precipitation type [(Code table 4.201)]
q          - Specific humidity [kg kg**-1]
r          - Relative humidity [%]
ro         - Runoff [m]
sithick    - Sea ice thickness [m]
skt        - Skin temperature [K]
sot        - Soil temperature [K]
sp         - Surface pressure [Pa]
ssr        - Surface net short-wave (solar) radiation [J m**-2]
ssrd       - Surface short-wave (solar) radiation downwards [J m**-2]
str        - Surface net long-wave (thermal) radiation [J m**-2]
strd       - Surface long-wave (thermal) radiation downwards [J m**-2]
sve        - Eastward surface sea water velocity [m s**-1]
svn        - Northward surface sea water velocity [m s**-1]
t          - Temperature [K]
tcw        - Total column water [kg m**-2]
tcwv       - Total column vertically-integrated water vapour [kg m**-2]
tp         - Total precipitation [m]
tprate     - Total precipitation rate [kg m**-2 s**-1]
ttr        - Top net long-wave (thermal) radiation [J m**-2]
u          - U component of wind [m s**-1]
u100       - 100 metre U wind component [m s**-1]
v          - V component of wind [m s**-1]
v100       - 100 metre V wind component [m s**-1]
vo         - Vorticity (relative) [s**-1]
vsw        - Volumetric soil moisture [m**3 m**-3]
w          - Vertical velocity [Pa s**-1]
zos        - Sea surface height [m]
"""


class EcmwfIfsEnsInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.
    """

    # TODO skipping this for now! to come back and do

    # NOAA examples:
    # grib_element: str
    # grib_description: str
    # grib_index_level: str
    # index_position: int
    # include_lead_time_suffix: bool = False
    # # for step_type != "instant"
    # window_reset_frequency: Timedelta | None = None


class EcmwfIfsEnsDataVar(DataVar[EcmwfIfsEnsInternalAttrs]):
    pass


class EcmwfIfsEnsForecast15Day025DegreeTemplateConfig(
    TemplateConfig[EcmwfIfsEnsDataVar]
):
    dims: tuple[Dim, ...] = (
        "init_time",
        "lead_time",
        "ensemble_member",
        "latitude",
        "longitude",
    )
    append_dim: AppendDim = "init_time"
    # forecasts available from same s3 bucket since 2023-01-18, but only with 0.4deg resolution from dataset start through 2024-01-31.
    append_dim_start: Timestamp = pd.Timestamp("2024-02-01T00:00")
    # starting with just 0z forecasts for now. 12z also available with same forecast length; 6 & 18z also available with 144hr max lead time.
    append_dim_frequency: Timedelta = pd.Timedelta("24h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            dataset_version="0.1.0",
            name="ECMWF IFS Ensemble (ENS) Forecast, 15 day, 0.25 degree",
            description="Ensemble weather forecasts from the ECMWF Integrated Forecasting System (IFS) - 15 day forecasts, 0.25 degree resolution.",
            attribution="ECMWF IFS Ensemble Forecast data processed by dynamical.org from ECMWF Open Data.",  # TODO correct all this
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-360 hours (0-15 days) ahead",
            forecast_resolution="Forecast step 0-144 hours: 3 hourly, 145-360 hours: 6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """
        Returns a dictionary of dimension names to coordinates for the dataset.
        """
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": (
                pd.timedelta_range("0h", "144h", freq="3h").union(
                    pd.timedelta_range("145h", "360h", freq="6h")
                )
            ),
            "ensemble_member": np.arange(1, 51),
            "latitude": np.flip(np.arange(-90, 90.25, 0.25)),
            "longitude": np.arange(-180, 180, 0.25),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Return a dictionary of non-dimension coordinates for the dataset.
        Called whenever len(ds.append_dim) changes.
        """
        # Non-dimension coordinates are additional labels for data along
        # one or more dimensions. Use them to make it easier to use and
        # understand your dataset.
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "ingested_forecast_length": (
                (self.append_dim,),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "ns")),
            ),
            "expected_forecast_length": (
                (self.append_dim,),
                np.full(
                    ds[self.append_dim].size,
                    ds["lead_time"].max(),
                    dtype="timedelta64[ns]",
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,  # TODO what should this be? seems wrong but other template_configs use it?
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        return [
            Coordinate(
                name=self.append_dim,
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
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=dim_coords[self.append_dim].min().isoformat(), max="Present"
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
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="ensemble_member",
                encoding=Encoding(
                    dtype="int32",
                    fill_value=-1,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["number"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="realization",  # TODO what does this mean lol I stole it from gefs
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["number"].min()),
                        max=int(dim_coords["number"].max()),
                    ),
                ),
            ),
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
                    units="degrees_north",
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
                    units="degrees_east",
                    statistics_approximate=StatisticsApproximate(
                        min=float(dim_coords["longitude"].min()),
                        max=float(dim_coords["longitude"].max()),
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
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=self.append_dim_start.isoformat(),
                        max="Present + 15 days",
                    ),
                ),
            ),
            Coordinate(
                name="ingested_forecast_length",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            # TODO: add expected forecast length?
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),  # Scalar coordinate
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    # TODO: Verify this CRS matches ECMWF data - copied from NOAA example
                    crs_wkt='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                    inverse_flattening=0.0,
                    reference_ellipsoid_name="unknown",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="unknown",
                    horizontal_datum_name="unknown",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]',
                    comment="This coordinate reference system matches the source data which follows WMO conventions of assuming the earth is a perfect sphere with a radius of 6,371,229m. It is similar to EPSG:4326, but EPSG:4326 uses a more accurate representation of the earth's shape.",
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[EcmwfIfsEnsDataVar]:
        """Define metadata and encoding for each data variable."""
        # Data variable chunking and sharding
        #
        # Aim for one of these roughly equivalent quantities:
        # 1-2mb chunks compressed
        # 4-8mb uncompressed
        # 4-8 million float32 values
        # TODO check the math on these chunks & shards being reasonable. reference slack notes
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 61,  # Updated for ECMWF 61-step forecast # TODO should be 83??
            "ensemble_member": 50,  # All ensemble members
            "latitude": 144,  # ~721/5 for reasonable chunk size
            "longitude": 144,  # ~1440/10 for reasonable chunk size
        }
        # Aim for one of these roughly equivalent quantities:
        # 64-256MB shards compressed
        # 256-1024MB uncompressed
        # 256 million to 1 billion float32 values
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 61 * 2,  # TODO should be 83 * 2??
            "ensemble_member": 50,
            "latitude": 144 * 5,
            "longitude": 144 * 5,
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        # default_keep_mantissa_bits = 7
        # TODO what does this mean ^? (gefs also has keep_mantissa_bits_categorical?)

        return [
            EcmwfIfsEnsDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="K",  # From GRIB metadata - Kelvin not Celsius
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    # TODO: Determine correct GRIB element name and other metadata from ECMWF-specific format
                ),
            ),
            EcmwfIfsEnsDataVar(
                name="wind_u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component",
                    units="m s**-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    # TODO: Determine correct GRIB element name and other metadata from ECMWF-specific format
                ),
            ),
            EcmwfIfsEnsDataVar(
                name="wind_v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component",
                    units="m s**-1",
                    step_type="instant",
                    standard_name="northward_wind",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    # TODO: Determine correct GRIB element name and other metadata from ECMWF-specific format
                ),
            ),
            EcmwfIfsEnsDataVar(
                name="precipitation_total",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tp",
                    long_name="Total precipitation",
                    units="m",  # From GRIB metadata - meters not mm/s
                    step_type="accum",  # From GRIB metadata - accumulated not instantaneous
                    comment="Accumulated precipitation since forecast start time.",
                ),
                internal_attrs=EcmwfIfsEnsInternalAttrs(
                    # TODO: Determine if deaccumulation needed and correct GRIB processing parameters
                ),
            ),
        ]
