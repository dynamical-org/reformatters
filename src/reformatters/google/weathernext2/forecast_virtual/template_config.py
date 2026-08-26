from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import computed_field
from zarr.codecs import BloscCodec, BytesCodec, ScaleOffset, TransposeCodec

from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    Group,
    StatisticsApproximate,
)
from reformatters.common.template_config import SPATIAL_REF_COORDS, TemplateConfig
from reformatters.common.types import (
    AppendDim,
    CodecConfig,
    Dims,
    Timedelta,
    Timestamp,
)
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE

_GRID_NLAT = 721
_GRID_NLON = 1440

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

PER_INIT_STORE_DATE = pd.Timestamp("2025-01-01T00:00")
_SPATIAL_REF_WKT = 'GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371229,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]'


# ScaleOffset decodes on read as value / scale + offset. Temperatures are served in
# degree_Celsius and geopotential is divided by standard gravity to serve geopotential
# height in metres, both matching ecmwf-aifs-single-forecast-virtual. The source's
# total_precipitation_6hr is in metres of liquid water equivalent; x1000 serves kg m-2.
_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()
_GEOPOTENTIAL_TO_HEIGHT = ScaleOffset(offset=0.0, scale=9.80665).to_dict()
_METRES_TO_KG_M2 = ScaleOffset(offset=0.0, scale=0.001).to_dict()
_SOURCE_BLOSC = BloscCodec(
    typesize=4,
    cname="lz4",
    clevel=5,
    shuffle="shuffle",
).to_dict()
_PRESSURE_TRANSPOSE = TransposeCodec(order=(0, 1, 2, 5, 3, 4)).to_dict()

type SourceLayout = Literal["historical", "operational"]


class GoogleWeathernext2InternalAttrs(BaseInternalAttrs):
    """Variable specific attributes used internally to drive processing.
    Not written to the dataset."""

    # The variable's array name in the source zarr stores.
    source_name: str
    # The first init time the source publishes this variable in a referenceable layout.
    date_available: Timestamp | None = None


class GoogleWeathernext2DataVar(DataVar[GoogleWeathernext2InternalAttrs]):
    pass


class GoogleWeathernext2ForecastVirtualTemplateConfig(
    TemplateConfig[GoogleWeathernext2DataVar]
):
    """Shared schema for native-chunk WeatherNext 2 virtual products."""

    source_layout: SourceLayout
    dataset_id_value: ClassVar[str]
    dataset_name_value: ClassVar[str]
    time_domain_end: ClassVar[str]
    init_time_statistics_max: ClassVar[str] = "Present"
    valid_time_statistics_max: ClassVar[str] = "Present + 15 days"

    dims: Dims = {
        ROOT: (
            "init_time",
            "ensemble_member",
            "lead_time",
            "latitude",
            "longitude",
        ),
        "pressure_level": (
            "init_time",
            "ensemble_member",
            "lead_time",
            "latitude",
            "longitude",
            "pressure_level",
        ),
    }
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2022-01-01T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id=self.dataset_id_value,
            dataset_version="0.1.0",
            name=self.dataset_name_value,
            description="Weather forecasts from the 64-member Google DeepMind WeatherNext 2 ensemble model.",
            attribution=(
                "Google requires this attribution: © 2025 DeepMind Technologies "
                "Limited's machine learning models "
                "used to create the experimental data made available at "
                "https://developers.google.com/earth-engine/datasets/catalog/"
                "projects_gcp-public-data-weathernext_assets_weathernext_2_0_0 under "
                "CC BY 4.0 licence terms. This data is intended for experimental "
                "modelling only and is not intended, validated, or approved for real "
                "world use. Use of the third-party materials referred to in the "
                "Acknowledgements section may be governed by separate terms and "
                "conditions or license provisions. Your use of the third-party "
                "materials is subject to any such terms and you should check that you "
                "can comply with any applicable restrictions or terms and conditions "
                "before use."
            ),
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution="0.25 degrees (~20km)",
            time_domain=(
                f"Forecasts initialized {self.append_dim_start} UTC to "
                f"{self.time_domain_end}"
            ),
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 6-360 hours (0.25-15 days) ahead",
            forecast_resolution="6 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            # The source publishes no lead time 0.
            "lead_time": pd.timedelta_range("6h", "360h", freq="6h"),
            "ensemble_member": np.arange(64),
            "latitude": np.arange(-90, 90.25, 0.25),
            "longitude": np.arange(0, 360, 0.25),
            "pressure_level": np.array(PRESSURE_LEVELS, dtype=np.int64),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                (self.append_dim,),
                np.full(
                    ds[self.append_dim].size,
                    self.dimension_coordinates()["lead_time"].max(),
                    dtype="timedelta64[us]",
                ),
            ),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
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
                    long_name="Forecast initialization time",
                    standard_name="forecast_reference_time",
                    units="seconds since 1970-01-01 00:00:00",
                    statistics_approximate=StatisticsApproximate(
                        min=dim_coords[self.append_dim].min().isoformat(),
                        max=self.init_time_statistics_max,
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
                    dtype="float64",
                    fill_value=float("nan"),
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
            Coordinate(
                name="pressure_level",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["pressure_level"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Pressure level",
                    standard_name="air_pressure",
                    units="hPa",
                    axis="Z",
                    positive="down",
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["pressure_level"].min()),
                        max=int(dim_coords["pressure_level"].max()),
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
                        min=self.append_dim_start.isoformat(),
                        max=self.valid_time_statistics_max,
                    ),
                ),
            ),
            Coordinate(
                name="expected_forecast_length",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=float("nan"),
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Expected forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].max()),
                        max=str(dim_coords["lead_time"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="spatial_ref",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=0,
                    chunks=(),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    crs_wkt=_SPATIAL_REF_WKT,
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                    inverse_flattening=0.0,
                    reference_ellipsoid_name="unknown",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="unknown",
                    horizontal_datum_name="unknown",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref=_SPATIAL_REF_WKT,
                    comment="The source declares no coordinate reference system. WeatherNext 2 runs on the ERA5 0.25 degree latitude-longitude grid, which follows WMO conventions of assuming the earth is a perfect sphere with a radius of 6,371,229m. It is similar to EPSG:4326, but EPSG:4326 uses a more accurate representation of the earth's shape.",
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[GoogleWeathernext2DataVar]:
        return [
            *_root_data_vars(self.source_layout),
            *_pressure_data_vars(self.source_layout),
        ]


def _virtual_encoding(
    group: Group, filters: Sequence[CodecConfig], source_layout: SourceLayout
) -> Encoding:
    if source_layout == "historical":
        chunks = (
            (1, 4, 1, _GRID_NLAT, _GRID_NLON)
            if group is ROOT
            else (1, 4, 1, _GRID_NLAT, _GRID_NLON, len(PRESSURE_LEVELS))
        )
    else:
        chunks = (
            (1, 1, 1, _GRID_NLAT, _GRID_NLON)
            if group is ROOT
            else (1, 1, 1, _GRID_NLAT, _GRID_NLON, 1)
        )
    encoding_filters = list(filters)
    if source_layout == "historical" and group is not ROOT:
        encoding_filters.append(_PRESSURE_TRANSPOSE)
    return Encoding(
        dtype="float32",
        fill_value=np.nan,
        chunks=chunks,
        shards=None,
        serializer=BytesCodec(endian="little").to_dict(),
        compressors=[_SOURCE_BLOSC],
        filters=encoding_filters,
    )


def _var(
    name: str,
    *,
    source_name: str,
    group: Group,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None,
    step_type: str,
    comment: str | None,
    date_available: Timestamp | None,
    filters: Sequence[CodecConfig],
    source_layout: SourceLayout,
) -> GoogleWeathernext2DataVar:
    return GoogleWeathernext2DataVar(
        name=name,
        group=group,
        encoding=_virtual_encoding(group, filters, source_layout),
        attrs=DataVarAttrs(
            short_name=short_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type=step_type,  # ty: ignore[invalid-argument-type]
            comment=comment,
        ),
        internal_attrs=GoogleWeathernext2InternalAttrs(
            source_name=source_name,
            date_available=date_available,
            # Virtual chunks are never rewritten, so no rounding.
            keep_mantissa_bits="no-rounding",
        ),
    )


def _root_var(
    name: str,
    *,
    source_name: str,
    short_name: str,
    long_name: str,
    units: str,
    source_layout: SourceLayout,
    standard_name: str | None = None,
    step_type: str = "instant",
    comment: str | None = None,
    filters: Sequence[CodecConfig] = (),
) -> GoogleWeathernext2DataVar:
    return _var(
        name,
        source_name=source_name,
        group=ROOT,
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        step_type=step_type,
        comment=comment,
        date_available=None,
        filters=filters,
        source_layout=source_layout,
    )


def _pressure_var(
    name: str,
    *,
    source_name: str,
    short_name: str,
    long_name: str,
    units: str,
    source_layout: SourceLayout,
    standard_name: str | None = None,
    filters: Sequence[CodecConfig] = (),
) -> GoogleWeathernext2DataVar:
    return _var(
        name,
        source_name=source_name,
        group="pressure_level",
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        step_type="instant",
        comment=None,
        date_available=None,
        filters=filters,
        source_layout=source_layout,
    )


def _root_data_vars(source_layout: SourceLayout) -> list[GoogleWeathernext2DataVar]:
    return [
        _root_var(
            "temperature_2m",
            source_name="2m_temperature",
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            filters=[_KELVIN_TO_CELSIUS],
            source_layout=source_layout,
        ),
        _root_var(
            "pressure_reduced_to_mean_sea_level",
            source_name="mean_sea_level_pressure",
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
            source_layout=source_layout,
        ),
        _root_var(
            "wind_u_10m",
            source_name="10m_u_component_of_wind",
            short_name="10u",
            long_name="10 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
            source_layout=source_layout,
        ),
        _root_var(
            "wind_v_10m",
            source_name="10m_v_component_of_wind",
            short_name="10v",
            long_name="10 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
            source_layout=source_layout,
        ),
        _root_var(
            "wind_u_100m",
            source_name="100m_u_component_of_wind",
            short_name="100u",
            long_name="100 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
            source_layout=source_layout,
        ),
        _root_var(
            "wind_v_100m",
            source_name="100m_v_component_of_wind",
            short_name="100v",
            long_name="100 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
            source_layout=source_layout,
        ),
        _root_var(
            "sea_surface_temperature",
            source_name="sea_surface_temperature",
            short_name="sst",
            long_name="Sea surface temperature",
            units="degree_Celsius",
            standard_name="sea_surface_temperature",
            comment="NaN over land where sea surface temperature does not apply.",
            filters=[_KELVIN_TO_CELSIUS],
            source_layout=source_layout,
        ),
        _root_var(
            "total_precipitation_surface",
            source_name="total_precipitation_6hr",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
            step_type="accum",
            comment=(
                "Accumulated over a six-hour forecast interval. Small "
                "negative values are raw model artifacts; set values < 0 to zero."
            ),
            filters=[_METRES_TO_KG_M2],
            source_layout=source_layout,
        ),
    ]


def _pressure_data_vars(
    source_layout: SourceLayout,
) -> list[GoogleWeathernext2DataVar]:
    return [
        _pressure_var(
            "geopotential_height",
            source_name="geopotential",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            filters=[_GEOPOTENTIAL_TO_HEIGHT],
            source_layout=source_layout,
        ),
        _pressure_var(
            "temperature",
            source_name="temperature",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
            filters=[_KELVIN_TO_CELSIUS],
            source_layout=source_layout,
        ),
        _pressure_var(
            "wind_u",
            source_name="u_component_of_wind",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
            source_layout=source_layout,
        ),
        _pressure_var(
            "wind_v",
            source_name="v_component_of_wind",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
            source_layout=source_layout,
        ),
        _pressure_var(
            "vertical_velocity",
            source_name="vertical_velocity",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
            source_layout=source_layout,
        ),
        _pressure_var(
            "specific_humidity",
            source_name="specific_humidity",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
            source_layout=source_layout,
        ),
    ]
