from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from gribberish.zarr import GribberishCodec
from pydantic import computed_field
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    Group,
    StatisticsApproximate,
)
from reformatters.common.pydantic import replace
from reformatters.common.template_config import SPATIAL_REF_COORDS
from reformatters.common.types import AppendDim, Array1D, Dim, Timedelta, Timestamp
from reformatters.common.zarr import BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE
from reformatters.noaa.hrrr.hrrr_config_models import (
    NoaaHrrrDataVar,
    NoaaHrrrFileType,
    NoaaHrrrInternalAttrs,
)
from reformatters.noaa.hrrr.template_config import NoaaHrrrCommonTemplateConfig

EXPECTED_FORECAST_LENGTH = pd.Timedelta(hours=48)

# HRRR CONUS Lambert Conformal grid, from NoaaHrrrCommonTemplateConfig._spatial_info.
# Asserted against _spatial_info in data_vars so the two cannot drift.
_GRID_NY = 1059
_GRID_NX = 1799

# 39 isobaric levels (hPa), descending like the GRIB order. The wrfprs 1013.2 mb
# pseudo-level is excluded - it is not part of the dense 25 hPa column.
PRESSURE_LEVELS = list(range(1000, 25, -25))  # 1000, 975, ..., 50
# 50 native hybrid (sigma) model levels, 1 (top) .. 50 (near surface).
MODEL_LEVELS = list(range(1, 51))

# Air temperature and dew point are served in Celsius to match the materialized
# noaa-hrrr-forecast-48-hour: GribberishCodec decodes the raw Kelvin message, then
# this array->array filter subtracts 273.15 on read. See docs/virtual_datasets.md.
_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()
_CELSIUS_ELEMENTS = frozenset({"TMP", "DPT"})

# ScaleOffset decodes on read as value / scale + offset (see zarr.codecs.ScaleOffset).
# Drop-in scaling filters that match the materialized noaa-hrrr-forecast-48-hour, which
# multiplies the raw GRIB value by its scale_factor. WEASD snow water equivalent decodes
# as kg m-2 of water; 1 kg m-2 = 0.001 m lwe, so scale=1000 yields metres. SNOWC snow
# cover decodes as a percent; scale=100 yields a 0-1 fraction.
_WATER_KG_M2_TO_M_LWE = ScaleOffset(offset=0.0, scale=1000.0).to_dict()
_PERCENT_TO_FRACTION = ScaleOffset(offset=0.0, scale=100.0).to_dict()

type WindowKind = Literal["instant", "max", "min", "avg", "acc_run", "acc_1h"]

# Each windowed kind's (step_type, window_reset_frequency). acc_run is the
# accumulation since init (window 0->lead, grows with lead); acc_1h is the
# preceding-hour bucket. noaa_grib_index._lead_time_str renders the matching idx
# window string per lead from step_type + window_reset_frequency.
_WINDOW_ATTRS: dict[WindowKind, tuple[str, Timedelta | None]] = {
    "instant": ("instant", None),
    "max": ("max", pd.Timedelta("1h")),
    "min": ("min", pd.Timedelta("1h")),
    "avg": ("avg", pd.Timedelta("1h")),
    "acc_run": ("accum", pd.Timedelta.max),
    "acc_1h": ("accum", pd.Timedelta("1h")),
}


def _virtual_encoding(element: str, group: Group, filters: Sequence[Any]) -> Encoding:
    """One chunk per GRIB message: chunk 1 along init_time/lead_time/vertical, full
    y/x, no shards, no compressors. GribberishCodec decodes the raw message and any
    array->array filters (K->C, unit scaling) are chained on read."""
    if group is ROOT:
        chunks: tuple[int, ...] = (1, 1, _GRID_NY, _GRID_NX)
    else:
        chunks = (1, 1, _GRID_NY, _GRID_NX, 1)
    return Encoding(
        # GribberishCodec decodes to float64 natively; declaring float64 avoids a cast.
        dtype="float64",
        fill_value=np.nan,
        chunks=chunks,
        shards=None,
        compressors=(),
        filters=filters,
        # adjust_longitude_range is a no-op on HRRR's projected grid (verified by a
        # real-message decode); applied uniformly per the drop-in codec strategy.
        serializer=GribberishCodec(var=element, adjust_longitude_range=True).to_dict(),
    )


def _data_var(
    name: str,
    *,
    element: str,
    grib_index_level: str,
    file_type: NoaaHrrrFileType,
    group: Group,
    window: WindowKind,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None,
    comment: str | None,
    hour_0: bool | None,
    filters: Sequence[Any] | None = None,
) -> NoaaHrrrDataVar:
    step_type, window_reset_frequency = _WINDOW_ATTRS[window]
    # Default to the K->C filter for temperature/dew point; a var may override with an
    # explicit unit-scaling filter (e.g. WEASD kg m-2 -> m, SNOWC percent -> fraction).
    resolved_filters: Sequence[Any] = (
        filters
        if filters is not None
        else ([_KELVIN_TO_CELSIUS] if element in _CELSIUS_ELEMENTS else ())
    )
    return NoaaHrrrDataVar(
        name=name,
        group=group,
        encoding=_virtual_encoding(element, group, resolved_filters),
        attrs=DataVarAttrs(
            short_name=short_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type=step_type,  # ty: ignore[invalid-argument-type]
            comment=comment,
        ),
        internal_attrs=NoaaHrrrInternalAttrs(
            grib_element=element,
            # Group vars carry a "{level} ..." format string the region job fills per
            # level; root vars carry the literal idx level string.
            grib_index_level=grib_index_level,
            hrrr_file_type=file_type,
            window_reset_frequency=window_reset_frequency,
            hour_0_values_override=hour_0,
            # Virtual chunks are never rewritten, so no rounding and no rasterio band
            # description / index position (unused fields the base model requires).
            keep_mantissa_bits="no-rounding",
            grib_description="",
            index_position=0,
        ),
    )


def _root(
    name: str,
    *,
    element: str,
    level: str,
    window: WindowKind = "instant",
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    comment: str | None = None,
    hour_0: bool | None = None,
    filters: Sequence[Any] | None = None,
) -> NoaaHrrrDataVar:
    return _data_var(
        name,
        element=element,
        grib_index_level=level,
        file_type="sfc",
        group=ROOT,
        window=window,
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=comment,
        hour_0=hour_0,
        filters=filters,
    )


def _pressure(
    name: str,
    *,
    element: str,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
) -> NoaaHrrrDataVar:
    return _data_var(
        name,
        element=element,
        grib_index_level="{level} mb",
        file_type="prs",
        group="pressure_level",
        window="instant",
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=None,
        hour_0=None,
    )


def _model(
    name: str,
    *,
    element: str,
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
) -> NoaaHrrrDataVar:
    return _data_var(
        name,
        element=element,
        grib_index_level="{level} hybrid level",
        file_type="nat",
        group="model_level",
        window="instant",
        short_name=short_name,
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        comment=None,
        hour_0=None,
    )


class NoaaHrrrForecast48HourSpatialTemplateConfig(NoaaHrrrCommonTemplateConfig):
    """Virtual, spatially-chunked (map-optimized) HRRR 48-hour forecast.

    Chunks are references to GRIB messages in NOAA's HRRR archive decoded at read
    time, so the grid is the native Lambert Conformal y/x grid with one chunk per
    message. Mirrors the materialized noaa-hrrr-forecast-48-hour temporal structure
    (00/06/12/18 UTC inits to f48) but covers every wrfsfc/wrfprs/wrfnat variable
    plus pressure_level and model_level vertical groups. See docs/virtual_datasets.md.
    """

    dims: dict[Group, tuple[Dim, ...]] = {
        ROOT: ("init_time", "lead_time", "y", "x"),
        "pressure_level": ("init_time", "lead_time", "y", "x", "pressure_level"),
        "model_level": ("init_time", "lead_time", "y", "x", "model_level"),
    }
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2018-07-13T12:00")  # start of HRRR v3
    append_dim_frequency: Timedelta = pd.Timedelta("6h")  # only 00/06/12/18 reach f48

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="noaa-hrrr-forecast-48-hour-spatial",
            dataset_version="0.1.0",
            name="NOAA HRRR forecast, 48 hour, spatial",
            description="Weather forecasts from the High-Resolution Rapid Refresh (HRRR) model operated by NOAA NWS NCEP, optimized for spatial (map) access patterns.",
            attribution="NOAA NWS NCEP HRRR data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Continental United States",
            spatial_resolution="3 km",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution="Forecasts initialized every 6 hours",
            forecast_domain="Forecast lead time 0-48 hours ahead",
            forecast_resolution="Hourly",
        )

    def _y_x_coordinates(self) -> tuple[Array1D[np.float64], Array1D[np.float64]]:
        # gribberish decodes the HRRR grid south-first (row 0 = southernmost), the
        # opposite of GDAL's north-first order the materialized config uses. Order y
        # ascending so the virtual chunk data aligns with its coordinates (native grid;
        # see docs/virtual_datasets.md). x is unaffected.
        y_north_first, x_coords = super()._y_x_coordinates()
        return np.ascontiguousarray(y_north_first[::-1]), x_coords

    def _south_first_geotransform(self) -> str:
        # The GeoTransform's y origin / sign must match the south-first y ordering above.
        _, bounds, resolution, _ = self._spatial_info()
        left, bottom, _right, _top = bounds
        dx = resolution[0]
        return f"{left} {dx} 0.0 {bottom} 0.0 {dx}"

    def dimension_coordinates(self) -> dict[str, Any]:
        y_coords, x_coords = self._y_x_coordinates()
        return {
            "init_time": self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "48h", freq=pd.Timedelta("1h")),
            "y": y_coords,
            "x": x_coords,
            "pressure_level": np.array(PRESSURE_LEVELS, dtype=np.int64),
            "model_level": np.array(MODEL_LEVELS, dtype=np.int64),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        latitudes, longitudes = self._latitude_longitude_coordinates(
            ds["x"].values, ds["y"].values
        )
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                ("init_time",),
                np.full(
                    ds["init_time"].size, EXPECTED_FORECAST_LENGTH.to_timedelta64()
                ),
            ),
            "latitude": (("y", "x"), latitudes),
            "longitude": (("y", "x"), longitudes),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()

        # Reuse the common HRRR grid coords (x, y, latitude, longitude, spatial_ref) but
        # repoint the spatial_ref GeoTransform to the south-first y ordering.
        common_coords = [
            replace(
                c, attrs=replace(c.attrs, GeoTransform=self._south_first_geotransform())
            )
            if c.name == "spatial_ref"
            else c
            for c in super().coords
        ]

        return [
            *common_coords,
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
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["pressure_level"].min()),
                        max=int(dim_coords["pressure_level"].max()),
                    ),
                ),
            ),
            Coordinate(
                name="model_level",
                encoding=Encoding(
                    dtype="int64",
                    fill_value=-1,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(dim_coords["model_level"]),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Hybrid model level number",
                    units="1",
                    axis="Z",
                    statistics_approximate=StatisticsApproximate(
                        min=int(dim_coords["model_level"].min()),
                        max=int(dim_coords["model_level"].max()),
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
                        max="Present + 48 hours",
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
                        min=str(EXPECTED_FORECAST_LENGTH),
                        max=str(EXPECTED_FORECAST_LENGTH),
                    ),
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[NoaaHrrrDataVar]:
        assert self._spatial_info()[0] == (_GRID_NY, _GRID_NX), (
            "grid shape drifted from _GRID_NY/_GRID_NX used in the virtual encoding"
        )
        return [*_root_data_vars(), *_pressure_data_vars(), *_model_data_vars()]


def _root_data_vars() -> list[NoaaHrrrDataVar]:
    return [
        _root(
            "composite_reflectivity",
            element="REFC",
            level="entire atmosphere",
            short_name="refc",
            long_name="Maximum/Composite radar reflectivity",
            units="dBZ",
        ),
        _root(
            "echo_top",
            element="RETOP",
            level="cloud top",
            short_name="retop",
            long_name="Echo top",
            units="m",
        ),
        _root(
            "vertically_integrated_liquid_atmosphere",
            element="VIL",
            level="entire atmosphere",
            short_name="veril",
            long_name="Vertically-integrated liquid",
            units="kg m-2",
        ),
        _root(
            "visibility_surface",
            element="VIS",
            level="surface",
            short_name="vis",
            long_name="Visibility",
            units="m",
            standard_name="visibility_in_air",
        ),
        _root(
            "derived_radar_reflectivity_1000m",
            element="REFD",
            level="1000 m above ground",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
        ),
        _root(
            "derived_radar_reflectivity_4000m",
            element="REFD",
            level="4000 m above ground",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
        ),
        _root(
            "derived_radar_reflectivity_263k",
            element="REFD",
            level="263 K level",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
        ),
        _root(
            "wind_gust_surface",
            element="GUST",
            level="surface",
            short_name="gust",
            long_name="Wind speed (gust)",
            units="m s-1",
            standard_name="wind_speed_of_gust",
        ),
        _root(
            "max_upward_vertical_velocity_1000_100mb",
            element="MAXUVV",
            level="100-1000 mb above ground",
            window="max",
            short_name="maxuvv",
            long_name="Maximum upward vertical velocity",
            units="m s-1",
        ),
        _root(
            "max_downward_vertical_velocity_1000_100mb",
            element="MAXDVV",
            level="100-1000 mb above ground",
            window="max",
            short_name="maxdvv",
            long_name="Maximum downward vertical velocity",
            units="m s-1",
        ),
        _root(
            "vertical_velocity_geometric_0p5_0p8_sigma",
            element="DZDT",
            level="0.5-0.8 sigma layer",
            window="avg",
            short_name="dzdt",
            long_name="Vertical velocity (geometric)",
            units="m s-1",
        ),
        _root(
            "pressure_reduced_to_mean_sea_level",
            element="MSLMA",
            level="mean sea level",
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
        ),
        _root(
            "hourly_maximum_radar_reflectivity_1000m",
            element="MAXREF",
            level="1000 m above ground",
            window="max",
            short_name="maxref",
            long_name="Hourly maximum of simulated reflectivity",
            units="dBZ",
        ),
        _root(
            "derived_radar_reflectivity_263k_max",
            element="REFD",
            level="263 K level",
            window="max",
            short_name="refd",
            long_name="Derived radar reflectivity",
            units="dBZ",
        ),
        _root(
            "max_updraft_helicity_5000_2000m",
            element="MXUPHL",
            level="5000-2000 m above ground",
            window="max",
            short_name="uphl",
            long_name="Updraft Helicity",
            units="m2 s-2",
        ),
        _root(
            "min_updraft_helicity_5000_2000m",
            element="MNUPHL",
            level="5000-2000 m above ground",
            window="min",
            short_name="mnuphl",
            long_name="Minimum updraft helicity",
            units="m2 s-2",
        ),
        _root(
            "max_updraft_helicity_2000_0m",
            element="MXUPHL",
            level="2000-0 m above ground",
            window="max",
            short_name="uphl",
            long_name="Updraft Helicity",
            units="m2 s-2",
        ),
        _root(
            "min_updraft_helicity_2000_0m",
            element="MNUPHL",
            level="2000-0 m above ground",
            window="min",
            short_name="mnuphl",
            long_name="Minimum updraft helicity",
            units="m2 s-2",
        ),
        _root(
            "max_updraft_helicity_3000_0m",
            element="MXUPHL",
            level="3000-0 m above ground",
            window="max",
            short_name="uphl",
            long_name="Updraft Helicity",
            units="m2 s-2",
        ),
        _root(
            "min_updraft_helicity_3000_0m",
            element="MNUPHL",
            level="3000-0 m above ground",
            window="min",
            short_name="mnuphl",
            long_name="Minimum updraft helicity",
            units="m2 s-2",
        ),
        _root(
            "max_relative_vorticity_2000_0m",
            element="RELV",
            level="2000-0 m above ground",
            window="max",
            short_name="vo",
            long_name="Vorticity (relative)",
            units="s-1",
        ),
        _root(
            "max_relative_vorticity_1000_0m",
            element="RELV",
            level="1000-0 m above ground",
            window="max",
            short_name="vo",
            long_name="Vorticity (relative)",
            units="s-1",
        ),
        _root(
            "max_hail_diameter_atmosphere",
            element="HAIL",
            level="entire atmosphere",
            window="max",
            short_name="hail",
            long_name="Hail",
            units="m",
        ),
        _root(
            "max_hail_diameter_0p1sigma",
            element="HAIL",
            level="0.1 sigma level",
            window="max",
            short_name="hail",
            long_name="Hail",
            units="m",
        ),
        _root(
            "max_hail_diameter_surface",
            element="HAIL",
            level="surface",
            window="max",
            short_name="hail",
            long_name="Hail",
            units="m",
        ),
        _root(
            "max_column_integrated_graupel_atmosphere",
            element="TCOLG",
            level="entire atmosphere (considered as a single layer)",
            window="max",
            short_name="tcolg",
            long_name="Total column vertically-integrated graupel (snow pellets)",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_graupel",
        ),
        _root(
            "lightning_standard_deviation_1m",
            element="LTNGSD",
            level="1 m above ground",
            short_name="ltngsd",
            long_name="Lightning standard deviation",
            units="1",
        ),
        _root(
            "lightning_standard_deviation_2m",
            element="LTNGSD",
            level="2 m above ground",
            short_name="ltngsd",
            long_name="Lightning standard deviation",
            units="1",
        ),
        _root(
            "lightning_atmosphere",
            element="LTNG",
            level="entire atmosphere",
            short_name="ltng",
            long_name="Lightning",
            units="1",
        ),
        _root(
            "wind_u_80m",
            element="UGRD",
            level="80 m above ground",
            short_name="80u",
            long_name="80 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _root(
            "wind_v_80m",
            element="VGRD",
            level="80 m above ground",
            short_name="80v",
            long_name="80 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _root(
            "pressure_surface",
            element="PRES",
            level="surface",
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            standard_name="surface_air_pressure",
        ),
        _root(
            "geopotential_height_surface",
            element="HGT",
            level="surface",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "temperature_surface",
            element="TMP",
            level="surface",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _root(
            "total_snowfall_run_total_surface",
            element="ASNOW",
            level="surface",
            window="acc_run",
            short_name="asnow",
            long_name="Total snowfall",
            units="m",
            standard_name="thickness_of_snowfall_amount",
        ),
        _root(
            "moisture_availability_0m_underground",
            element="MSTAV",
            level="0 m underground",
            short_name="mstav",
            long_name="Moisture availability",
            units="percent",
        ),
        _root(
            "plant_canopy_surface_water_surface",
            element="CNWAT",
            level="surface",
            short_name="cnwat",
            long_name="Plant canopy surface water",
            units="kg m-2",
            standard_name="canopy_water_amount",
        ),
        _root(
            "snow_water_equivalent_surface",
            element="WEASD",
            level="surface",
            short_name="sd",
            long_name="Snow depth water equivalent",
            units="m",
            standard_name="lwe_thickness_of_surface_snow_amount",
            filters=[_WATER_KG_M2_TO_M_LWE],
        ),
        _root(
            "snow_area_fraction_surface",
            element="SNOWC",
            level="surface",
            short_name="snowc",
            long_name="Snow cover",
            units="1",
            standard_name="surface_snow_area_fraction",
            filters=[_PERCENT_TO_FRACTION],
        ),
        _root(
            "snow_thickness_surface",
            element="SNOD",
            level="surface",
            short_name="sde",
            long_name="Snow depth",
            units="m",
            standard_name="surface_snow_thickness",
        ),
        _root(
            "temperature_2m",
            element="TMP",
            level="2 m above ground",
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _root(
            "potential_temperature_2m",
            element="POT",
            level="2 m above ground",
            short_name="pt",
            long_name="Potential temperature",
            units="K",
            standard_name="air_potential_temperature",
        ),
        _root(
            "specific_humidity_2m",
            element="SPFH",
            level="2 m above ground",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        _root(
            "dew_point_temperature_2m",
            element="DPT",
            level="2 m above ground",
            short_name="2d",
            long_name="2 metre dewpoint temperature",
            units="degree_Celsius",
            standard_name="dew_point_temperature",
        ),
        _root(
            "relative_humidity_2m",
            element="RH",
            level="2 m above ground",
            short_name="2r",
            long_name="2 metre relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        _root(
            "mass_density_8m",
            element="MASSDEN",
            level="8 m above ground",
            short_name="mdens",
            long_name="Mass density",
            units="kg m-3",
        ),
        _root(
            "wind_u_10m",
            element="UGRD",
            level="10 m above ground",
            short_name="10u",
            long_name="10 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _root(
            "wind_v_10m",
            element="VGRD",
            level="10 m above ground",
            short_name="10v",
            long_name="10 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _root(
            "max_wind_speed_10m",
            element="WIND",
            level="10 m above ground",
            window="max",
            short_name="10si",
            long_name="10 metre wind speed",
            units="m s-1",
            standard_name="wind_speed",
        ),
        _root(
            "max_wind_u_component_10m",
            element="MAXUW",
            level="10 m above ground",
            window="max",
            short_name="maxuw",
            long_name="Maximum 10 metre wind speed u component",
            units="m s-1",
        ),
        _root(
            "max_wind_v_component_10m",
            element="MAXVW",
            level="10 m above ground",
            window="max",
            short_name="maxvw",
            long_name="Maximum 10 metre wind speed v component",
            units="m s-1",
        ),
        _root(
            "percent_frozen_precipitation_surface",
            element="CPOFP",
            level="surface",
            short_name="cpofp",
            long_name="Percent frozen precipitation",
            units="percent",
            hour_0=False,
        ),
        _root(
            "precipitation_rate_surface",
            element="PRATE",
            level="surface",
            short_name="prate",
            long_name="Precipitation rate",
            units="kg m-2 s-1",
            standard_name="precipitation_flux",
        ),
        _root(
            "total_precipitation_run_total_surface",
            element="APCP",
            level="surface",
            window="acc_run",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
        ),
        _root(
            "snowfall_water_equivalent_run_total_surface",
            element="WEASD",
            level="surface",
            window="acc_run",
            short_name="sf",
            long_name="Snowfall water equivalent",
            units="kg m-2",
            standard_name="snowfall_amount",
        ),
        _root(
            "frozen_precipitation_run_total_surface",
            element="FROZR",
            level="surface",
            window="acc_run",
            short_name="frozr",
            long_name="Frozen precipitation",
            units="kg m-2",
        ),
        _root(
            "freezing_rain_run_total_surface",
            element="FRZR",
            level="surface",
            window="acc_run",
            short_name="frzr",
            long_name="Freezing Rain",
            units="kg m-2",
        ),
        _root(
            "storm_surface_runoff_surface",
            element="SSRUN",
            level="surface",
            window="acc_1h",
            short_name="ssrun",
            long_name="Storm surface runoff",
            units="kg m-2",
            standard_name="surface_runoff_amount",
        ),
        _root(
            "baseflow_groundwater_runoff_surface",
            element="BGRUN",
            level="surface",
            window="acc_1h",
            short_name="bgrun",
            long_name="Baseflow-groundwater runoff",
            units="kg m-2",
            standard_name="subsurface_runoff_amount",
        ),
        _root(
            "total_precipitation_surface",
            element="APCP",
            level="surface",
            window="acc_1h",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
        ),
        _root(
            "snowfall_water_equivalent_surface",
            element="WEASD",
            level="surface",
            window="acc_1h",
            short_name="sf",
            long_name="Snowfall water equivalent",
            units="kg m-2",
            standard_name="snowfall_amount",
        ),
        _root(
            "frozen_precipitation_surface",
            element="FROZR",
            level="surface",
            window="acc_1h",
            short_name="frozr",
            long_name="Frozen precipitation",
            units="kg m-2",
        ),
        _root(
            "categorical_snow_surface",
            element="CSNOW",
            level="surface",
            short_name="csnow",
            long_name="Categorical snow",
            units="1",
            comment="0=no; 1=yes",
            hour_0=False,
        ),
        _root(
            "categorical_ice_pellets_surface",
            element="CICEP",
            level="surface",
            short_name="cicep",
            long_name="Categorical ice pellets",
            units="1",
            comment="0=no; 1=yes",
            hour_0=False,
        ),
        _root(
            "categorical_freezing_rain_surface",
            element="CFRZR",
            level="surface",
            short_name="cfrzr",
            long_name="Categorical freezing rain",
            units="1",
            comment="0=no; 1=yes",
            hour_0=False,
        ),
        _root(
            "categorical_rain_surface",
            element="CRAIN",
            level="surface",
            short_name="crain",
            long_name="Categorical rain",
            units="1",
            comment="0=no; 1=yes",
            hour_0=False,
        ),
        _root(
            "surface_roughness_surface",
            element="SFCR",
            level="surface",
            short_name="fsr",
            long_name="Forecast surface roughness",
            units="m",
            standard_name="surface_roughness_length",
        ),
        _root(
            "friction_velocity_surface",
            element="FRICV",
            level="surface",
            short_name="zust",
            long_name="Friction velocity",
            units="m s-1",
        ),
        _root(
            "sensible_heat_flux_surface",
            element="SHTFL",
            level="surface",
            short_name="shf",
            long_name="Sensible heat flux",
            units="W m-2",
            standard_name="surface_upward_sensible_heat_flux",
        ),
        _root(
            "latent_heat_flux_surface",
            element="LHTFL",
            level="surface",
            short_name="lhf",
            long_name="Latent heat flux",
            units="W m-2",
            standard_name="surface_upward_latent_heat_flux",
        ),
        _root(
            "vegetation_surface",
            element="VEG",
            level="surface",
            short_name="veg",
            long_name="Vegetation fraction",
            units="percent",
        ),
        _root(
            "minimum_vegetation_surface",
            element="VEGMIN",
            level="surface",
            short_name="vegmin",
            long_name="Minimum vegetation fraction",
            units="percent",
        ),
        _root(
            "maximum_vegetation_surface",
            element="VEGMAX",
            level="surface",
            short_name="vegmax",
            long_name="Maximum vegetation fraction",
            units="percent",
        ),
        _root(
            "leaf_area_index_surface",
            element="LAI",
            level="surface",
            short_name="lai",
            long_name="Leaf Area Index",
            units="1",
            standard_name="leaf_area_index",
        ),
        _root(
            "ground_heat_flux_surface",
            element="GFLUX",
            level="surface",
            short_name="gflux",
            long_name="Ground heat flux",
            units="W m-2",
        ),
        _root(
            "vegetation_type_surface",
            element="VGTYP",
            level="surface",
            short_name="vgtyp",
            long_name="Vegetation Type",
            units="1",
        ),
        _root(
            "surface_lifted_index_500_1000mb",
            element="LFTX",
            level="500-1000 mb",
            short_name="lftx",
            long_name="Surface lifted index",
            units="K",
        ),
        _root(
            "convective_available_potential_energy_surface",
            element="CAPE",
            level="surface",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        _root(
            "convective_inhibition_surface",
            element="CIN",
            level="surface",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        _root(
            "precipitable_water_atmosphere",
            element="PWAT",
            level="entire atmosphere (considered as a single layer)",
            short_name="pwat",
            long_name="Precipitable water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_water_vapor",
        ),
        _root(
            "aerosol_optical_thickness_atmosphere",
            element="AOTK",
            level="entire atmosphere (considered as a single layer)",
            short_name="aotk",
            long_name="Aerosol optical thickness",
            units="1",
            standard_name="atmosphere_optical_thickness_due_to_ambient_aerosol_particles",
        ),
        _root(
            "column_integrated_mass_density_atmosphere",
            element="COLMD",
            level="entire atmosphere (considered as a single layer)",
            short_name="colmd",
            long_name="Column-integrated mass density",
            units="kg m-2",
        ),
        _root(
            "total_column_cloud_water_atmosphere",
            element="TCOLW",
            level="entire atmosphere",
            short_name="tcolw",
            long_name="Total column-integrated cloud water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_cloud_liquid_water",
        ),
        _root(
            "total_column_cloud_ice_atmosphere",
            element="TCOLI",
            level="entire atmosphere",
            short_name="tcoli",
            long_name="Total column-integrated cloud ice",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_cloud_ice",
        ),
        _root(
            "total_cloud_cover_boundary_layer",
            element="TCDC",
            level="boundary layer cloud layer",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        _root(
            "low_cloud_cover",
            element="LCDC",
            level="low cloud layer",
            short_name="lcc",
            long_name="Low cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        _root(
            "medium_cloud_cover",
            element="MCDC",
            level="middle cloud layer",
            short_name="mcc",
            long_name="Medium cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        _root(
            "high_cloud_cover",
            element="HCDC",
            level="high cloud layer",
            short_name="hcc",
            long_name="High cloud cover",
            units="percent",
            standard_name="cloud_area_fraction_in_atmosphere_layer",
        ),
        _root(
            "total_cloud_cover_atmosphere",
            element="TCDC",
            level="entire atmosphere",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction",
        ),
        _root(
            "geopotential_height_cloud_ceiling",
            element="HGT",
            level="cloud ceiling",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "geopotential_height_cloud_base",
            element="HGT",
            level="cloud base",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "pressure_cloud_base",
            element="PRES",
            level="cloud base",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_base",
        ),
        _root(
            "pressure_cloud_top",
            element="PRES",
            level="cloud top",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure_at_cloud_top",
        ),
        _root(
            "geopotential_height_cloud_top",
            element="HGT",
            level="cloud top",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height_at_cloud_top",
        ),
        _root(
            "upward_long_wave_radiation_flux_top_of_atmosphere",
            element="ULWRF",
            level="top of atmosphere",
            short_name="ulwrf",
            long_name="Upward long-wave radiation flux",
            units="W m-2",
            standard_name="toa_outgoing_longwave_flux",
        ),
        _root(
            "downward_short_wave_radiation_flux_surface",
            element="DSWRF",
            level="surface",
            short_name="sdswrf",
            long_name="Surface downward short-wave radiation flux",
            units="W m-2",
            standard_name="surface_downwelling_shortwave_flux_in_air",
        ),
        _root(
            "downward_long_wave_radiation_flux_surface",
            element="DLWRF",
            level="surface",
            short_name="sdlwrf",
            long_name="Surface downward long-wave radiation flux",
            units="W m-2",
            standard_name="surface_downwelling_longwave_flux_in_air",
        ),
        _root(
            "upward_short_wave_radiation_flux_surface",
            element="USWRF",
            level="surface",
            short_name="suswrf",
            long_name="Surface upward short-wave radiation flux",
            units="W m-2",
            standard_name="surface_upwelling_shortwave_flux_in_air",
        ),
        _root(
            "upward_long_wave_radiation_flux_surface",
            element="ULWRF",
            level="surface",
            short_name="sulwrf",
            long_name="Surface upward long-wave radiation flux",
            units="W m-2",
            standard_name="surface_upwelling_longwave_flux_in_air",
        ),
        _root(
            "cloud_forcing_net_solar_flux_surface",
            element="CFNSF",
            level="surface",
            short_name="cfnsf",
            long_name="Cloud Forcing Net Solar Flux",
            units="W m-2",
        ),
        _root(
            "visible_beam_downward_solar_flux_surface",
            element="VBDSF",
            level="surface",
            short_name="vbdsf",
            long_name="Visible Beam Downward Solar Flux",
            units="W m-2",
        ),
        _root(
            "visible_diffuse_downward_solar_flux_surface",
            element="VDDSF",
            level="surface",
            short_name="vddsf",
            long_name="Visible Diffuse Downward Solar Flux",
            units="W m-2",
        ),
        _root(
            "upward_short_wave_radiation_flux_top_of_atmosphere",
            element="USWRF",
            level="top of atmosphere",
            short_name="uswrf",
            long_name="Upward short-wave radiation flux",
            units="W m-2",
            standard_name="toa_outgoing_shortwave_flux",
        ),
        _root(
            "storm_relative_helicity_3000_0m",
            element="HLCY",
            level="3000-0 m above ground",
            short_name="hlcy",
            long_name="Storm relative helicity",
            units="m2 s-2",
        ),
        _root(
            "storm_relative_helicity_1000_0m",
            element="HLCY",
            level="1000-0 m above ground",
            short_name="hlcy",
            long_name="Storm relative helicity",
            units="m2 s-2",
        ),
        _root(
            "u_component_storm_motion_0_6000m",
            element="USTM",
            level="0-6000 m above ground",
            short_name="ustm",
            long_name="U-component storm motion",
            units="m s-1",
        ),
        _root(
            "v_component_storm_motion_0_6000m",
            element="VSTM",
            level="0-6000 m above ground",
            short_name="vstm",
            long_name="V-component storm motion",
            units="m s-1",
        ),
        _root(
            "vertical_u_component_shear_0_1000m",
            element="VUCSH",
            level="0-1000 m above ground",
            short_name="vucsh",
            long_name="Vertical u-component shear",
            units="s-1",
        ),
        _root(
            "vertical_v_component_shear_0_1000m",
            element="VVCSH",
            level="0-1000 m above ground",
            short_name="vvcsh",
            long_name="Vertical v-component shear",
            units="s-1",
        ),
        _root(
            "vertical_u_component_shear_0_6000m",
            element="VUCSH",
            level="0-6000 m above ground",
            short_name="vucsh",
            long_name="Vertical u-component shear",
            units="s-1",
        ),
        _root(
            "vertical_v_component_shear_0_6000m",
            element="VVCSH",
            level="0-6000 m above ground",
            short_name="vvcsh",
            long_name="Vertical v-component shear",
            units="s-1",
        ),
        _root(
            "geopotential_height_0c_isotherm",
            element="HGT",
            level="0C isotherm",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "relative_humidity_0c_isotherm",
            element="RH",
            level="0C isotherm",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        _root(
            "pressure_0c_isotherm",
            element="PRES",
            level="0C isotherm",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
        ),
        _root(
            "geopotential_height_highest_tropospheric_freezing_level",
            element="HGT",
            level="highest tropospheric freezing level",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "relative_humidity_highest_tropospheric_freezing_level",
            element="RH",
            level="highest tropospheric freezing level",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        _root(
            "pressure_highest_tropospheric_freezing_level",
            element="PRES",
            level="highest tropospheric freezing level",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
        ),
        _root(
            "geopotential_height_263k",
            element="HGT",
            level="263 K level",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "geopotential_height_253k",
            element="HGT",
            level="253 K level",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "best_4_layer_lifted_index_180_0mb",
            element="4LFTX",
            level="180-0 mb above ground",
            short_name="4lftx",
            long_name="Best (4-layer) lifted index",
            units="K",
        ),
        _root(
            "convective_available_potential_energy_180_0mb",
            element="CAPE",
            level="180-0 mb above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        _root(
            "convective_inhibition_180_0mb",
            element="CIN",
            level="180-0 mb above ground",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        _root(
            "planetary_boundary_layer_height_surface",
            element="HPBL",
            level="surface",
            short_name="blh",
            long_name="Boundary layer height",
            units="m",
            standard_name="atmosphere_boundary_layer_thickness",
        ),
        _root(
            "geopotential_height_adiabatic_condensation_level",
            element="HGT",
            level="level of adiabatic condensation from sfc",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "convective_available_potential_energy_90_0mb",
            element="CAPE",
            level="90-0 mb above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        _root(
            "convective_inhibition_90_0mb",
            element="CIN",
            level="90-0 mb above ground",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        _root(
            "convective_available_potential_energy_255_0mb",
            element="CAPE",
            level="255-0 mb above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        _root(
            "convective_inhibition_255_0mb",
            element="CIN",
            level="255-0 mb above ground",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        _root(
            "geopotential_height_equilibrium_level",
            element="HGT",
            level="equilibrium level",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "pressure_of_lifted_parcel_level_255_0mb",
            element="PLPL",
            level="255-0 mb above ground",
            short_name="plpl",
            long_name="Pressure of level from which parcel was lifted",
            units="Pa",
        ),
        _root(
            "convective_available_potential_energy_0_3000m",
            element="CAPE",
            level="0-3000 m above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        _root(
            "geopotential_height_level_of_free_convection",
            element="HGT",
            level="level of free convection",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _root(
            "effective_layer_helicity_surface",
            element="EFHL",
            level="surface",
            short_name="efhl",
            long_name="Effective layer helicity",
            units="m2 s-2",
        ),
        _root(
            "critical_angle_0_500m",
            element="CANGLE",
            level="0-500 m above ground",
            short_name="cangle",
            long_name="Critical angle",
            units="degree",
        ),
        _root(
            "layer_thickness_261k_256k",
            element="LAYTH",
            level="261 K level - 256 K level",
            short_name="layth",
            long_name="Layer Thickness",
            units="m",
        ),
        _root(
            "enhanced_stretching_potential_0_3000m",
            element="ESP",
            level="0-3000 m above ground",
            short_name="esp",
            long_name="Enhanced stretching potential",
            units="1",
        ),
        _root(
            "relative_humidity_with_respect_to_precipitable_water_atmosphere",
            element="RHPW",
            level="entire atmosphere",
            short_name="rhpw",
            long_name="Relative humidity with respect to precipitable water",
            units="percent",
        ),
        _root(
            "land_sea_mask_surface",
            element="LAND",
            level="surface",
            short_name="lsm",
            long_name="Land-sea mask",
            units="1",
            standard_name="land_binary_mask",
        ),
        _root(
            "ice_cover_surface",
            element="ICEC",
            level="surface",
            short_name="icec",
            long_name="Ice cover (1=ice, 0=no ice)",
            units="1",
            standard_name="sea_ice_area_fraction",
        ),
        _root(
            "brightness_temperature_channel_123",
            element="SBT123",
            level="top of atmosphere",
            short_name="sbt123",
            long_name="Simulated brightness temperature (channel 123)",
            units="K",
            standard_name="toa_brightness_temperature",
        ),
        _root(
            "brightness_temperature_channel_124",
            element="SBT124",
            level="top of atmosphere",
            short_name="sbt124",
            long_name="Simulated brightness temperature (channel 124)",
            units="K",
            standard_name="toa_brightness_temperature",
        ),
        _root(
            "brightness_temperature_channel_113",
            element="SBT113",
            level="top of atmosphere",
            short_name="sbt113",
            long_name="Simulated brightness temperature (channel 113)",
            units="K",
            standard_name="toa_brightness_temperature",
        ),
        _root(
            "brightness_temperature_channel_114",
            element="SBT114",
            level="top of atmosphere",
            short_name="sbt114",
            long_name="Simulated brightness temperature (channel 114)",
            units="K",
            standard_name="toa_brightness_temperature",
        ),
    ]


def _pressure_data_vars() -> list[NoaaHrrrDataVar]:
    return [
        _pressure(
            "absolute_vorticity",
            element="ABSV",
            short_name="absv",
            long_name="Absolute vorticity",
            units="s-1",
            standard_name="atmosphere_upward_absolute_vorticity",
        ),
        _pressure(
            "cloud_ice_mixing_ratio",
            element="CIMIXR",
            short_name="icmr",
            long_name="Cloud ice mixing ratio",
            units="kg kg-1",
            standard_name="cloud_ice_mixing_ratio",
        ),
        _pressure(
            "cloud_mixing_ratio",
            element="CLMR",
            short_name="clwmr",
            long_name="Cloud mixing ratio",
            units="kg kg-1",
            standard_name="cloud_liquid_water_mixing_ratio",
        ),
        _pressure(
            "dew_point_temperature",
            element="DPT",
            short_name="dpt",
            long_name="Dew point temperature",
            units="degree_Celsius",
            standard_name="dew_point_temperature",
        ),
        _pressure(
            "graupel",
            element="GRLE",
            short_name="grle",
            long_name="Graupel (snow pellets)",
            units="kg kg-1",
        ),
        _pressure(
            "geopotential_height",
            element="HGT",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _pressure(
            "relative_humidity",
            element="RH",
            short_name="r",
            long_name="Relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        _pressure(
            "rain_mixing_ratio",
            element="RWMR",
            short_name="rwmr",
            long_name="Rain mixing ratio",
            units="kg kg-1",
        ),
        _pressure(
            "snow_mixing_ratio",
            element="SNMR",
            short_name="snmr",
            long_name="Snow mixing ratio",
            units="kg kg-1",
        ),
        _pressure(
            "specific_humidity",
            element="SPFH",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        _pressure(
            "temperature",
            element="TMP",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _pressure(
            "wind_u",
            element="UGRD",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _pressure(
            "wind_v",
            element="VGRD",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _pressure(
            "vertical_velocity",
            element="VVEL",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
        ),
    ]


def _model_data_vars() -> list[NoaaHrrrDataVar]:
    return [
        _model(
            "temperature",
            element="TMP",
            short_name="t",
            long_name="Temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        _model(
            "specific_humidity",
            element="SPFH",
            short_name="q",
            long_name="Specific humidity",
            units="1",
            standard_name="specific_humidity",
        ),
        _model(
            "wind_u",
            element="UGRD",
            short_name="u",
            long_name="U component of wind",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        _model(
            "wind_v",
            element="VGRD",
            short_name="v",
            long_name="V component of wind",
            units="m s-1",
            standard_name="northward_wind",
        ),
        _model(
            "vertical_velocity",
            element="VVEL",
            short_name="w",
            long_name="Vertical velocity",
            units="Pa s-1",
            standard_name="lagrangian_tendency_of_air_pressure",
        ),
        _model(
            "geopotential_height",
            element="HGT",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
        ),
        _model(
            "pressure",
            element="PRES",
            short_name="pres",
            long_name="Pressure",
            units="Pa",
            standard_name="air_pressure",
        ),
        _model(
            "turbulent_kinetic_energy",
            element="TKE",
            short_name="tke",
            long_name="Turbulent kinetic energy",
            units="J kg-1",
        ),
        _model(
            "cloud_mixing_ratio",
            element="CLMR",
            short_name="clwmr",
            long_name="Cloud mixing ratio",
            units="kg kg-1",
            standard_name="cloud_liquid_water_mixing_ratio",
        ),
        _model(
            "cloud_ice_mixing_ratio",
            element="CIMIXR",
            short_name="icmr",
            long_name="Cloud ice mixing ratio",
            units="kg kg-1",
            standard_name="cloud_ice_mixing_ratio",
        ),
        _model(
            "rain_mixing_ratio",
            element="RWMR",
            short_name="rwmr",
            long_name="Rain mixing ratio",
            units="kg kg-1",
        ),
        _model(
            "snow_mixing_ratio",
            element="SNMR",
            short_name="snmr",
            long_name="Snow mixing ratio",
            units="kg kg-1",
        ),
        _model(
            "graupel",
            element="GRLE",
            short_name="grle",
            long_name="Graupel (snow pellets)",
            units="kg kg-1",
        ),
        _model(
            "mass_density",
            element="MASSDEN",
            short_name="mdens",
            long_name="Mass density",
            units="kg m-3",
        ),
        _model(
            "fraction_of_cloud_cover",
            element="FRACCC",
            short_name="ccl",
            long_name="Cloud cover",
            units="percent",
        ),
        _model(
            "number_concentration_cloud_ice",
            element="NCCICE",
            short_name="nccice",
            long_name="Number concentration of cloud ice",
            units="kg-1",
        ),
        _model(
            "number_concentration_cloud_droplets",
            element="NCONCD",
            short_name="nconcd",
            long_name="Number concentration of cloud droplets",
            units="kg-1",
        ),
        _model(
            "number_concentration_rain",
            element="SPNCR",
            short_name="spncr",
            long_name="Number concentration of rain",
            units="kg-1",
        ),
        _model(
            "particulate_matter_fine",
            element="PMTF",
            short_name="pmtf",
            long_name="Particulate matter (fine)",
            units="kg m-3",
        ),
        _model(
            "particulate_matter_coarse",
            element="PMTC",
            short_name="pmtc",
            long_name="Particulate matter (coarse)",
            units="kg m-3",
        ),
    ]
