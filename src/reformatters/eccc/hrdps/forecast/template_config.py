import functools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from pydantic import computed_field

from reformatters.common.config_models import (
    ROOT,
    BaseInternalAttrs,
    Coordinate,
    CoordinateAttrs,
    DatasetAttributes,
    DataVar,
    DataVarAttrs,
    Encoding,
    StatisticsApproximate,
)
from reformatters.common.deaccumulation import (
    PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD,
    RADIATION_INVALID_BELOW_THRESHOLD,
)
from reformatters.common.template_config import (
    SPATIAL_REF_COORDS,
    TemplateConfig,
)
from reformatters.common.types import (
    AppendDim,
    Array1D,
    Array2D,
    Dim,
    Dims,
    Timedelta,
    Timestamp,
)
from reformatters.common.zarr import (
    BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE,
    BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE,
)


class EcccHrdpsInternalAttrs(BaseInternalAttrs):
    """
    Variable specific attributes used internally to drive processing.
    Not written to the dataset.
    """

    # The {FIELD} and {LEVEL} parts of the source file name,
    # e.g. "APCP-Accum1h" and "Sfc" in ..._MSC_HRDPS_APCP-Accum1h_Sfc_....grib2
    grib_field: str
    grib_level: str
    window_reset_frequency: Timedelta | None = None
    # Multiply raw values by this factor after reading (e.g. 0.001 to convert kg m-2 to m lwe)
    scale_factor: float | None = None
    deaccumulation_invalid_below_threshold_rate: float = (
        PRECIPITATION_RATE_INVALID_BELOW_THRESHOLD
    )


class EcccHrdpsDataVar(DataVar[EcccHrdpsInternalAttrs]):
    pass


class EcccHrdpsForecastTemplateConfig(TemplateConfig[EcccHrdpsDataVar]):
    dims: Dims = {ROOT: ("init_time", "lead_time", "y", "x")}
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp("2026-07-09T00:00")
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="eccc-hrdps-forecast",
            dataset_version="0.1.0",
            name="ECCC HRDPS forecast",
            description="Weather forecasts from the High Resolution Deterministic "
            "Prediction System (HRDPS) continental domain, operated by Environment "
            "and Climate Change Canada (ECCC).",
            attribution="ECCC HRDPS data processed by dynamical.org from Environment "
            "and Climate Change Canada, used under the ECCC Data Servers End-use "
            "Licence version 2.1 (https://eccc-msc.github.io/open-data/licence/readme_en/).",
            license="CC-BY-4.0",
            spatial_domain="Canada and the northern continental United States",
            spatial_resolution="2.5 km",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-48 hours ahead",
            forecast_resolution="Forecast step 0-48 hours: hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns a dictionary of dimension names to coordinates for the dataset."""
        y_coords, x_coords = self._y_x_coordinates()
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": pd.timedelta_range("0h", "48h", freq="1h"),
            "y": y_coords,
            "x": x_coords,
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """
        Return a dictionary of non-dimension coordinates for the dataset.
        Called whenever len(ds.append_dim) changes.
        """
        latitudes, longitudes = self._latitude_longitude_coordinates(
            ds["x"].values, ds["y"].values
        )
        return {
            "valid_time": ds["init_time"] + ds["lead_time"],
            "expected_forecast_length": (
                (self.append_dim,),
                np.full(ds[self.append_dim].size, np.timedelta64(48, "h")),
            ),
            "ingested_forecast_length": (
                (self.append_dim,),
                np.full(ds[self.append_dim].size, np.timedelta64("NaT", "us")),
            ),
            "latitude": (("y", "x"), latitudes),
            "longitude": (("y", "x"), longitudes),
            "spatial_ref": SPATIAL_REF_COORDS,
        }

    @computed_field
    @property
    def coords(self) -> Sequence[Coordinate]:
        """Define metadata and encoding for each coordinate."""
        dim_coords = self.dimension_coordinates()
        append_dim_coordinate_chunk_size = self.append_dim_coordinate_chunk_size()
        y_coords, x_coords = dim_coords["y"], dim_coords["x"]

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
                name="x",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(x_coords),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Longitude in rotated pole grid",
                    standard_name="grid_longitude",
                    units="degrees",
                    axis="X",
                    statistics_approximate=StatisticsApproximate(
                        min=float(x_coords.min()),
                        max=float(x_coords.max()),
                    ),
                ),
            ),
            Coordinate(
                name="y",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=np.nan,
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=len(y_coords),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Latitude in rotated pole grid",
                    standard_name="grid_latitude",
                    units="degrees",
                    axis="Y",
                    statistics_approximate=StatisticsApproximate(
                        min=float(y_coords.min()),
                        max=float(y_coords.max()),
                    ),
                ),
            ),
            Coordinate(
                name="latitude",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=(len(y_coords), len(x_coords)),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Latitude",
                    standard_name="latitude",
                    units="degree_north",
                    statistics_approximate=StatisticsApproximate(
                        min=27.284597,
                        max=70.61148,
                    ),
                ),
            ),
            Coordinate(
                name="longitude",
                encoding=Encoding(
                    dtype="float32",
                    fill_value=np.nan,
                    compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
                    chunks=(len(y_coords), len(x_coords)),
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Longitude",
                    standard_name="longitude",
                    units="degree_east",
                    statistics_approximate=StatisticsApproximate(
                        min=-152.73067,
                        max=-40.70856,
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
                name="ingested_forecast_length",
                encoding=Encoding(
                    dtype="float64",
                    fill_value=float("nan"),
                    compressors=[BLOSC_8BYTE_ZSTD_LEVEL3_SHUFFLE],
                    units="seconds",
                    chunks=append_dim_coordinate_chunk_size,
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    long_name="Ingested forecast length",
                    units="seconds",
                    statistics_approximate=StatisticsApproximate(
                        min=str(dim_coords["lead_time"].min()),
                        max=str(dim_coords["lead_time"].max()),
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
                    chunks=(),  # Scalar coordinate
                    shards=None,
                ),
                attrs=CoordinateAttrs(
                    units=None,
                    statistics_approximate=None,
                    # Derived from opening a sample HRDPS file, see
                    # tests/eccc/hrdps/forecast/template_config_test.py::test_spatial_info_matches_file
                    GeoTransform="-14.832470000590822 0.022500001181567565 0.0 16.711251000775796 0.0 -0.02250000155159038",
                    crs_wkt=_CRS_WKT,
                    spatial_ref=_CRS_WKT,
                    geographic_crs_name="Coordinate System imported from GRIB file",
                    grid_mapping_name="rotated_latitude_longitude",
                    # The source states the southern pole; CF names the northern one.
                    grid_north_pole_latitude=36.08852,
                    grid_north_pole_longitude=65.305142,
                    horizontal_datum_name="unnamed",
                    inverse_flattening=0.0,
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    reference_ellipsoid_name="Sphere",
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                ),
            ),
        ]

    @computed_field
    @property
    def data_vars(self) -> Sequence[EcccHrdpsDataVar]:
        """Define metadata and encoding for each data variable."""
        # ~12MB uncompressed, ~2.5MB compressed
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,  # all lead times
            "y": 258,  # 5 chunks (1290 pixels)
            "x": 254,  # 10 chunks (2540 pixels)
        }

        # Single shard for each init time
        # ~612MB uncompressed, ~122MB compressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 49,
            "y": var_chunks["y"] * 5,
            "x": var_chunks["x"] * 10,
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims[ROOT]),
            shards=tuple(var_shards[d] for d in self.dims[ROOT]),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 7
        # A wind direction spans 0-360 degrees, where the wind rounding of 6 bits would
        # quantize to ~2.8 degrees, far coarser than the source's 0.1 degree steps.
        wind_direction_keep_mantissa_bits = 12

        return [
            EcccHrdpsDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="2t",
                    long_name="2 metre temperature",
                    units="degree_Celsius",
                    step_type="instant",
                    standard_name="air_temperature",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="TMP",
                    grib_level="AGL-2m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="dew_point_temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="2d",
                    long_name="2 metre dewpoint temperature",
                    units="degree_Celsius",
                    step_type="instant",
                    standard_name="dew_point_temperature",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="DPT",
                    grib_level="AGL-2m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="specific_humidity_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="q",
                    long_name="Specific humidity",
                    units="1",
                    step_type="instant",
                    standard_name="specific_humidity",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="SPFH",
                    grib_level="AGL-2m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_speed_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="10si",
                    long_name="10 metre wind speed",
                    units="m s-1",
                    step_type="instant",
                    standard_name="wind_speed",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="WIND",
                    grib_level="AGL-10m",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_direction_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="10wdir",
                    long_name="10 metre wind direction",
                    units="degree",
                    step_type="instant",
                    standard_name="wind_from_direction",
                    comment="Direction the wind blows from, clockwise from true north rather than from the rotated grid's north.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="WDIR",
                    grib_level="AGL-10m",
                    keep_mantissa_bits=wind_direction_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_speed_80m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="80si",
                    long_name="80 metre wind speed",
                    units="m s-1",
                    step_type="instant",
                    standard_name="wind_speed",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="WIND",
                    grib_level="AGL-80m",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_direction_80m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="80wdir",
                    long_name="80 metre wind direction",
                    units="degree",
                    step_type="instant",
                    standard_name="wind_from_direction",
                    comment="Direction the wind blows from, clockwise from true north rather than from the rotated grid's north.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="WDIR",
                    grib_level="AGL-80m",
                    keep_mantissa_bits=wind_direction_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="wind_gust_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="gust",
                    long_name="Wind speed (gust)",
                    units="m s-1",
                    step_type="instant",
                    standard_name="wind_speed_of_gust",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="GUST",
                    grib_level="AGL-10m",
                    keep_mantissa_bits=6,
                ),
            ),
            EcccHrdpsDataVar(
                name="pressure_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sp",
                    long_name="Surface pressure",
                    units="Pa",
                    step_type="instant",
                    standard_name="surface_air_pressure",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="PRES",
                    grib_level="Sfc",
                    keep_mantissa_bits=11,
                ),
            ),
            EcccHrdpsDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    long_name="Pressure reduced to MSL",
                    units="Pa",
                    step_type="instant",
                    standard_name="air_pressure_at_mean_sea_level",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="PRMSL",
                    grib_level="MSL",
                    keep_mantissa_bits=11,
                ),
            ),
            EcccHrdpsDataVar(
                name="precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prate",
                    long_name="Precipitation rate",
                    units="kg m-2 s-1",
                    step_type="avg",
                    standard_name="precipitation_flux",
                    comment="Average precipitation rate since the previous forecast step. Units equivalent to mm/s.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="APCP-Accum1h",
                    grib_level="Sfc",
                    keep_mantissa_bits=8,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta("1h"),
                ),
            ),
            EcccHrdpsDataVar(
                name="categorical_precipitation_type_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="ptype",
                    long_name="Precipitation type",
                    units="1",
                    step_type="instant",
                    comment="1=Rain; 2=Rain/snow; 3=Freezing rain; 4=Ice pellets; 5=Snow; 6=None; 7=Drizzle; 8=Freezing drizzle; 9=Freezing rain/ice pellets. These are ECCC's codes, not the GRIB 4.201 codes the source messages claim.",
                    flag_values=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                    flag_meanings="rain mixture_of_rain_and_snow freezing_rain ice_pellets snow no_precipitation drizzle freezing_drizzle mixture_of_freezing_rain_and_ice_pellets",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="PTYPE",
                    grib_level="Sfc",
                    keep_mantissa_bits="no-rounding",
                    hour_0_values_override=False,
                ),
            ),
            EcccHrdpsDataVar(
                name="downward_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdswrf",
                    long_name="Surface downward short-wave radiation flux",
                    units="W m-2",
                    step_type="avg",
                    standard_name="surface_downwelling_shortwave_flux_in_air",
                    comment="Average flux since the previous forecast step.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="DSWRF",
                    grib_level="Sfc",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcccHrdpsDataVar(
                name="downward_long_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sdlwrf",
                    long_name="Surface downward long-wave radiation flux",
                    units="W m-2",
                    step_type="avg",
                    standard_name="surface_downwelling_longwave_flux_in_air",
                    comment="Average flux since the previous forecast step.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="DLWRF",
                    grib_level="Sfc",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,
                    deaccumulation_invalid_below_threshold_rate=RADIATION_INVALID_BELOW_THRESHOLD,
                ),
            ),
            EcccHrdpsDataVar(
                name="total_cloud_cover_atmosphere",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tcc",
                    long_name="Total cloud cover",
                    units="percent",
                    step_type="instant",
                    standard_name="cloud_area_fraction",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="TCDC",
                    grib_level="Sfc",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    # The source's lead time 0 field is zero everywhere, in every run.
                    hour_0_values_override=False,
                ),
            ),
            EcccHrdpsDataVar(
                name="convective_available_potential_energy_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cape",
                    long_name="Convective available potential energy",
                    units="J kg-1",
                    step_type="instant",
                    standard_name="atmosphere_convective_available_potential_energy",
                    comment="Values of -1, over roughly half the domain, and -999, a handful of cells per forecast step, are source markers rather than energies. Mask values < -0.1.",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="CAPE",
                    grib_level="Sfc",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            EcccHrdpsDataVar(
                name="snow_water_equivalent_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sd",
                    long_name="Snow depth water equivalent",
                    units="m",
                    step_type="instant",
                    standard_name="lwe_thickness_of_surface_snow_amount",
                ),
                internal_attrs=EcccHrdpsInternalAttrs(
                    grib_field="SDWE",
                    grib_level="Sfc",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    # Source values are kg m-2; 1 kg m-2 = 0.001 m lwe
                    scale_factor=0.001,
                ),
            ),
        ]

    def _spatial_info(
        self,
    ) -> tuple[
        tuple[int, int], tuple[float, float, float, float], tuple[float, float], str
    ]:
        """
        Returns (shape, bounds, resolution, crs proj4 string).
        Useful for deriving x, y and latitude, longitude coordinates.
        See tests/eccc/hrdps/forecast/template_config_test.py::test_spatial_info_matches_file
        """
        return (
            (1290, 2540),
            (
                -14.832470000590822,
                -12.313751000775795,
                42.31753300059079,
                16.711251000775796,
            ),
            (0.022500001181567565, -0.02250000155159038),
            "+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=36.08852 +lon_0=-114.694858 +R=6371229 +no_defs=True",
        )

    def _y_x_coordinates(self) -> tuple[Array1D[np.float64], Array1D[np.float64]]:
        shape, bounds, resolution, _crs = self._spatial_info()
        dx, dy = resolution
        left, _bottom, _right, top = bounds
        ny, nx = shape
        # add 1/2 a pixel to corner of bounds to get pixel center
        y_coords = (top + (0.5 * dy)) + (np.arange(ny) * dy)
        x_coords = (left + (0.5 * dx)) + (np.arange(nx) * dx)
        # astype is no-op for type checker
        return y_coords.astype(np.float64), x_coords.astype(np.float64)

    def _latitude_longitude_coordinates(
        self, x_coords: Array1D[np.float64], y_coords: Array1D[np.float64]
    ) -> tuple[Array2D[np.float32], Array2D[np.float32]]:
        _, _, _, crs = self._spatial_info()
        return _latitude_longitude_grids(crs, x_coords.tobytes(), y_coords.tobytes())


_CRS_WKT = 'GEOGCRS["Coordinate System imported from GRIB file",BASEGEOGCRS["Coordinate System imported from GRIB file",DATUM["unnamed",ELLIPSOID["Sphere",6371229,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],DERIVINGCONVERSION["Pole rotation (GRIB convention)",METHOD["Pole rotation (GRIB convention)"],PARAMETER["Latitude of the southern pole (GRIB convention)",-36.08852,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]],PARAMETER["Longitude of the southern pole (GRIB convention)",-114.694858,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]],PARAMETER["Axis rotation (GRIB convention)",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]]'


# The inverse transform costs ~1s on the full HRDPS grid while a template build
# requests it once per zarr group. Callers must not mutate the returned (cached) arrays.
@functools.cache
def _latitude_longitude_grids(
    crs: str, x_bytes: bytes, y_bytes: bytes
) -> tuple[Array2D[np.float32], Array2D[np.float32]]:
    x_coords = np.frombuffer(x_bytes, dtype=np.float64)
    y_coords = np.frombuffer(y_bytes, dtype=np.float64)
    xs, ys = np.meshgrid(x_coords, y_coords)
    # PROJ's ob_tran takes and returns radians, not the degrees the grid is defined in.
    lons, lats = pyproj.Proj(crs)(np.radians(xs), np.radians(ys), inverse=True)
    # Dropping to 32 bit precision still gets us < 1 meter precision and
    # makes each array about 13MB vs 26MB for float64.
    lats = lats.astype(np.float32)
    lons = lons.astype(np.float32)
    return lats, lons
