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


class DwdIconEuInternalAttrs(BaseInternalAttrs):
    """Variable specific attributes used internally to drive processing.

    Not written to the dataset.

    Attributes:
        variable_name_in_filename (str): The name used in ICON-EU's GRIB filename for this variable.
            For example, `alb_rad` (for `surface_albedo`).
    """

    variable_name_in_filename: str
    window_reset_frequency: Timedelta | None = None


class DwdIconEuDataVar(DataVar[DwdIconEuInternalAttrs]):
    pass


class DwdIconEuForecastTemplateConfig(TemplateConfig[DwdIconEuDataVar]):
    dims: tuple[Dim, ...] = ("init_time", "lead_time", "latitude", "longitude")
    append_dim: AppendDim = "init_time"
    append_dim_start: Timestamp = pd.Timestamp(
        "2025-10-03T00:00"  # TODO @JackKelly: Update this when we actual deploy operationally.
    )
    append_dim_frequency: Timedelta = pd.Timedelta("6h")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id="dwd-icon-eu-forecast",
            dataset_version="0.1.0",
            name="DWD ICON-EU Forecast",
            description="High-resolution weather forecasts for Europe from the ICON-EU model operated by Deutscher Wetterdienst (DWD).",
            attribution="DWD ICON-EU data processed by dynamical.org.",
            spatial_domain="Europe",
            spatial_resolution="0.0625 degrees (~7km)",
            time_domain=f"Forecasts initialized {self.append_dim_start} UTC to Present",
            time_resolution=f"Forecasts initialized every {self.append_dim_frequency.total_seconds() / 3600:.0f} hours",
            forecast_domain="Forecast lead time 0-120 hours (0-5 days) ahead",
            forecast_resolution="Forecast step 0-78 hours: hourly, 81-120 hours: 3 hourly",
        )

    def dimension_coordinates(self) -> dict[str, Any]:
        """Returns a dictionary of dimension names to coordinates for the
        dataset."""
        return {
            self.append_dim: self.append_dim_coordinates(
                self.append_dim_start + self.append_dim_frequency
            ),
            "lead_time": (  # Called "step" in the ICON-EU GRIB files.
                pd.timedelta_range("0h", "78h", freq="1h").union(
                    pd.timedelta_range("81h", "120h", freq="3h")
                )
            ),
            # These coordinates are for the pixel centers:
            "latitude": np.linspace(29.5, 70.5, 657),
            "longitude": np.linspace(-23.5, 62.5, 1377),
        }

    def derive_coordinates(
        self, ds: xr.Dataset
    ) -> dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray[Any, Any]]]:
        """Return a dictionary of non-dimension coordinates for the dataset.

        Called whenever len(ds.append_dim) changes.
        """
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
            "spatial_ref": SPATIAL_REF_COORDS,
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
                    long_name="Forecast initialization time",
                    standard_name="forecast_reference_time",
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
                        max="Present + 5 days",
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
                        min=str(dim_coords["lead_time"].min()),
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
                    # Derived by installing xarray, cfgrib, and rioxarray, and then running:
                    #     from pyproj import CRS
                    #     spherical_crs = CRS.from_wkt(WKT_STRING_EXTRACTED_FROM_ICON_EU_GRIB_BY_GDALINFO)
                    #     ds = xr.load_dataset(ICON_EU_GRIB_FILENAME_FROM_DWD, engine='cfgrib')
                    #     ds.rio.write_crs(spherical_crs)["spatial_ref"].attrs
                    crs_wkt='GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]',
                    semi_major_axis=6371229.0,
                    semi_minor_axis=6371229.0,
                    inverse_flattening=0.0,
                    reference_ellipsoid_name="Sphere",
                    longitude_of_prime_meridian=0.0,
                    prime_meridian_name="Greenwich",
                    geographic_crs_name="Coordinate System imported from GRIB file",
                    horizontal_datum_name="unnamed",
                    grid_mapping_name="latitude_longitude",
                    spatial_ref='GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]',
                    comment="A perfect sphere geographic CRS with a radius of 6,371,229m, extracted from grib.",
                ),
            ),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_vars(self) -> Sequence[DwdIconEuDataVar]:
        """Define metadata and encoding for each data variable."""
        # Roughly 12.5MB uncompressed, 2.5MB compressed
        var_chunks: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 93,
            "latitude": 219,  # 219 = 657 / 3
            "longitude": 153,  # 153 = 1377 / 9
        }
        # Roughly 337MB uncompressed, 67MB compressed
        var_shards: dict[Dim, int] = {
            "init_time": 1,
            "lead_time": 93,
            "latitude": 657,
            "longitude": 1377,
        }

        encoding_float32_default = Encoding(
            dtype="float32",
            fill_value=np.nan,
            chunks=tuple(var_chunks[d] for d in self.dims),
            shards=tuple(var_shards[d] for d in self.dims),
            compressors=[BLOSC_4BYTE_ZSTD_LEVEL3_SHUFFLE],
        )

        default_keep_mantissa_bits = 7

        return [
            # Some of the `comment` text is taken from the DWD Database Reference PDF:
            # https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf
            #
            # We don't include `alb_rad` (shortwave broadband albedo for
            # diffuse radiation) in the Zarr because, to quote the DWD
            # Database Reference: "Values over snow-free land points are based
            # on a monthly mean MODIS climatology." It's much more data-efficient
            # to just download those monthly means from DWD.
            DwdIconEuDataVar(
                name="downward_diffuse_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="aswdifd_s",
                    long_name="Downward diffusive short wave radiation flux at surface",
                    units="W m-2",
                    step_type="avg",
                    standard_name="surface_diffuse_downwelling_shortwave_flux_in_air",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="aswdifd_s",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="downward_direct_short_wave_radiation_flux_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="aswdir_s",
                    standard_name="surface_direct_downwelling_shortwave_flux_in_air",
                    long_name="Downward direct short wave radiation flux at surface",
                    units="W m-2",
                    step_type="avg",
                    comment=(
                        "Downward solar direct radiation flux at the surface, averaged over forecast time."
                        " This quantity is not directly provided by the radiation scheme."
                        " It is aposteriori diagnosed from the definition of the surface net"
                        " shortwave radiation flux."
                    ),
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="aswdir_s",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="convective_available_potential_energy",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="cape_con",
                    standard_name="atmosphere_convective_available_potential_energy",
                    long_name="Convective available potential energy",
                    units="J kg-1",
                    step_type="instant",
                    comment="Convective available potential energy",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="cape_con",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="high_cloud_cover",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="clch",
                    standard_name="cloud_area_fraction_in_atmosphere_layer",
                    long_name="High level clouds",
                    units="percent",
                    step_type="instant",
                    comment="Cloud Cover (0 - 400 hPa). Different agencies use different short_names for this same parameter: ECMWF: HCC; WMO GRIB table: HCDC.",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="clch",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="low_cloud_cover",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="clcl",
                    standard_name="cloud_area_fraction_in_atmosphere_layer",
                    long_name="Low level clouds",
                    units="percent",
                    step_type="instant",
                    comment="Cloud Cover (800 hPa - Soil). Different agencies use different short_names for this same parameter: ECMWF: LCC; WMO GRIB table: LCDC.",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="clcl",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="medium_cloud_cover",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="clcm",
                    standard_name="cloud_area_fraction_in_atmosphere_layer",
                    long_name="Mid level clouds",
                    units="percent",
                    step_type="instant",
                    comment="Cloud Cover (400 - 800 hPa). Different agencies use different short_names for this same parameter: ECMWF: MCC; WMO GRIB table: MCDC.",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="clcm",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="total_cloud_cover",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="clct",
                    standard_name="cloud_area_fraction",
                    long_name="Total Cloud Cover",
                    units="percent",
                    step_type="instant",
                    comment="Total cloud cover. Different agencies use different short_names for this same parameter: ECMWF: TCC; NOAA & WMO: TCDC.",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="clct",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="snow_depth",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sde",
                    long_name="Snow depth",
                    standard_name="surface_snow_thickness",
                    units="m",
                    step_type="instant",
                    comment=(
                        "Snow depth in m. It is diagnosed from RHO_SNOW and W_SNOW according to"
                        " H_SNOW = W_SNOW / RHO_SNOW and is limited to H_SNOW <= 40 m."
                    ),
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="h_snow",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="pressure_reduced_to_mean_sea_level",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="prmsl",
                    standard_name="air_pressure_at_mean_sea_level",
                    long_name="Pressure reduced to mean sea level (MSL)",
                    units="Pa",
                    step_type="instant",
                    comment="Surface pressure reduced to MSL",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="pmsl",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="relative_humidity_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="r2",
                    long_name="2 metre relative humidity",
                    units="percent",
                    step_type="instant",
                    comment="Relative humidity at 2m above ground. Other short_names used for this parameter: rh, 2r, r.",
                    standard_name="relative_humidity",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="relhum_2m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="soil_water_runoff",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="watr",
                    long_name="Soil water runoff",
                    units="kg m-2",
                    step_type="accum",
                    comment="Soil water runoff (accumulated since model start)",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="runoff_g",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    deaccumulate_to_rate=True,
                ),
            ),
            DwdIconEuDataVar(
                name="surface_water_runoff",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="watr",
                    standard_name="surface_runoff_amount",
                    long_name="Surface water Runoff",
                    units="kg m-2",
                    step_type="accum",
                    comment=(
                        "Surface water runoff from interception and snow reservoir and from"
                        " limited infiltration rate. Sum over forecast."
                    ),
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="runoff_s",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    deaccumulate_to_rate=True,
                ),
            ),
            DwdIconEuDataVar(
                name="temperature_2m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="t2m",
                    long_name="2 metre temperature",
                    units="K",
                    step_type="instant",
                    comment=(
                        "Temperature at 2m above ground, averaged over all tiles of a grid point. Different agencies use different short_names for this parameter: ECMWF: 2t; NOAA & DWD: t2m."
                    ),
                    standard_name="air_temperature",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="t_2m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="precipitation_surface",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="tp",
                    standard_name="precipitation_flux",
                    long_name="Total Precipitation",
                    units="kg m-2 s-1",
                    step_type="accum",
                    comment=(
                        "Precipitation rate since previous forecast step."
                        " TOT_PREC = RAIN_GSP + SNOW_GSP + RAIN_CON + SNOW_CON."
                        " Units equivalent to mm/s."
                    ),
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="tot_prec",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                    deaccumulate_to_rate=True,
                    window_reset_frequency=pd.Timedelta.max,  # accumulates over full lead time, never resetting
                ),
            ),
            DwdIconEuDataVar(
                name="wind_u_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="u10",
                    long_name="10 metre U wind component (eastward)",
                    units="m s-1",
                    step_type="instant",
                    standard_name="eastward_wind",
                    comment="Zonal wind at 10m above ground",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="u_10m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="wind_v_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="v10",
                    long_name="10 metre V wind component (northward)",
                    units="m s-1",
                    step_type="instant",
                    standard_name="northward_wind",
                    comment="Meridional wind at 10m above ground",
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="v_10m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="maximum_wind_10m",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="i10fg",
                    long_name="Time-maximum instantaneous 10 metre wind gust",
                    standard_name="wind_speed_of_gust",
                    units="m s-1",
                    step_type="max",
                    comment=(
                        "Maximum wind gust at 10 m above ground. It is diagnosed from the turbulence"
                        " state in the atmospheric boundary layer, including a potential"
                        " enhancement by the SSO parameterization over mountainous terrain."
                        " In the presence of deep convection, it contains an additional"
                        " contribution due to convective gusts."
                    ),
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="vmax_10m",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
            DwdIconEuDataVar(
                name="snow_depth_water_equivalent",
                encoding=encoding_float32_default,
                attrs=DataVarAttrs(
                    short_name="sd",
                    standard_name="lwe_thickness_of_surface_snow_amount",
                    long_name="Snow depth water equivalent",
                    units="mm",
                    step_type="instant",
                    comment=(
                        "Snow depth water equivalent in mm (kg/m2)."
                        " Set to 0 above water surfaces and snow-free land points."
                    ),
                ),
                internal_attrs=DwdIconEuInternalAttrs(
                    variable_name_in_filename="w_snow",
                    keep_mantissa_bits=default_keep_mantissa_bits,
                ),
            ),
        ]
