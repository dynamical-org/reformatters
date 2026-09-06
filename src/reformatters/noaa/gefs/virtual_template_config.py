import functools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from gribberish.zarr import GribberishCodec
from pydantic import computed_field
from zarr.codecs import ScaleOffset

from reformatters.common.config_models import (
    ROOT,
    Coordinate,
    DatasetAttributes,
    DataVarAttrs,
    Encoding,
    Group,
    SpatialResolution,
)
from reformatters.common.iterating import item
from reformatters.common.template_config import TemplateConfig
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import CodecConfig, Dim, Timestamp
from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_coordinate_configs,
    get_shared_template_dimension_coordinates,
)
from reformatters.noaa.gefs.gefs_config_models import (
    FILE_RESOLUTIONS,
    GEFS_ACCUMULATION_RESET_FREQUENCY,
    GEFS_B22_TRANSITION_DATE,
    GEFSSourceFileType,
    NoaaGefsVirtualDataVar,
    NoaaGefsVirtualInternalAttrs,
)

# The inits between the v12 configuration change and this date are ragged.
GEFS_VIRTUAL_ARCHIVE_START = pd.Timestamp("2020-10-01T00:00")

# The catalog's spelling of each grid FILE_RESOLUTIONS resolves to.
_SPATIAL_RESOLUTIONS: dict[float, SpatialResolution] = {0.25: "0.25 degrees (~20km)"}

# GribberishCodec decodes the raw Kelvin message and this array->array filter subtracts
# 273.15 on read. ScaleOffset decodes as value / scale + offset.
_KELVIN_TO_CELSIUS = ScaleOffset(offset=-273.15, scale=1.0).to_dict()
# TSOIL carries this filter at its own definition instead.
_CELSIUS_ELEMENTS = frozenset({"TMP", "DPT", "TMAX", "TMIN"})

# WEASD decodes as kg m-2 of water; 1 kg m-2 = 0.001 m lwe, so scale=1000 yields the
# metres CF gives lwe_thickness_of_surface_snow_amount.
_WATER_KG_M2_TO_M_LWE = ScaleOffset(offset=0.0, scale=1000.0).to_dict()

# MSLET entered the s file at this cycle; CPOFP, HGT@cloud ceiling and VIS entered at
# GEFS_B22_TRANSITION_DATE.
MSLET_AVAILABLE_FROM = pd.Timestamp("2021-07-20T12:00")


class NoaaGefsVirtualTemplateConfig(TemplateConfig[NoaaGefsVirtualDataVar]):
    """Virtual GEFS template: one chunk per GRIB message on the source's native
    latitude/longitude grid.

    `source_file_types` selects both the grid and the variable catalog. A subclass
    declares dims, time structure and the window wording its time axis implies.
    """

    source_file_types: frozenset[GEFSSourceFileType]
    # Keyed by step_type, applied to every windowed variable. The window a value covers
    # depends on the dataset's time structure.
    window_comments: dict[str, str]

    @property
    def resolution_degrees(self) -> float:
        """The one grid spacing `source_file_types` resolves to, in degrees."""
        resolution = item(
            {FILE_RESOLUTIONS[file_type] for file_type in self.source_file_types}
        )
        return float(resolution.replace("p", "."))

    def _spatial_dimension_coordinates(self) -> dict[str, Any]:
        return get_shared_template_dimension_coordinates(self.resolution_degrees)

    def _spatial_coords(self) -> Sequence[Coordinate]:
        return get_shared_coordinate_configs(self.resolution_degrees)

    def _dataset_attributes(
        self, *, dataset_id: str, dataset_version: str, name: str, description: str
    ) -> DatasetAttributes:
        return DatasetAttributes(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            name=name,
            description=description,
            attribution="NOAA NWS NCEP GEFS data processed by dynamical.org from NOAA Open Data Dissemination archives.",
            license="CC-BY-4.0",
            spatial_domain="Global",
            spatial_resolution=_SPATIAL_RESOLUTIONS[self.resolution_degrees],
            time_domain=f"{self.append_dim_start} UTC to Present",
            time_resolution=f"{whole_hours(self.append_dim_frequency)} hours",
        )

    @computed_field
    @property
    def data_vars(self) -> Sequence[NoaaGefsVirtualDataVar]:
        return self._catalog_data_vars()

    def _catalog_data_vars(self) -> list[NoaaGefsVirtualDataVar]:
        """The variable catalog for the source files this config declares."""
        assert self.source_file_types == frozenset({"s"}), (
            "only the s file catalog is populated; the a and b catalogs land with the "
            "0.5 degree datasets"
        )
        return _s_file_data_vars(self._message_chunks(ROOT), self.window_comments)

    def _message_chunks(self, group: Group) -> tuple[int, ...]:
        """One chunk per GRIB message: the full grid, size 1 along every other dim."""
        dims = self.dims[group]
        assert {"latitude", "longitude"} <= set(dims), (
            f"{group} dims {dims} do not span the grid, so a chunk would not hold one "
            "whole GRIB message"
        )
        dim_coords = self._spatial_dimension_coordinates()
        sizes: dict[Dim, int] = {
            "latitude": len(dim_coords["latitude"]),
            "longitude": len(dim_coords["longitude"]),
        }
        return tuple(sizes.get(dim, 1) for dim in dims)


def _virtual_encoding(
    element: str, chunks: tuple[int, ...], filters: Sequence[CodecConfig]
) -> Encoding:
    """No shards, no compressors; GribberishCodec decodes the raw message and any
    array->array filters (K->C, unit scaling) are chained on read."""
    return Encoding(
        # GribberishCodec decodes to float64 natively; declaring float64 avoids a cast.
        dtype="float64",
        fill_value=np.nan,
        chunks=chunks,
        shards=None,
        compressors=(),
        filters=filters,
        serializer=GribberishCodec(
            var=element, adjust_longitude_range=True, north_up=True
        ).to_dict(),
    )


def _data_var(
    name: str,
    *,
    chunks: tuple[int, ...],
    window_comments: Mapping[str, str],
    element: str,
    level: str,
    source_file_type: GEFSSourceFileType,
    step_type: str = "instant",
    short_name: str,
    long_name: str,
    units: str,
    standard_name: str | None = None,
    comment: str | None = None,
    filters: Sequence[CodecConfig] | None = None,
    flag_values: tuple[int, ...] | None = None,
    flag_meanings: str | None = None,
    available_from: Timestamp | None = None,
) -> NoaaGefsVirtualDataVar:
    resolved_filters: Sequence[CodecConfig] = (
        filters
        if filters is not None
        else ([_KELVIN_TO_CELSIUS] if element in _CELSIUS_ELEMENTS else ())
    )
    # A flag variable's values are codes, not an average, so window wording would
    # contradict its flag_values.
    window_comment = None if flag_values else window_comments.get(step_type)
    if window_comment is not None:
        comment = f"{window_comment} {comment}" if comment else window_comment
    return NoaaGefsVirtualDataVar(
        name=name,
        encoding=_virtual_encoding(element, chunks, resolved_filters),
        attrs=DataVarAttrs(
            short_name=short_name,
            long_name=long_name,
            units=units,
            standard_name=standard_name,
            step_type=step_type,  # ty: ignore[invalid-argument-type]
            comment=comment,
            flag_values=flag_values,
            flag_meanings=flag_meanings,
        ),
        internal_attrs=NoaaGefsVirtualInternalAttrs(
            grib_element=element,
            grib_index_level=level,
            source_file_type=source_file_type,
            available_from=available_from,
            window_reset_frequency=(
                GEFS_ACCUMULATION_RESET_FREQUENCY if step_type != "instant" else None
            ),
            # Virtual chunks are never rewritten, so no rounding and no rasterio band
            # description / index position (unused fields the base model requires).
            keep_mantissa_bits="no-rounding",
            grib_description="",
            index_position=0,
        ),
    )


def _s_file_data_vars(
    chunks: tuple[int, ...], window_comments: Mapping[str, str]
) -> list[NoaaGefsVirtualDataVar]:
    """Every message the pgrb2s.0p25 file publishes, except HGT at the surface.

    The s file carries surface geopotential height only at lead 0, unlike every other
    message in it.
    """
    var = functools.partial(
        _data_var,
        chunks=chunks,
        source_file_type="s",
        window_comments=window_comments,
    )
    return [
        var(
            "visibility_surface",
            element="VIS",
            level="surface",
            short_name="vis",
            long_name="Visibility",
            units="m",
            standard_name="visibility_in_air",
            available_from=GEFS_B22_TRANSITION_DATE,
        ),
        var(
            "wind_gust_surface",
            element="GUST",
            level="surface",
            short_name="gust",
            long_name="Wind speed (gust)",
            units="m s-1",
            standard_name="wind_speed_of_gust",
        ),
        var(
            "pressure_reduced_to_mean_sea_level_eta_model",
            element="MSLET",
            level="mean sea level",
            short_name="mslet",
            long_name="MSLP (Eta model reduction)",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
            comment="Reduced to mean sea level by the Eta model method, unlike pressure_reduced_to_mean_sea_level in this dataset.",
            available_from=MSLET_AVAILABLE_FROM,
        ),
        var(
            "pressure_surface",
            element="PRES",
            level="surface",
            short_name="sp",
            long_name="Surface pressure",
            units="Pa",
            standard_name="surface_air_pressure",
        ),
        var(
            "soil_temperature_0_10cm",
            element="TSOIL",
            level="0-0.1 m below ground",
            short_name="sot",
            long_name="Soil temperature",
            units="degree_Celsius",
            standard_name="soil_temperature",
            comment="NaN over water, where there is no soil.",
            # The source is Kelvin despite GDAL labelling this element [C].
            filters=[_KELVIN_TO_CELSIUS],
        ),
        var(
            "volumetric_soil_moisture_0_10cm",
            element="SOILW",
            level="0-0.1 m below ground",
            short_name="vsw",
            long_name="Volumetric soil moisture",
            units="1",
            standard_name="volume_fraction_of_condensed_water_in_soil",
            comment="NaN over water, where there is no soil.",
        ),
        var(
            "snow_water_equivalent_surface",
            element="WEASD",
            level="surface",
            short_name="sd",
            long_name="Snow depth water equivalent",
            units="m",
            standard_name="lwe_thickness_of_surface_snow_amount",
            comment="NaN over open water, where snow does not accumulate.",
            filters=[_WATER_KG_M2_TO_M_LWE],
        ),
        var(
            "snow_thickness_surface",
            element="SNOD",
            level="surface",
            short_name="sde",
            long_name="Snow depth",
            units="m",
            standard_name="surface_snow_thickness",
            comment="NaN over open water, where snow does not accumulate.",
        ),
        var(
            "ice_thickness_surface",
            element="ICETK",
            level="surface",
            short_name="icetk",
            long_name="Ice thickness",
            units="m",
        ),
        var(
            "temperature_2m",
            element="TMP",
            level="2 m above ground",
            short_name="2t",
            long_name="2 metre temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        var(
            "dew_point_temperature_2m",
            element="DPT",
            level="2 m above ground",
            short_name="2d",
            long_name="2 metre dewpoint temperature",
            units="degree_Celsius",
            standard_name="dew_point_temperature",
        ),
        var(
            "relative_humidity_2m",
            element="RH",
            level="2 m above ground",
            short_name="2r",
            long_name="2 metre relative humidity",
            units="percent",
            standard_name="relative_humidity",
        ),
        var(
            "maximum_temperature_2m",
            element="TMAX",
            level="2 m above ground",
            step_type="max",
            short_name="tmax",
            long_name="Maximum temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        var(
            "minimum_temperature_2m",
            element="TMIN",
            level="2 m above ground",
            step_type="min",
            short_name="tmin",
            long_name="Minimum temperature",
            units="degree_Celsius",
            standard_name="air_temperature",
        ),
        var(
            "wind_u_10m",
            element="UGRD",
            level="10 m above ground",
            short_name="10u",
            long_name="10 metre U wind component",
            units="m s-1",
            standard_name="eastward_wind",
        ),
        var(
            "wind_v_10m",
            element="VGRD",
            level="10 m above ground",
            short_name="10v",
            long_name="10 metre V wind component",
            units="m s-1",
            standard_name="northward_wind",
        ),
        var(
            "percent_frozen_precipitation_surface",
            element="CPOFP",
            level="surface",
            short_name="cpofp",
            long_name="Percent frozen precipitation",
            units="percent",
            comment="Negative values mark no precipitation. Interpolation in the source mixes the no data value with real percentages, so unusable values span a range rather than one value and are not converted to NaN. Mask values < -0.1.",
            available_from=GEFS_B22_TRANSITION_DATE,
        ),
        var(
            "total_precipitation_surface",
            element="APCP",
            level="surface",
            step_type="accum",
            short_name="tp",
            long_name="Total precipitation",
            units="kg m-2",
            standard_name="precipitation_amount",
        ),
        var(
            "categorical_snow_surface",
            element="CSNOW",
            level="surface",
            step_type="avg",
            short_name="csnow",
            long_name="Categorical snow",
            units="1",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        var(
            "categorical_ice_pellets_surface",
            element="CICEP",
            level="surface",
            step_type="avg",
            short_name="cicep",
            long_name="Categorical ice pellets",
            units="1",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        var(
            "categorical_freezing_rain_surface",
            element="CFRZR",
            level="surface",
            step_type="avg",
            short_name="cfrzr",
            long_name="Categorical freezing rain",
            units="1",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        var(
            "categorical_rain_surface",
            element="CRAIN",
            level="surface",
            step_type="avg",
            short_name="crain",
            long_name="Categorical rain",
            units="1",
            flag_values=(0, 1),
            flag_meanings="no yes",
        ),
        var(
            "latent_heat_flux_surface",
            element="LHTFL",
            level="surface",
            step_type="avg",
            short_name="lhf",
            long_name="Latent heat flux",
            units="W m-2",
            standard_name="surface_upward_latent_heat_flux",
        ),
        var(
            "sensible_heat_flux_surface",
            element="SHTFL",
            level="surface",
            step_type="avg",
            short_name="shf",
            long_name="Sensible heat flux",
            units="W m-2",
            standard_name="surface_upward_sensible_heat_flux",
        ),
        var(
            "convective_available_potential_energy_surface",
            element="CAPE",
            level="surface",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        var(
            "convective_inhibition_surface",
            element="CIN",
            level="surface",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        var(
            "precipitable_water_atmosphere",
            element="PWAT",
            level="entire atmosphere (considered as a single layer)",
            short_name="pwat",
            long_name="Precipitable water",
            units="kg m-2",
            standard_name="atmosphere_mass_content_of_water_vapor",
        ),
        var(
            "total_cloud_cover_atmosphere",
            element="TCDC",
            level="entire atmosphere",
            step_type="avg",
            short_name="tcc",
            long_name="Total cloud cover",
            units="percent",
            standard_name="cloud_area_fraction",
        ),
        var(
            "geopotential_height_cloud_ceiling",
            element="HGT",
            level="cloud ceiling",
            short_name="gh",
            long_name="Geopotential height",
            units="m",
            standard_name="geopotential_height",
            comment="Values near 20,000m mark no cloud ceiling. Interpolation in the source mixes the no data value with real ceiling heights, so unusable values span a range rather than one value and are not converted to NaN. Mask values above 19,000m.",
            available_from=GEFS_B22_TRANSITION_DATE,
        ),
        var(
            "downward_short_wave_radiation_flux_surface",
            element="DSWRF",
            level="surface",
            step_type="avg",
            short_name="sdswrf",
            long_name="Surface downward short-wave radiation flux",
            units="W m-2",
            standard_name="surface_downwelling_shortwave_flux_in_air",
        ),
        var(
            "downward_long_wave_radiation_flux_surface",
            element="DLWRF",
            level="surface",
            step_type="avg",
            short_name="sdlwrf",
            long_name="Surface downward long-wave radiation flux",
            units="W m-2",
            standard_name="surface_downwelling_longwave_flux_in_air",
        ),
        var(
            "upward_short_wave_radiation_flux_surface",
            element="USWRF",
            level="surface",
            step_type="avg",
            short_name="suswrf",
            long_name="Surface upward short-wave radiation flux",
            units="W m-2",
            standard_name="surface_upwelling_shortwave_flux_in_air",
        ),
        var(
            "upward_long_wave_radiation_flux_surface",
            element="ULWRF",
            level="surface",
            step_type="avg",
            short_name="sulwrf",
            long_name="Surface upward long-wave radiation flux",
            units="W m-2",
            standard_name="surface_upwelling_longwave_flux_in_air",
        ),
        var(
            "upward_long_wave_radiation_flux_top_of_atmosphere",
            element="ULWRF",
            level="top of atmosphere",
            step_type="avg",
            short_name="ulwrf",
            long_name="Upward long-wave radiation flux",
            units="W m-2",
            standard_name="toa_outgoing_longwave_flux",
        ),
        var(
            "storm_relative_helicity_3000_0m",
            element="HLCY",
            level="3000-0 m above ground",
            short_name="hlcy",
            long_name="Storm relative helicity",
            units="m2 s-2",
        ),
        var(
            "convective_available_potential_energy_180_0mb",
            element="CAPE",
            level="180-0 mb above ground",
            short_name="cape",
            long_name="Convective available potential energy",
            units="J kg-1",
            standard_name="atmosphere_convective_available_potential_energy",
        ),
        var(
            "convective_inhibition_180_0mb",
            element="CIN",
            level="180-0 mb above ground",
            short_name="cin",
            long_name="Convective inhibition",
            units="J kg-1",
            standard_name="atmosphere_convective_inhibition",
        ),
        var(
            "pressure_reduced_to_mean_sea_level",
            element="PRMSL",
            level="mean sea level",
            short_name="prmsl",
            long_name="Pressure reduced to MSL",
            units="Pa",
            standard_name="air_pressure_at_mean_sea_level",
        ),
    ]
