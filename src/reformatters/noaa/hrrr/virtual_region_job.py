from collections.abc import Mapping, Sequence
from typing import ClassVar, Generic, TypeVar

import icechunk
import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim
from reformatters.noaa.hrrr.hrrr_config_models import (
    HRRR_V2_FIRST_INIT,
    HRRR_V3_FIRST_INIT,
    HRRR_V4_FIRST_INIT,
    NoaaHrrrDataVar,
)
from reformatters.noaa.hrrr.region_job import (
    NODD_BUCKET,
    NODD_BUCKET_REGION,
    NODD_HTTPS_PREFIX,
    DownloadSource,
    NoaaHrrrSourceFileCoord,
)
from reformatters.noaa.noaa_virtual_region_job import (
    NoaaVirtualRegionJob,
    NoaaVirtualSourceFileCoord,
)

S3_LOCATION_PREFIX = f"s3://{NODD_BUCKET}/"

_FIRST_EXPECTED_INIT = {
    path: first_init
    for first_init, paths in (
        (
            HRRR_V2_FIRST_INIT,
            {
                "model_level/fraction_of_cloud_cover",
                "model_level/number_concentration_cloud_droplets",
                "model_level/number_concentration_cloud_ice",
                "model_level/number_concentration_ice_friendly_aerosol",
                "model_level/number_concentration_rain",
                "model_level/number_concentration_water_friendly_aerosol",
                "baseflow_groundwater_runoff_surface",
                "brightness_temperature_channel_113",
                "brightness_temperature_channel_114",
                "brightness_temperature_channel_123",
                "brightness_temperature_channel_124",
                "derived_radar_reflectivity_263k",
                "downward_long_wave_radiation_flux_surface",
                "friction_velocity_surface",
                "frozen_precipitation_run_total_surface",
                "frozen_precipitation_surface",
                "geopotential_height_0c_isotherm",
                "geopotential_height_253k",
                "geopotential_height_263k",
                "geopotential_height_highest_tropospheric_freezing_level",
                "ground_heat_flux_surface",
                "latent_heat_flux_surface",
                "lightning_atmosphere",
                "plant_canopy_surface_water_surface",
                "potential_temperature_2m",
                "pressure_0c_isotherm",
                "pressure_highest_tropospheric_freezing_level",
                "relative_humidity_0c_isotherm",
                "relative_humidity_2m",
                "relative_humidity_highest_tropospheric_freezing_level",
                "relative_humidity_with_respect_to_precipitable_water_atmosphere",
                "sensible_heat_flux_surface",
                "storm_surface_runoff_surface",
                "surface_roughness_surface",
                "total_snowfall_run_total_surface",
                "upward_long_wave_radiation_flux_surface",
                "upward_short_wave_radiation_flux_surface",
                "visible_beam_downward_solar_flux_surface",
                "visible_diffuse_downward_solar_flux_surface",
            },
        ),
        (
            HRRR_V3_FIRST_INIT,
            {
                "freezing_rain_run_total_surface",
                "maximum_derived_radar_reflectivity_263k",
                "maximum_downward_vertical_velocity_100_1000mb",
                "maximum_hail_diameter_0p1sigma",
                "maximum_hail_diameter_atmosphere",
                "maximum_relative_vorticity_1000_0m",
                "maximum_relative_vorticity_2000_0m",
                "maximum_updraft_helicity_2000_0m",
                "maximum_updraft_helicity_3000_0m",
                "maximum_upward_vertical_velocity_100_1000mb",
                "maximum_wind_u_component_10m",
                "maximum_wind_v_component_10m",
                "minimum_updraft_helicity_2000_0m",
                "minimum_updraft_helicity_3000_0m",
                "minimum_updraft_helicity_5000_2000m",
                "upward_short_wave_radiation_flux_top_of_atmosphere",
            },
        ),
        (
            HRRR_V4_FIRST_INIT,
            {
                "maximum_hail_diameter_surface",
                "aerosol_optical_thickness_atmosphere",
                "mass_density_8m",
                "vegetation_surface",
                "minimum_vegetation_surface",
                "maximum_vegetation_surface",
                "leaf_area_index_surface",
                "column_integrated_mass_density_atmosphere",
                "total_column_cloud_water_atmosphere",
                "total_column_cloud_ice_atmosphere",
                "total_cloud_cover_boundary_layer",
                "convective_available_potential_energy_0_3000m",
                "effective_layer_helicity_surface",
                "critical_angle_0_500m",
                "layer_thickness_261k_256k",
                "enhanced_stretching_potential_0_3000m",
                "model_level/mass_density",
            },
        ),
        (pd.Timestamp("2022-06-28T12"), {"lightning_threat_1m", "lightning_threat_2m"}),
        # Earlier LFC messages alternate between unmatched "no_level" and "reserved".
        (
            pd.Timestamp("2022-06-28T00"),
            {"geopotential_height_level_of_free_convection"},
        ),
    )
    for path in paths
}


def hrrr_virtual_chunk_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    """Fresh container objects per call; icechunk containers can't be shared
    pydantic defaults."""
    return (
        icechunk.VirtualChunkContainer(
            S3_LOCATION_PREFIX, icechunk.s3_store(region=NODD_BUCKET_REGION)
        ),
    )


class NoaaHrrrVirtualSourceFileCoord(
    NoaaHrrrSourceFileCoord, NoaaVirtualSourceFileCoord[NoaaHrrrDataVar]
):
    """One HRRR product file (init_time, lead_time, file_type) and the vars it packs."""

    def get_url(self, source: DownloadSource = "s3") -> str:
        """The s3:// source location refs point at, matching the virtual chunk container."""
        url = super().get_url(source=source)
        assert url.startswith(NODD_HTTPS_PREFIX), url
        return S3_LOCATION_PREFIX + url.removeprefix(NODD_HTTPS_PREFIX)


HRRR_VIRTUAL_COORD = TypeVar("HRRR_VIRTUAL_COORD", bound=NoaaHrrrVirtualSourceFileCoord)


class NoaaHrrrVirtualRegionJob(
    NoaaVirtualRegionJob[NoaaHrrrDataVar, HRRR_VIRTUAL_COORD],
    Generic[HRRR_VIRTUAL_COORD],
):
    """The HRRR NODD source bucket."""

    source_location_prefix: ClassVar[str] = S3_LOCATION_PREFIX
    source_bucket_region: ClassVar[str] = NODD_BUCKET_REGION

    def expects_index_message(
        self, coord: HRRR_VIRTUAL_COORD, var: NoaaHrrrDataVar, level: str
    ) -> bool:
        first_init = _FIRST_EXPECTED_INIT.get(var.path)
        if first_init is not None and coord.init_time < first_init:
            return False
        if (
            var.path == "model_level/turbulent_kinetic_energy"
            and level == "50 hybrid level"
            and coord.init_time < HRRR_V2_FIRST_INIT
        ):
            return False
        return super().expects_index_message(coord, var, level)


class NoaaHrrrForecastVirtualSourceFileCoord(NoaaHrrrVirtualSourceFileCoord):
    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class NoaaHrrrForecastVirtualRegionJob(
    NoaaHrrrVirtualRegionJob[NoaaHrrrForecastVirtualSourceFileCoord]
):
    """RegionJob shared by the HRRR virtual forecast datasets; a forecast-length
    subclass declares operational_update_window."""

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[NoaaHrrrDataVar],
    ) -> Sequence[NoaaHrrrForecastVirtualSourceFileCoord]:
        init_times = pd.to_datetime(processing_region_ds["init_time"].values)
        lead_times = pd.to_timedelta(processing_region_ds["lead_time"].values)
        file_types = sorted({v.internal_attrs.hrrr_file_type for v in data_var_group})

        coords = []
        for init_time in init_times:
            for lead_time in lead_times:
                for file_type in file_types:
                    # Accumulated/categorical vars have no valid hour-0 data, so drop
                    # them at lead 0 (keeps the completeness validator consistent).
                    vars_in_file = [
                        var
                        for var in data_var_group
                        if var.internal_attrs.hrrr_file_type == file_type
                        and (lead_time > pd.Timedelta(0) or var.has_hour_0_values())
                    ]
                    if not vars_in_file:
                        continue
                    coords.append(
                        NoaaHrrrForecastVirtualSourceFileCoord(
                            init_time=init_time,
                            lead_time=lead_time,
                            domain="conus",
                            file_type=file_type,
                            data_vars=vars_in_file,
                        )
                    )
        return coords
