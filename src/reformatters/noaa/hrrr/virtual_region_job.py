from collections.abc import Mapping, Sequence
from typing import ClassVar, Generic, TypeVar

import icechunk
import pandas as pd
import xarray as xr

from reformatters.common.region_job import CoordinateValue
from reformatters.common.types import Dim
from reformatters.noaa.hrrr.hrrr_config_models import NoaaHrrrDataVar
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

_HRRR_V4_VARIABLES = frozenset(
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
    }
)


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
        if var.path in _HRRR_V4_VARIABLES and coord.init_time < pd.Timestamp(
            "2020-12-02T12"
        ):
            return False
        if var.path in {
            "lightning_threat_1m",
            "lightning_threat_2m",
        } and coord.init_time < pd.Timestamp("2022-06-28T12"):
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
