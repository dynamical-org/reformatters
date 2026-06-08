import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, assert_never

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest, group_by
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValue,
    MaterializedRegionJob,
    RegionJob,
    SourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Dim,
    Timedelta,
    Timestamp,
)
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar, vars_available
from reformatters.ecmwf.ecmwf_grib_index import grib_message_byte_ranges_from_index
from reformatters.ecmwf.ecmwf_utils import (
    EcmwfOpenDataSource,
    ecmwf_download_with_fallback,
)

log = get_logger(__name__)

# aifs-ens files land on the GCS mirror ~40 min after S3. For recent forecasts
# prefer S3 (freshness); for older forecasts prefer GCS (better availability under
# S3 throttling). See https://forum.ecmwf.int/t/503-slowdown-errors-for-requests-to-s3-ecmwf-forecasts/14345
_RECENT_CUTOFF = pd.Timedelta(hours=24)

# GRIB master table version changes caused metadata differences for precipitation.
# Early data (table v27): generic product template codes.
# Recent data (table v34+): specific parameter names with different step encoding.
# Maps grib_index_param -> (alt_grib_comment, alt_grib_description)
_PRECIP_ALT_GRIB_METADATA: dict[str, tuple[str, str]] = {
    "tp": (
        "Total precipitation rate [kg/(m^2*s)]",
        '0[-] SFC="Ground or water surface"',
    ),
}


class EcmwfAifsEnsForecastSourceFileCoord(SourceFileCoord):
    init_time: Timestamp
    lead_time: Timedelta
    ensemble_member: int
    data_var_group: Sequence[EcmwfDataVar]

    @property
    def file_type(self) -> str:
        # ensemble_member 0 is the control forecast (cf), 1-50 are perturbed members (pf).
        return "cf" if self.ensemble_member == 0 else "pf"

    def _get_base_url(self, source: EcmwfOpenDataSource) -> str:
        match source:
            case "s3":
                root_url = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
            case "gcs":
                root_url = "https://storage.googleapis.com/ecmwf-open-data"
            case _ as unreachable:
                assert_never(unreachable)

        init_time_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")
        lead_time_hour_str = whole_hours(self.lead_time)

        directory_path = f"{init_time_str}/{init_hour_str}z/aifs-ens/0p25/enfo"
        filename = f"{init_time_str}{init_hour_str}0000-{lead_time_hour_str}h-enfo-{self.file_type}"
        return f"{root_url}/{directory_path}/{filename}"

    def get_url(self, source: EcmwfOpenDataSource = "s3") -> str:
        return self._get_base_url(source) + ".grib2"

    def get_index_url(self, source: EcmwfOpenDataSource = "s3") -> str:
        return self._get_base_url(source) + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


class EcmwfAifsEnsForecastRegionJob(
    MaterializedRegionJob[EcmwfDataVar, EcmwfAifsEnsForecastSourceFileCoord]
):
    max_vars_per_download_group: ClassVar[int] = 2
    max_vars_per_job: ClassVar[int] = 4

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[EcmwfDataVar],
    ) -> Sequence[Sequence[EcmwfDataVar]]:
        return group_by(data_vars, lambda v: v.internal_attrs.date_available)

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfDataVar],
    ) -> Sequence[EcmwfAifsEnsForecastSourceFileCoord]:
        coords = []
        for init_time, lead_time, ensemble_member in itertools.product(
            processing_region_ds["init_time"].values,
            processing_region_ds["lead_time"].values,
            processing_region_ds["ensemble_member"].values,
        ):
            if not vars_available(data_var_group, init_time):
                continue

            coords.append(
                EcmwfAifsEnsForecastSourceFileCoord(
                    init_time=init_time,
                    lead_time=lead_time,
                    ensemble_member=int(ensemble_member),
                    data_var_group=data_var_group,
                )
            )
        return coords

    def download_file(self, coord: EcmwfAifsEnsForecastSourceFileCoord) -> Path:
        if coord.init_time >= pd.Timestamp.now() - _RECENT_CUTOFF:
            sources: tuple[EcmwfOpenDataSource, ...] = ("s3", "gcs")
        else:
            sources = ("gcs", "s3")
        return ecmwf_download_with_fallback(
            sources, lambda source: self._download_from_source(coord, source)
        )

    def _download_from_source(
        self,
        coord: EcmwfAifsEnsForecastSourceFileCoord,
        source: EcmwfOpenDataSource,
    ) -> Path:
        idx_local_path = http_download_to_disk(
            coord.get_index_url(source), self.dataset_id, disk_cache=True
        )

        # cf files have no "number" column (single member); pf files have a "number" column.
        index_ensemble_member = (
            None if coord.file_type == "cf" else coord.ensemble_member
        )
        byte_range_starts, byte_range_ends = grib_message_byte_ranges_from_index(
            idx_local_path,
            coord.data_var_group,
            ensemble_member=index_ensemble_member,
        )
        suffix = digest(
            f"{s}-{e}" for s, e in zip(byte_range_starts, byte_range_ends, strict=True)
        )
        return http_download_to_disk(
            coord.get_url(source),
            self.dataset_id,
            byte_ranges=(byte_range_starts, byte_range_ends),
            local_path_suffix=f"-{suffix}",
        )

    def read_data(
        self,
        coord: EcmwfAifsEnsForecastSourceFileCoord,
        data_var: EcmwfDataVar,
    ) -> ArrayFloat32:
        expected_comment = data_var.internal_attrs.grib_comment
        expected_description = data_var.internal_attrs.grib_description

        alt_metadata = _PRECIP_ALT_GRIB_METADATA.get(
            data_var.internal_attrs.grib_index_param
        )
        allowed_comments = {expected_comment}
        allowed_descriptions = {expected_description}
        if alt_metadata is not None:
            allowed_comments.add(alt_metadata[0])
            allowed_descriptions.add(alt_metadata[1])

        with rasterio.open(coord.downloaded_path) as reader:
            matching_bands: list[int] = []
            for band_i in range(reader.count):
                rasterio_band_i = band_i + 1
                if (
                    reader.tags(rasterio_band_i)["GRIB_COMMENT"] in allowed_comments
                    and reader.descriptions[band_i] in allowed_descriptions
                ):
                    matching_bands.append(rasterio_band_i)

            assert len(matching_bands) == 1, (
                f"Expected exactly 1 matching band, found {len(matching_bands)}. "
                f"{expected_comment=}, {expected_description=}, {coord.downloaded_path=}"
            )
            result: ArrayFloat32 = reader.read(matching_bands[0], out_dtype=np.float32)
            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: EcmwfDataVar
    ) -> None:
        if data_var.internal_attrs.scale_factor is not None:
            data_array *= data_var.internal_attrs.scale_factor

        if data_var.internal_attrs.deaccumulate_to_rate:
            reset_freq = data_var.internal_attrs.window_reset_frequency
            deaccumulation_invalid_below_threshold_rate = (
                data_var.internal_attrs.deaccumulation_invalid_below_threshold_rate
            )
            assert deaccumulation_invalid_below_threshold_rate is not None
            assert reset_freq is not None

            try:
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="lead_time",
                    reset_frequency=reset_freq,
                    invalid_below_threshold_rate=deaccumulation_invalid_below_threshold_rate,
                    # Short wave radiation sees 5-7% clamped due to lossy grib2 compression
                    expected_clamp_fraction=0.08,
                )
            except ValueError:
                log.exception(f"Error deaccumulating {data_var.name}")

        super().apply_data_transformations(data_array, data_var)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.Dataset],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[EcmwfDataVar, EcmwfAifsEnsForecastSourceFileCoord]],
        xr.Dataset,
    ]:
        existing_ds = xr.open_zarr(primary_store, chunks=None)
        append_dim_start = existing_ds[append_dim].max()
        # Subtracting 5.5h here keeps a not-yet-published
        # cycle out of scope so pods don't waste time trying files that won't exist.
        append_dim_end = pd.Timestamp.now() - pd.Timedelta(hours=5.5)
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds
