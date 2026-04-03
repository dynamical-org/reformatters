import itertools
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from zarr.abc.store import Store

from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest, item
from reformatters.common.logging import get_logger
from reformatters.common.region_job import RegionJob
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
    Timedelta,
)
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
    has_hour_0_values,
    vars_available,
)
from reformatters.ecmwf.ecmwf_grib_index import get_message_byte_ranges_from_index

from .source_file_coord import (
    MARS_OPEN_DATA_CUTOVER,
    IfsEnsSourceFileCoord,
    MarsSourceFileCoord,
    OpenDataSourceFileCoord,
)

log = get_logger(__name__)


def _get_all_byte_ranges(
    idx_local_path: Path,
    resolved_vars: Sequence[EcmwfDataVar],
    coord: IfsEnsSourceFileCoord,
) -> tuple[list[int], list[int]]:
    """Get byte ranges for all data vars across all steps in the coord."""
    if isinstance(coord, MarsSourceFileCoord):
        # MARS files contain all steps; extract byte ranges per step and combine
        all_starts: list[int] = []
        all_ends: list[int] = []
        for lead_time in coord.lead_times:
            step = whole_hours(lead_time)
            starts, ends = get_message_byte_ranges_from_index(
                idx_local_path, resolved_vars, coord.ensemble_member, step=step
            )
            all_starts.extend(starts)
            all_ends.extend(ends)
        return all_starts, all_ends
    else:
        return get_message_byte_ranges_from_index(
            idx_local_path, resolved_vars, coord.ensemble_member
        )


def _find_bands_by_forecast_seconds(
    reader: rasterio.DatasetReader, lead_times: tuple[Timedelta, ...]
) -> list[int]:
    """Find rasterio band indexes (1-indexed) matching the requested lead times."""
    target_seconds = {int(pd.Timedelta(lt).total_seconds()): lt for lt in lead_times}
    band_map: dict[int, int] = {}
    for band_idx in range(1, reader.count + 1):
        forecast_seconds = int(reader.tags(band_idx)["GRIB_FORECAST_SECONDS"])
        if forecast_seconds in target_seconds:
            band_map[forecast_seconds] = band_idx

    assert set(band_map.keys()) == set(target_seconds.keys()), (
        f"Could not find bands for all lead times. "
        f"Wanted seconds {sorted(target_seconds.keys())}, found {sorted(band_map.keys())}"
    )
    # Return bands ordered by lead_time
    return [band_map[int(pd.Timedelta(lt).total_seconds())] for lt in lead_times]


def _validate_grib_comment(
    actual_comment: str,
    expected_comment: str,
    var_name: str,
    *,
    unit_only: bool = False,
) -> None:
    """Validate grib comment matches expected.

    When unit_only=True, only checks the bracketed unit suffix matches (e.g. "[C]").
    This is useful for MARS GRIBs where the descriptive text differs from open data
    but the physical unit is the same.
    """
    if unit_only:
        actual_unit = actual_comment[actual_comment.rfind("[") :]
        expected_unit = expected_comment[expected_comment.rfind("[") :]
        assert actual_unit == expected_unit, (
            f"Unit mismatch: {actual_comment=} vs {expected_comment=}"
        )
    elif var_name == "categorical_precipitation_type_surface":
        # ECMWF occasionally adds new values in the reserved range.
        # Check the first 6 categories that shouldn't change.
        assert actual_comment[:100] == expected_comment[:100], (
            f"{actual_comment=} != {expected_comment=}"
        )
    else:
        assert actual_comment == expected_comment, (
            f"{actual_comment=} != {expected_comment=}"
        )


class EcmwfIfsEnsForecast15Day025DegreeRegionJob(
    RegionJob[EcmwfDataVar, IfsEnsSourceFileCoord]
):
    # Limits the number of variables downloaded together.
    # All variables are scattered throughout the grib file without any organization,
    # so it's more efficient to do separate windowed downloads & reads for each
    # variable that we can parallelize.
    max_vars_per_download_group: ClassVar[int] = 1
    max_vars_per_backfill_job: ClassVar[int] = 1

    @classmethod
    def source_groups(
        cls,
        data_vars: Sequence[EcmwfDataVar],
    ) -> Sequence[Sequence[EcmwfDataVar]]:
        """Return groups of variables, where all variables in a group can be retrieved from the same source file."""
        vars_by_key: defaultdict[tuple[object, bool], list[EcmwfDataVar]] = defaultdict(
            list
        )
        for data_var in data_vars:
            key = (data_var.internal_attrs.date_available, has_hour_0_values(data_var))
            vars_by_key[key].append(data_var)
        return list(vars_by_key.values())

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfDataVar],
    ) -> Sequence[IfsEnsSourceFileCoord]:
        """Returns a sequence of coords, one for each source file required to process the data covered by processing_region_ds.

        For open data (>= MARS_OPEN_DATA_CUTOVER): one coord per init_time x lead_time x ensemble_member.
        For MARS archive: one coord per init_time x request_type x ensemble_member (all lead_times grouped).
        """
        coords: list[IfsEnsSourceFileCoord] = []
        group_has_hour_0_values = item({has_hour_0_values(v) for v in data_var_group})
        for init_time, ensemble_member in itertools.product(
            processing_region_ds["init_time"].values,
            processing_region_ds["ensemble_member"].values,
        ):
            if not vars_available(data_var_group, init_time):
                continue

            lead_times = processing_region_ds["lead_time"].values
            if not group_has_hour_0_values:
                lead_times = lead_times[lead_times != np.timedelta64(0)]

            if pd.Timestamp(init_time) < MARS_OPEN_DATA_CUTOVER:
                # Group vars by MARS request type (one file per request type)
                vars_by_request_type: defaultdict[str, list[EcmwfDataVar]] = (
                    defaultdict(list)
                )
                for v in data_var_group:
                    rt = MarsSourceFileCoord.get_request_type(
                        v.internal_attrs.grib_index_level_type, int(ensemble_member)
                    )
                    vars_by_request_type[rt].append(v)
                for request_type, rt_vars in vars_by_request_type.items():
                    coords.append(
                        MarsSourceFileCoord(
                            init_time=init_time,
                            ensemble_member=int(ensemble_member),
                            data_var_group=rt_vars,
                            request_type=request_type,
                            lead_times=tuple(lead_times),
                        )
                    )
            else:
                coords.extend(
                    OpenDataSourceFileCoord(
                        init_time=init_time,
                        lead_time=lead_time,
                        data_var_group=data_var_group,
                        ensemble_member=int(ensemble_member),
                    )
                    for lead_time in lead_times
                )
        return coords

    def download_file(self, coord: IfsEnsSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        idx_local_path = http_download_to_disk(coord.get_index_url(), self.dataset_id)

        # Resolve data vars with source-appropriate attrs
        resolved_vars = [coord.resolve_data_var(v) for v in coord.data_var_group]

        byte_range_starts, byte_range_ends = _get_all_byte_ranges(
            idx_local_path, resolved_vars, coord
        )
        suffix = digest(
            f"{s}-{e}" for s, e in zip(byte_range_starts, byte_range_ends, strict=True)
        )
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(byte_range_starts, byte_range_ends),
            local_path_suffix=f"-{suffix}",
        )

    def read_data(
        self,
        coord: IfsEnsSourceFileCoord,
        data_var: EcmwfDataVar,
    ) -> ArrayFloat32:
        """Read and return an array of data for the given variable and source file coordinate."""
        resolved = coord.resolve_data_var(data_var)
        expected_spatial_shape = (721, 1440)

        with rasterio.open(coord.downloaded_path) as reader:
            if isinstance(coord, MarsSourceFileCoord):
                band_indexes = _find_bands_by_forecast_seconds(reader, coord.lead_times)
                # MARS GRIBs use different descriptive text than open data (e.g.
                # "2 metre temperature" vs "Temperature") but the physical unit
                # in brackets should match.
                _validate_grib_comment(
                    reader.tags(band_indexes[0])["GRIB_COMMENT"],
                    resolved.internal_attrs.grib_comment,
                    data_var.name,
                    unit_only=True,
                )
                result: ArrayFloat32 = reader.read(band_indexes, out_dtype=np.float32)
                assert result.shape == (len(band_indexes), *expected_spatial_shape), (
                    f"Expected {(len(band_indexes), *expected_spatial_shape)} shape, found {result.shape}"
                )
            else:
                assert reader.count == 1, f"Expected 1 band, found {reader.count}"
                _validate_grib_comment(
                    reader.tags(1)["GRIB_COMMENT"],
                    resolved.internal_attrs.grib_comment,
                    data_var.name,
                )
                assert (
                    reader.descriptions[0] == resolved.internal_attrs.grib_description
                ), (
                    f"{reader.descriptions[0]=} != {resolved.internal_attrs.grib_description}"
                )
                result = reader.read(1, out_dtype=np.float32)
                assert result.shape == expected_spatial_shape, (
                    f"Expected {expected_spatial_shape} shape, found {result.shape}"
                )

            # Apply MARS-specific scale factor (e.g. geopotential → geopotential height)
            if (
                data_var.internal_attrs.mars is not None
                and data_var.internal_attrs.mars.scale_factor is not None
            ):
                result = result * data_var.internal_attrs.mars.scale_factor

            return result

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: EcmwfDataVar
    ) -> None:
        """
        Apply in-place data transformations to the output data array for a given data variable.
        Deaccumulates precipitation to rates.

        Parameters
        ----------
        data_array : xr.DataArray
            The output data array to be transformed in-place.
        data_var : EcmwfDataVar
            The data variable metadata object, which may contain transformation parameters.
        """
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
        Sequence["RegionJob[EcmwfDataVar, IfsEnsSourceFileCoord]"],
        xr.Dataset,
    ]:
        """
        Return the sequence of RegionJob instances necessary to update the dataset
        from its current state to include the latest available data.

        Also return the template_ds, expanded along append_dim through the end of
        the data to process.

        Parameters
        ----------
        primary_store : Store
            The primary store to read existing data from and write updates to.
        tmp_store : Path
            The temporary Zarr store to write into while processing.
        get_template_fn : Callable[[DatetimeLike], xr.Dataset]
            Function to get the template_ds for the operational update.
        append_dim : AppendDim
            The dimension along which data is appended (e.g., "time").
        all_data_vars : Sequence[EcmwfDataVar]
            Sequence of all data variable configs for this dataset.
        reformat_job_name : str
            The name of the reformatting job, used for progress tracking.
            This is often the name of the Kubernetes job, or "local".

        Returns
        -------
        Sequence[RegionJob[EcmwfDataVar, IfsEnsSourceFileCoord]]
            RegionJob instances that need processing for operational updates.
        xr.Dataset
            The template_ds for the operational update.
        """
        existing_ds = xr.open_zarr(primary_store, chunks=None)
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = pd.Timestamp.now()
        template_ds = get_template_fn(append_dim_end)

        jobs = cls.get_jobs(
            kind="operational-update",
            tmp_store=tmp_store,
            template_ds=template_ds,
            append_dim=append_dim,
            all_data_vars=all_data_vars,
            reformat_job_name=reformat_job_name,
            filter_start=append_dim_start,
        )
        return jobs, template_ds
