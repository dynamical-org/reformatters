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
from reformatters.common.types import (
    AppendDim,
    ArrayFloat32,
    DatetimeLike,
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
        """Returns a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        coords: list[IfsEnsSourceFileCoord] = []
        group_has_hour_0_values = item({has_hour_0_values(v) for v in data_var_group})
        for init_time, lead_time, ensemble_member in itertools.product(
            processing_region_ds["init_time"].values,
            processing_region_ds["lead_time"].values,
            processing_region_ds["ensemble_member"].values,
        ):
            if not vars_available(data_var_group, init_time):
                continue

            if not group_has_hour_0_values and lead_time == np.timedelta64(0):
                continue

            member = int(ensemble_member)
            if init_time < MARS_OPEN_DATA_CUTOVER:
                # max_vars_per_download_group=1 ensures all vars share a level type
                levtype = item(
                    {v.internal_attrs.grib_index_level_type for v in data_var_group}
                )
                coords.append(
                    MarsSourceFileCoord(
                        init_time=init_time,
                        lead_time=lead_time,
                        ensemble_member=member,
                        data_var_group=data_var_group,
                        request_type=MarsSourceFileCoord.get_request_type(
                            levtype, member
                        ),
                    ).resolve_data_vars()
                )
            else:
                coords.append(
                    OpenDataSourceFileCoord(
                        init_time=init_time,
                        lead_time=lead_time,
                        data_var_group=data_var_group,
                        ensemble_member=member,
                    ).resolve_data_vars()
                )
        return coords

    def download_file(self, coord: IfsEnsSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path."""
        idx_local_path = http_download_to_disk(coord.get_index_url(), self.dataset_id)

        byte_range_starts, byte_range_ends = get_message_byte_ranges_from_index(
            idx_local_path,
            coord.data_var_group,
            coord.ensemble_member,
            step=coord.index_step,
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
        # Resolved var has source-appropriate grib metadata (e.g. MARS param/comment overrides)
        resolved_data_var = item(
            v for v in coord.data_var_group if v.name == data_var.name
        )
        expected_shape = (721, 1440)

        with rasterio.open(coord.downloaded_path) as reader:
            assert reader.count == 1, f"Expected 1 band, found {reader.count}"
            _validate_grib_metadata(
                reader,
                resolved_data_var.internal_attrs.grib_comment,
                resolved_data_var.internal_attrs.grib_description,
                resolved_data_var.internal_attrs.grib_element,
                data_var.name,
                unit_only=coord.validate_grib_comment_unit_only,
            )
            result: ArrayFloat32 = reader.read(1, out_dtype=np.float32)
            assert result.shape == expected_shape, (
                f"Expected {expected_shape} shape, found {result.shape}"
            )

            # Apply MARS-specific scale factor (e.g. geopotential m^2/s^2 to
            # geopotential height gpm). Applied here rather than in
            # apply_data_transformations because a shard could mix sources and
            # the conversion must only apply to MARS-sourced values.
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
                    expected_invalid_fraction=0.01,
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


def _validate_grib_metadata(
    reader: rasterio.DatasetReader,
    expected_comment: str,
    expected_description: str,
    expected_element: str,
    var_name: str,
    *,
    unit_only: bool = False,
) -> None:
    """Validate GRIB metadata for the first band matches expected values.

    When unit_only=True, only checks the bracketed unit suffix of the comment
    (e.g. "[C]") and skips description validation. This is useful for MARS GRIBs
    where the descriptive text differs from open data but the physical unit matches.

    GRIB_ELEMENT is always validated as it's the most reliable identifier across sources.
    """
    tags = reader.tags(1)

    actual_element = tags["GRIB_ELEMENT"]
    assert actual_element == expected_element, (
        f"Element mismatch: {actual_element=} vs {expected_element=}"
    )

    actual_comment = tags["GRIB_COMMENT"]
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

    if not unit_only:
        actual_description = reader.descriptions[0]
        assert actual_description == expected_description, (
            f"{actual_description=} != {expected_description=}"
        )
