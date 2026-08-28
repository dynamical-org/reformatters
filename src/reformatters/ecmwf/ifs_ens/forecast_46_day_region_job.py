"""Reformat ECMWF IFS ENS 46-day GRIBs from dynamical's archive into zarr.

The archive is the authoritative source: ECDS has no addressable objects, so
reading at reformat time would put its request queue into the write path. Every archived
blob carries a byte-range index written when its inventory was validated, which is what
makes a single message readable without scanning the whole blob.
"""

import functools
import itertools
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.env import Env
from zarr.abc.store import Store

from reformatters.common.config_models import mask_source_fill_value_inplace
from reformatters.common.deaccumulation import deaccumulate_to_rates_inplace
from reformatters.common.download import http_download_to_disk
from reformatters.common.iterating import digest, item
from reformatters.common.logging import get_logger
from reformatters.common.materialized_region_job import MaterializedRegionJob
from reformatters.common.region_job import (
    CoordinateValue,
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
from reformatters.ecmwf.archive_gribs.archive import format_init_time
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    ARCHIVE_BASE_URL,
    ECDS_VARIABLES,
)
from reformatters.ecmwf.archive_gribs.grib_inventory import (
    INDEX_SUFFIX,
    MessageRecord,
    read_index,
)
from reformatters.ecmwf.archive_gribs.request_shards import (
    CONTROL_MEMBER,
    PRESSURE_LEVEL_VARIABLES,
    EcdsSelection,
    initialization_selections,
)
from reformatters.ecmwf.ifs_ens.forecast_46_day_config_models import (
    EcmwfIfsEns46DayDataVar,
    SubStepReduction,
)

log = get_logger(__name__)

GRID_SHAPE = (121, 240)
# Radiation and precipitation accumulations deaccumulate to slightly negative rates
# where simple packing rounds a step down.
EXPECTED_CLAMP_FRACTION = 0.08


@lru_cache
def selections_by_variable() -> Mapping[tuple[str, str], EcdsSelection]:
    """The ECDS request each variable is archived in, keyed by (variable, forecast type).

    The grouping is over the archive's whole variable manifest. A request's file name
    identifies the group it was retrieved in, so grouping any subset names blobs that
    do not exist.
    """
    return {
        (variable, selection.forecast_type): selection
        for selection in initialization_selections(ECDS_VARIABLES)
        for variable in selection.variables
    }


class EcmwfIfsEns46DaySourceFileCoord(SourceFileCoord):
    """The messages of one archived blob that fill one (init, lead, member) slot.

    `levels` is the variable's ECDS level values in output order, `None` where the
    source has no such level and the output stays NaN. It is empty for a single-level
    variable, whose one message fills a 2D slot.

    `sub_step_lead_times` is empty unless several source messages tile this slot's
    lead time, in which case it names them and their values are reduced on read.
    """

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_member: int
    ecds_variable: str
    levels: tuple[str | None, ...]
    selection: EcdsSelection
    sub_step_lead_times: tuple[Timedelta, ...] = ()

    def get_url(self) -> str:
        return (
            f"{ARCHIVE_BASE_URL}/{format_init_time(self.init_time)}/"
            f"{self.selection.file_name}"
        )

    def get_index_url(self) -> str:
        return self.get_url() + INDEX_SUFFIX

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }

    @property
    def present_levels(self) -> tuple[str, ...]:
        return tuple(level for level in self.levels if level is not None)

    @property
    def source_lead_times(self) -> tuple[Timedelta, ...]:
        return self.sub_step_lead_times or (self.lead_time,)


class EcmwfIfsEns46DayRegionJob(
    MaterializedRegionJob[EcmwfIfsEns46DayDataVar, EcmwfIfsEns46DaySourceFileCoord]
):
    # A pressure level variable's region array is ten times a surface variable's, so
    # one variable per job keeps every job's shared memory buffer to that one array.
    max_vars_per_job: ClassVar[int] = 1

    @classmethod
    def source_file_var_groups(
        cls,
        data_vars: Sequence[EcmwfIfsEns46DayDataVar],
    ) -> Sequence[Sequence[EcmwfIfsEns46DayDataVar]]:
        """Group variables by the archived blob they were retrieved in."""
        selections = selections_by_variable()
        groups: defaultdict[str, list[EcmwfIfsEns46DayDataVar]] = defaultdict(list)
        for data_var in data_vars:
            selection = selections[
                (data_var.internal_attrs.ecds_variable, "perturbed_forecast")
            ]
            groups[selection.file_name].append(data_var)
        return list(groups.values())

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfIfsEns46DayDataVar],
    ) -> Sequence[EcmwfIfsEns46DaySourceFileCoord]:
        data_var = item(data_var_group)
        ecds_variable = data_var.internal_attrs.ecds_variable
        selections = selections_by_variable()
        levels = _output_levels(processing_region_ds, ecds_variable, data_var)

        reduction = data_var.internal_attrs.sub_step_reduction

        return [
            EcmwfIfsEns46DaySourceFileCoord(
                init_time=init_time,
                lead_time=lead_time,
                ensemble_member=int(ensemble_member),
                ecds_variable=ecds_variable,
                levels=levels,
                selection=selections[(ecds_variable, _forecast_type(ensemble_member))],
                sub_step_lead_times=_sub_step_lead_times(lead_time, reduction),
            )
            for init_time, lead_time, ensemble_member in itertools.product(
                processing_region_ds["init_time"].values,
                processing_region_ds["lead_time"].values,
                processing_region_ds["ensemble_member"].values,
            )
            if data_var.has_hour_0_values() or lead_time != np.timedelta64(0)
        ]

    def download_file(self, coord: EcmwfIfsEns46DaySourceFileCoord) -> Path:
        index_path = http_download_to_disk(
            coord.get_index_url(), self.dataset_id, disk_cache=True
        )
        starts, ends = _message_byte_ranges(index_path, coord)
        suffix = digest(
            f"{start}-{end}" for start, end in zip(starts, ends, strict=True)
        )
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(starts, ends),
            local_path_suffix=f"-{suffix}",
        )

    def read_data(
        self,
        coord: EcmwfIfsEns46DaySourceFileCoord,
        data_var: EcmwfIfsEns46DayDataVar,
    ) -> ArrayFloat32:
        # Disable GDAL's unit normalization on read. Instead,
        # apply_data_transformations uses a variable's scale_factor and add_offset.
        with (
            Env(GRIB_NORMALIZE_UNITS="NO"),
            rasterio.open(coord.downloaded_path) as reader,
        ):
            expected_messages = len(coord.source_lead_times) * len(
                coord.present_levels or (None,)
            )
            assert reader.count == expected_messages, (
                f"Expected {expected_messages} messages, "
                f"found {reader.count} in {coord.downloaded_path}"
            )
            _validate_grib_metadata(reader, data_var)
            # Byte ranges are requested in output level order, so band order is too.
            present = [
                reader.read(band, out_dtype=np.float32)
                for band in range(1, reader.count + 1)
            ]

        reduction = data_var.internal_attrs.sub_step_reduction
        if reduction is not None:
            assert not coord.levels, (
                f"{data_var.name}: reducing sub step messages is only supported "
                "for single level variables"
            )
            reduce = np.maximum if reduction.operation == "maximum" else np.minimum
            values = functools.reduce(reduce, present)
        elif not coord.levels:
            values = item(present)
        else:
            missing = np.full(GRID_SHAPE, np.nan, dtype=np.float32)
            bands = iter(present)
            values = np.stack(
                [missing if level is None else next(bands) for level in coord.levels],
                axis=0,
            )
        assert values.shape[-2:] == GRID_SHAPE, (
            f"Expected a {GRID_SHAPE} grid, found {values.shape[-2:]}"
        )
        mask_source_fill_value_inplace(values, data_var.internal_attrs)
        return values

    def apply_data_transformations(
        self, data_array: xr.DataArray, data_var: EcmwfIfsEns46DayDataVar
    ) -> None:
        internal_attrs = data_var.internal_attrs
        if internal_attrs.scale_factor is not None:
            data_array *= internal_attrs.scale_factor
        if internal_attrs.add_offset is not None:
            data_array += internal_attrs.add_offset

        if internal_attrs.deaccumulate_to_rate:
            assert internal_attrs.window_reset_frequency is not None
            if internal_attrs.deaccumulation_type == "signed":
                _deaccumulate_signed_inplace(data_array)
            else:
                threshold = internal_attrs.deaccumulation_invalid_below_threshold_rate
                assert threshold is not None
                deaccumulate_to_rates_inplace(
                    data_array,
                    dim="lead_time",
                    reset_frequency=internal_attrs.window_reset_frequency,
                    invalid_below_threshold_rate=threshold,
                    expected_clamp_fraction=EXPECTED_CLAMP_FRACTION,
                )

        super().apply_data_transformations(data_array, data_var)

    @classmethod
    def operational_update_jobs(
        cls,
        primary_store: Store,
        tmp_store: Path,
        get_template_fn: Callable[[DatetimeLike], xr.DataTree],
        append_dim: AppendDim,
        all_data_vars: Sequence[EcmwfIfsEns46DayDataVar],
        reformat_job_name: str,
    ) -> tuple[
        Sequence[RegionJob[EcmwfIfsEns46DayDataVar, EcmwfIfsEns46DaySourceFileCoord]],
        xr.DataTree,
    ]:
        existing_ds = xr.open_zarr(primary_store, chunks=None)
        append_dim_start = existing_ds[append_dim].max()
        append_dim_end = pd.Timestamp.now()
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


def _deaccumulate_signed_inplace(data_array: xr.DataArray) -> None:
    """Convert a signed accumulation to per-second rates over `lead_time`.

    `deaccumulate_to_rates_inplace` treats a negative step rate as a packing artifact
    to clamp or a corruption to drop, which is correct for the accumulations that can
    only grow. A net radiation, heat flux or surface stress accumulation moves in both
    directions, so its steps are differenced without any validity threshold.
    """
    seconds = np.diff(data_array["lead_time"].values) / np.timedelta64(1, "s")
    values = np.moveaxis(data_array.values, data_array.dims.index("lead_time"), 0)
    # Backwards so each step still sees the accumulation it is differenced against.
    for step in range(values.shape[0] - 1, 0, -1):
        values[step] -= values[step - 1]
        values[step] /= seconds[step - 1]
    values[0] = np.nan


def _sub_step_lead_times(
    lead_time: Timedelta, reduction: SubStepReduction | None
) -> tuple[Timedelta, ...]:
    if reduction is None:
        return ()
    return tuple(lead_time - offset for offset in reduction.offsets)


def _forecast_type(ensemble_member: object) -> str:
    if int(ensemble_member) == CONTROL_MEMBER:  # ty: ignore[invalid-argument-type]
        return "control_forecast"
    return "perturbed_forecast"


def _output_levels(
    processing_region_ds: xr.Dataset,
    ecds_variable: str,
    data_var: EcmwfIfsEns46DayDataVar,
) -> tuple[str | None, ...]:
    if "pressure_level" not in processing_region_ds[data_var.path].dims:
        return ()
    available = set(PRESSURE_LEVEL_VARIABLES[ecds_variable])
    levels = (
        f"{int(level)}_hpa" for level in processing_region_ds["pressure_level"].values
    )
    return tuple(level if level in available else None for level in levels)


@lru_cache(maxsize=4)
def _index_by_message(
    index_path: Path,
) -> dict[tuple[str, str, int, int], MessageRecord]:
    return {
        (
            record.variable,
            record.level,
            record.ensemble_member,
            record.lead_hours,
        ): record
        for record in read_index(index_path)
    }


def _message_byte_ranges(
    index_path: Path, coord: EcmwfIfsEns46DaySourceFileCoord
) -> tuple[list[int], list[int]]:
    index = _index_by_message(index_path)
    starts, ends = [], []
    for lead_time in coord.source_lead_times:
        lead_hours = whole_hours(pd.Timedelta(lead_time))
        for level in coord.present_levels or ("",):
            key = (coord.ecds_variable, level, coord.ensemble_member, lead_hours)
            record = index.get(key)
            assert record is not None, f"{key} is not in {index_path}"
            starts.append(record.offset)
            ends.append(record.offset + record.length)
    return starts, ends


def _validate_grib_metadata(
    reader: rasterio.DatasetReader, data_var: EcmwfIfsEns46DayDataVar
) -> None:
    internal_attrs = data_var.internal_attrs
    for band in range(1, reader.count + 1):
        tags = reader.tags(band)
        element = tags["GRIB_ELEMENT"]
        assert element == internal_attrs.grib_element, (
            f"{data_var.name}: {element=} != {internal_attrs.grib_element=}"
        )
        comment = tags["GRIB_COMMENT"]
        unit = comment[comment.rfind("[") :]
        assert unit == internal_attrs.grib_unit, (
            f"{data_var.name}: {unit=} != {internal_attrs.grib_unit=}"
        )
        if internal_attrs.grib_description:
            description = reader.descriptions[band - 1]
            assert description == internal_attrs.grib_description, (
                f"{data_var.name}: {description=} != {internal_attrs.grib_description=}"
            )
