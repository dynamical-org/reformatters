from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd
import xarray as xr

from reformatters.common.download import (  # noqa: F401
    s3_download_to_disk,
    s3_store,
)
from reformatters.common.logging import get_logger
from reformatters.common.region_job import (
    CoordinateValue,
    SourceFileCoord,
)
from reformatters.common.types import (
    Dim,
    Timedelta,
)
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.common.virtual_source_listing import (
    discover_available_by_obstore_listing,  # noqa: F401
)

from .template_config import ExampleDataVar

log = get_logger(__name__)

# The url:// prefix the source files live under. It must match a registered
# VirtualChunkContainer in the dataset's icechunk_virtual_config (see
# dynamical_dataset.py) so the manifest is allowed to reference these bytes.
# _SOURCE_PREFIX = "s3://noaa-gefs-pds/"
# _SOURCE_REGION = "us-east-1"


class ExampleSpatialSourceFileCoord(SourceFileCoord):
    """Coordinates of a single source file, plus the data variables it packs.

    A virtual dataset turns each source GRIB message into one chunk reference, so a
    coord must carry enough to (a) locate the file (`get_url`/`get_index_url`) and
    (b) locate every output cell its messages fill (`out_loc`). `data_vars` records
    which variables this particular file contains - the message subset `file_refs`
    will resolve byte ranges for.
    """

    # init_time: Timestamp
    # lead_time: Timedelta
    # data_vars: Sequence[ExampleDataVar]

    def get_url(self) -> str:
        """The source file's location. Refs point here, so it must start with the
        registered virtual chunk container prefix (e.g. ``s3://...``), not an
        ``https://`` mirror of the same bucket."""
        # return (
        #     f"{_SOURCE_PREFIX}gefs.{self.init_time:%Y%m%d}/{self.init_time:%H}/atmos/"
        #     f"pgrb2sp25/gec00.t{self.init_time:%H}z.pgrb2s.0p25."
        #     f"f{self.lead_time.total_seconds() / 3600:03.0f}"
        # )
        raise NotImplementedError("Return the location of the source file.")

    def get_index_url(self) -> str:
        """The byte-range index sidecar (`.idx`) listing each message's start byte.

        `file_refs` reads this to resolve message byte ranges without downloading the
        whole data file. Implement only if the source publishes such an index; a source
        without one resolves byte ranges by scanning the data file instead.
        """
        # return self.get_url() + ".idx"
        raise NotImplementedError(
            "Return the URL of the source file's byte-range index."
        )

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        """The output cell(s) this file's messages fill, as a {dim: label} map.

        The default returns every field of the coord, which here would wrongly include
        `data_vars` (not an output dimension), so override to return just the dim
        labels. Override is also where an analysis dataset folds init+lead into a single
        `time` (``{"time": self.init_time + self.lead_time}``).
        """
        # return {"init_time": self.init_time, "lead_time": self.lead_time}
        raise NotImplementedError(
            "Return the {dim: label} location of this file's data."
        )


class ExampleSpatialRegionJob(
    VirtualRegionJob[ExampleDataVar, ExampleSpatialSourceFileCoord]
):
    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[ExampleDataVar]
    ) -> Sequence[ExampleSpatialSourceFileCoord]:
        """One coord per source file covering the data in processing_region_ds.

        Reused by operational validation to probe the manifest, so it must list exactly
        the files the dataset expects (drop files the source genuinely lacks, e.g.
        accumulated variables at hour 0).
        """
        # return [
        #     ExampleSpatialSourceFileCoord(
        #         init_time=pd.Timestamp(init_time),
        #         lead_time=pd.Timedelta(lead_time),
        #         data_vars=data_var_group,
        #     )
        #     for init_time in processing_region_ds["init_time"].values
        #     for lead_time in processing_region_ds["lead_time"].values
        # ]
        raise NotImplementedError(
            "Return one SourceFileCoord per source file covering processing_region_ds."
        )

    def discover_available(
        self, pending: list[ExampleSpatialSourceFileCoord]
    ) -> list[tuple[ExampleSpatialSourceFileCoord, int]]:
        """Of the not-yet-ingested files, the subset fetchable now, each with its size.

        The write loop calls this each tick (once for a backfill, repeatedly while an
        update polls). For an object store obstore can list (S3/GCS/Azure/local) this is
        one line; `require_index=True` also waits for each file's `.idx` sidecar to land.
        For a source obstore can't list (an HTML directory index, a frontier to probe,
        or "assume every coord is available"), implement it directly and return the
        ready (coord, file_size) pairs.
        """
        # return discover_available_by_obstore_listing(
        #     pending,
        #     store=s3_store(_SOURCE_PREFIX, region=_SOURCE_REGION),
        #     location_prefix=_SOURCE_PREFIX,
        #     require_index=True,
        # )
        raise NotImplementedError(
            "Return the (coord, file_size) pairs ready to fetch now."
        )

    def file_refs(
        self, coord: ExampleSpatialSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        """Every virtual ref a single source file contributes (or [] to skip it).

        Resolve each message's byte range - parse the `.idx` sidecar
        (`coord.get_index_url()`), scan the data file, or, for one-message files, point
        at the whole file - and return one VirtualRef per (output cell, variable). The
        chunk index is resolved centrally later, so refs are in coordinate-label space:
        give each `out_loc` (the cell it fills) and the source byte range. Return [] to
        drop an unreadable or stale file. `file_size` is what discover_available
        reported - use it to supply a final message's missing end byte.
        """
        # index_path = s3_download_to_disk(
        #     coord.get_index_url(), self.dataset_id, region=_SOURCE_REGION
        # )
        # try:
        #     # reformatters.noaa.noaa_grib_index.grib_message_byte_ranges_from_index
        #     # parses a .idx into per-variable (start, end) byte ranges.
        #     starts, ends = grib_message_byte_ranges_from_index(
        #         index_path, coord.data_vars, coord.init_time, coord.lead_time
        #     )
        # finally:
        #     index_path.unlink()
        #
        # out_loc = coord.out_loc()
        # location = coord.get_url()
        # return [
        #     VirtualRef(
        #         data_var=var,
        #         out_loc=out_loc,
        #         location=location,
        #         offset=start,
        #         length=end - start,
        #     )
        #     for var, start, end in zip(coord.data_vars, starts, ends, strict=True)
        # ]
        raise NotImplementedError(
            "Return the VirtualRefs for one source file, or [] to skip it."
        )

    # filter_already_present probes one "representative" variable per file to decide
    # whether the file is already in the manifest. The default picks the first instant
    # variable the file carries. Override only if that variable's chunk isn't a reliable
    # proxy for "the whole file was ingested":
    #
    # def representative_var(self, coord: ExampleSpatialSourceFileCoord) -> ExampleDataVar:
    #     return next(v for v in self.data_vars if v.name == "temperature_2m")

    # The recent append-dim window each operational update fire re-sweeps. The base
    # VirtualRegionJob.operational_update_jobs builds a single job over this window
    # with processing_mode="update", which makes the write loop POLL: it keeps
    # sweeping discover_available until every expected file is ingested, committing
    # each batch as files publish (backfills sweep once and exit). What is already
    # ingested is derived from the icechunk manifest by filter_already_present, so
    # size the window for late-publishing files and re-checking recent steps, and
    # override operational_update_jobs itself only for an update shape a single
    # windowed job can't express. See "Operational updates" in
    # docs/virtual_datasets.md.
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("24h")
