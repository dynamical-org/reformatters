from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import ClassVar

import obstore
import pandas as pd
import xarray as xr
from obstore.exceptions import GenericError

from reformatters.common.download import s3_store_from_secret, source_access
from reformatters.common.logging import get_logger
from reformatters.common.retry import retry
from reformatters.common.types import Timedelta
from reformatters.common.virtual_region_job import VirtualRef, VirtualRegionJob
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .grib import (
    InterruptedGribStreamError,
    SourceContractError,
    StreamReader,
    inventory,
)
from .source import (
    ENDPOINT,
    SOURCE_PREFIX,
    SOURCE_SECRET,
    EcmwfFastSourceFileCoord,
    list_prefix_or_defer,
    listing_client,
)
from .template_config import FastModel, lead_hours

log = get_logger(__name__)


class TransientSourceReadError(ConnectionError):
    pass


class EcmwfFastRegionJob(VirtualRegionJob[EcmwfDataVar, EcmwfFastSourceFileCoord]):
    model: ClassVar[FastModel]
    operational_update_window: ClassVar[Timedelta] = pd.Timedelta("24h")

    def generate_source_file_coords(
        self,
        processing_region_ds: xr.Dataset,
        data_var_group: Sequence[EcmwfDataVar],
    ) -> Sequence[EcmwfFastSourceFileCoord]:
        leads = set(pd.to_timedelta(processing_region_ds.lead_time.values))
        members = set(processing_region_ds.ensemble_member.values)
        return [
            EcmwfFastSourceFileCoord(
                model=self.model,
                family=family,
                init_time=init,
                lead_time=lead,
                data_vars=data_var_group,
            )
            for init in pd.to_datetime(processing_region_ds.init_time.values)
            for hours in lead_hours(self.model, init.hour)
            if (lead := pd.Timedelta(hours=hours)) in leads
            for family in ("cf", "pf")
            if members.intersection((0,) if family == "cf" else range(1, 51))
        ]

    def discover_available(
        self, pending: list[EcmwfFastSourceFileCoord]
    ) -> list[tuple[EcmwfFastSourceFileCoord, int]]:
        by_key = {coord.key: coord for coord in pending}
        prefixes = sorted({coord.prefix for coord in pending})
        if not prefixes:
            return []
        client = listing_client(len(prefixes))
        available: dict[str, tuple[EcmwfFastSourceFileCoord, int]] = {}
        with ThreadPoolExecutor(max_workers=len(prefixes)) as pool:
            for rows in pool.map(
                partial(list_prefix_or_defer, client=client), prefixes
            ):
                for key, size in rows:
                    if size > 0 and (coord := by_key.get(key)) is not None:
                        available[key] = (coord, size)
        return list(available.values())

    def file_refs(
        self, coord: EcmwfFastSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        return retry(
            partial(self._stream_file_refs, coord, file_size),
            max_attempts=3,
            retryable_exceptions=(InterruptedGribStreamError, TransientSourceReadError),
        )

    def _stream_file_refs(
        self, coord: EcmwfFastSourceFileCoord, file_size: int
    ) -> list[VirtualRef]:
        store = s3_store_from_secret(
            SOURCE_PREFIX,
            region="auto",
            k8s_secret_name=SOURCE_SECRET,
            endpoint=ENDPOINT,
        )
        try:
            with source_access():
                response = obstore.get(store, coord.key)
                if response.meta["size"] != file_size:
                    raise SourceContractError("Source size differs from listing")
                reader = StreamReader(iter(response.stream(min_chunk_size=64 * 1024)))
                messages = inventory(reader, file_size, coord)
        except GenericError as exc:
            error = str(exc).lower()
            if any(
                marker in error
                for marker in (
                    "timed out",
                    "timeout",
                    "connection reset",
                    "connection closed",
                    "connection refused",
                    "error sending request",
                    "error decoding response body",
                    "incomplete message",
                    "500 internal",
                    "502 bad gateway",
                    "503 service unavailable",
                    "504 gateway timeout",
                    "429 too many",
                )
            ):
                raise TransientSourceReadError(
                    "Transient source streaming failure"
                ) from exc
            raise
        variables = {
            var.internal_attrs.grib_index_param: var for var in coord.data_vars
        }
        members = set(self.template_ds.ensemble_member.values)
        log.info(
            f"Scanned {coord.key}: {len(messages)} fields, {reader.position} bytes, peak stream chunk {reader.max_buffer_bytes} bytes"
        )
        return [
            VirtualRef(
                data_var=variables[message.param],
                out_loc={
                    "init_time": coord.init_time,
                    "lead_time": coord.lead_time,
                    "ensemble_member": message.member,
                },
                location=coord.get_url(),
                offset=message.offset,
                length=message.length,
            )
            for message in messages
            if message.param in variables and message.member in members
        ]
