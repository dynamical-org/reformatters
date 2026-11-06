from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from functools import cache
from typing import Literal, Protocol

import boto3
import icechunk
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from botocore.paginate import Paginator

from reformatters.common import download
from reformatters.common.logging import get_logger
from reformatters.common.region_job import CoordinateValue, InitLeadSourceFileCoord
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import Dim
from reformatters.ecmwf.ecmwf_config_models import EcmwfDataVar

from .template_config import FastModel

SOURCE_PREFIX = "s3://dynamical-ecmwf/"
SOURCE_BUCKET = "dynamical-ecmwf"
SOURCE_SECRET = "dynamical-ecmwf-r2-storage-options-key"  # noqa: S105
ENDPOINT = "https://a037e2d3d5f9c6ecfc95350360f2ab8d.r2.cloudflarestorage.com"
log = get_logger(__name__)

_LIST_TRANSPORT_ERRORS = (
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

type Family = Literal["cf", "pf"]


class ListingClient(Protocol):
    def get_paginator(self, operation_name: str) -> Paginator: ...


FAMILY_PREFIXES: dict[tuple[FastModel, Family], str] = {
    ("ifs", "cf"): "dyn_a2_ifs-ens-cf_od_oper_fc_",
    ("ifs", "pf"): "dyn_a2_ifs-ens_od_enfo_pf_",
    ("aifs", "cf"): "dyn_a1_aifs-ens_ai_enfo_cf_",
    ("aifs", "pf"): "dyn_a1_aifs-ens_ai_enfo_pf_",
}


def virtual_containers() -> tuple[icechunk.VirtualChunkContainer, ...]:
    return (
        icechunk.VirtualChunkContainer(
            SOURCE_PREFIX,
            icechunk.s3_store(
                region="auto", endpoint_url=ENDPOINT, force_path_style=True
            ),
        ),
    )


class EcmwfFastSourceFileCoord(InitLeadSourceFileCoord):
    model: FastModel
    family: Family
    data_vars: Sequence[EcmwfDataVar]
    source_params: frozenset[str] = frozenset(
        {"2t", "10u", "10v", "100u", "100v", "tp", "ssrd"}
    )
    source_members: tuple[int, ...] | None = None

    @property
    def prefix(self) -> str:
        return FAMILY_PREFIXES[self.model, self.family] + self.init_time.strftime(
            "%Y%m%dT%H%M%SZ_"
        )

    @property
    def key(self) -> str:
        valid = self.init_time + self.lead_time
        return f"{self.prefix}{valid:%Y%m%dT%H%M%SZ}_{whole_hours(self.lead_time)}h"

    @property
    def members(self) -> Sequence[int]:
        if self.source_members is not None:
            return self.source_members
        return range(1) if self.family == "cf" else range(1, 51)

    def get_url(self) -> str:
        return SOURCE_PREFIX + self.key

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": min(self.members),
        }


@cache
def listing_client() -> ListingClient:
    credentials = download.load_s3_credentials(SOURCE_SECRET)
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="auto",
        aws_access_key_id=credentials["access_key_id"],
        aws_secret_access_key=credentials["secret_access_key"],
        aws_session_token=credentials.get("session_token"),
        config=Config(
            connect_timeout=2,
            read_timeout=2,
            retries={"mode": "standard", "total_max_attempts": 1},
            max_pool_connections=64,
            s3={"addressing_style": "path"},
        ),
    )


def list_prefix(
    prefix: str, client: ListingClient | None = None
) -> Iterator[tuple[str, int, datetime, str | None]]:
    client = client if client is not None else listing_client()
    try:
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=SOURCE_BUCKET, Prefix=prefix, PaginationConfig={"PageSize": 1000}
        )
        for page in pages:
            for obj in page.get("Contents", []):
                key, size = obj["Key"], obj["Size"]
                assert key.startswith(prefix), (
                    "S3 returned a key outside the requested prefix"
                )
                yield key, size, obj["LastModified"], obj.get("ETag")
    except ClientError as exc:
        response = exc.response
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (
            401,
            403,
        ) or response.get("Error", {}).get("Code") in {
            "AccessDenied",
            "InvalidAccessKeyId",
            "SignatureDoesNotMatch",
            "ExpiredToken",
            "InvalidToken",
        }:
            raise download.SourceAccessError(
                "Private source authentication failed"
            ) from exc
        raise


def list_prefix_or_defer(
    prefix: str, client: ListingClient
) -> list[tuple[str, int, datetime, str | None]]:
    try:
        return list(list_prefix(prefix, client))
    except _LIST_TRANSPORT_ERRORS:
        log.warning(f"Source listing transport failure; deferring {prefix}")
        return []
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = exc.response.get("Error", {}).get("Code")
        if status in (429, 500, 502, 503, 504) or code in {
            "SlowDown",
            "Throttling",
            "ThrottlingException",
            "RequestLimitExceeded",
            "RequestTimeout",
            "InternalError",
            "ServiceUnavailable",
        }:
            log.warning(f"Transient source listing failure; deferring {prefix}")
            return []
        raise
