from collections.abc import Mapping, Sequence
from typing import ClassVar

import pandas as pd

from reformatters.common.pydantic import replace
from reformatters.common.region_job import (
    CoordinateValueOrRange,
    SourceFileCoord,
)
from reformatters.common.time_utils import whole_hours
from reformatters.common.types import (
    Dim,
    Timedelta,
    Timestamp,
)
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfDataVar,
    _resolve_grib_index_param,
)

MARS_OPEN_DATA_CUTOVER = pd.Timestamp("2024-04-01T00:00")

DYNAMICAL_MARS_GRIB_BASE_URL = (
    "https://data.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens"
)


class OpenDataSourceFileCoord(SourceFileCoord):
    """Source file coord for ECMWF open data (one GRIB per init_time x lead_time)."""

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_member: int
    data_var_group: Sequence[EcmwfDataVar]

    s3_bucket_url: ClassVar[str] = "ecmwf-forecasts"
    s3_region: ClassVar[str] = "eu-central-1"

    def resolve_data_vars(self) -> "OpenDataSourceFileCoord":
        return replace(
            self,
            data_var_group=[
                _resolve_grib_index_param(v, self.lead_time)
                for v in self.data_var_group
            ],
        )

    @property
    def index_step(self) -> int | None:
        return None

    @property
    def validate_grib_comment_unit_only(self) -> bool:
        return False

    def _get_base_url(self) -> str:
        base_url = f"https://{self.s3_bucket_url}.s3.{self.s3_region}.amazonaws.com"

        init_time_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")  # pads 0 to be "00", as desired
        lead_time_hour_str = whole_hours(self.lead_time)

        # On 2024-02-29 and onward, the /ifs/ directory is included in the URL path.
        if self.init_time >= pd.Timestamp("2024-02-29T00:00"):
            directory_path = f"{init_time_str}/{init_hour_str}z/ifs/0p25/enfo"
        else:
            directory_path = f"{init_time_str}/{init_hour_str}z/0p25/enfo"

        filename = f"{init_time_str}{init_hour_str}0000-{lead_time_hour_str}h-enfo-ef"
        return f"{base_url}/{directory_path}/{filename}"

    def get_url(self) -> str:
        return self._get_base_url() + ".grib2"

    def get_index_url(self) -> str:
        return self._get_base_url() + ".index"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


def _resolve_mars_data_var(data_var: EcmwfDataVar) -> EcmwfDataVar:
    """Resolve data var attributes for the MARS source.

    Clears date_available (MARS has all configured vars) and applies any
    MARS-specific attribute overrides.
    """
    overrides: dict[str, object] = {"date_available": None, "mars": None}
    if data_var.internal_attrs.mars is not None:
        overrides.update(
            {
                k: v
                for k, v in data_var.internal_attrs.mars.model_dump().items()
                if v is not None
            }
        )
    return replace(
        data_var,
        internal_attrs=replace(data_var.internal_attrs, **overrides),
    )


class MarsSourceFileCoord(SourceFileCoord):
    """Source file coord for MARS archive data on source.coop."""

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_member: int
    data_var_group: Sequence[EcmwfDataVar]
    request_type: str

    @staticmethod
    def get_request_type(levtype: str, ensemble_member: int) -> str:
        """Map a level type and ensemble member to the MARS request type used in file paths."""
        if levtype == "sfc":
            if ensemble_member == 0:
                return "cf_sfc"
            return "pf_sfc_0" if ensemble_member <= 25 else "pf_sfc_1"
        return "cf_pl" if ensemble_member == 0 else "pf_pl"

    def resolve_data_vars(self) -> "MarsSourceFileCoord":
        return replace(
            self,
            data_var_group=[_resolve_mars_data_var(v) for v in self.data_var_group],
        )

    @property
    def index_step(self) -> int | None:
        # MARS indexes contain all steps; filter to the one we need
        return whole_hours(self.lead_time)

    @property
    def validate_grib_comment_unit_only(self) -> bool:
        # MARS GRIBs use different descriptive text than open data (e.g.
        # "2 metre temperature" vs "Temperature") but the physical unit matches
        return True

    def _date_str(self) -> str:
        return self.init_time.strftime("%Y-%m-%d")

    def get_url(self) -> str:
        return f"{DYNAMICAL_MARS_GRIB_BASE_URL}/{self._date_str()}/{self.request_type}.grib"

    def get_index_url(self) -> str:
        return f"{DYNAMICAL_MARS_GRIB_BASE_URL}/{self._date_str()}/{self.request_type}.grib.idx"

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


type IfsEnsSourceFileCoord = OpenDataSourceFileCoord | MarsSourceFileCoord
