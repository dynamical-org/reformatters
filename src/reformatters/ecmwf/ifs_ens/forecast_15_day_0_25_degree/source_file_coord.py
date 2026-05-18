from collections.abc import Mapping, Sequence
from typing import Literal, assert_never

import pandas as pd

from reformatters.common.pydantic import replace
from reformatters.common.region_job import (
    CoordinateValue,
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
from reformatters.ecmwf.ecmwf_utils import EcmwfOpenDataSource

type MarsSource = Literal["s3-source-coop"]

MARS_OPEN_DATA_CUTOVER = pd.Timestamp("2024-04-01T00:00")

# IFS Cycle 50r1 (2026-05-12 06z) merged ex-HRES and the ENS control into one
# forecast, disseminated as stream=oper, type=fc instead of stream=enfo, type=cf.
# From the cutover, the control member lives in oper-fc; perturbed members remain in enfo-ef.
IFS_CYCLE_50R1_CUTOVER = pd.Timestamp("2026-05-12T06:00")

DYNAMICAL_MARS_GRIB_BASE_URL = (
    "https://data.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens"
)


class OpenDataSourceFileCoord(SourceFileCoord):
    """Source file coord for ECMWF open data (one GRIB per init_time x lead_time)."""

    init_time: Timestamp
    lead_time: Timedelta
    ensemble_member: int
    data_var_group: Sequence[EcmwfDataVar]

    def resolve_data_vars(self) -> OpenDataSourceFileCoord:
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

    def _get_base_url(self, source: EcmwfOpenDataSource) -> str:
        match source:
            case "s3":
                base_url = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
            case "gcs":
                base_url = "https://storage.googleapis.com/ecmwf-open-data"
            case _ as unreachable:
                assert_never(unreachable)

        init_time_str = self.init_time.strftime("%Y%m%d")
        init_hour_str = self.init_time.strftime("%H")  # pads 0 to be "00", as desired
        lead_time_hour_str = whole_hours(self.lead_time)

        use_oper_control = (
            self.init_time >= IFS_CYCLE_50R1_CUTOVER and self.ensemble_member == 0
        )
        stream_dir = "oper" if use_oper_control else "enfo"
        file_kind = "oper-fc" if use_oper_control else "enfo-ef"

        # On 2024-02-29 and onward, the /ifs/ directory is included in the URL path.
        if self.init_time >= pd.Timestamp("2024-02-29T00:00"):
            directory_path = f"{init_time_str}/{init_hour_str}z/ifs/0p25/{stream_dir}"
        else:
            directory_path = f"{init_time_str}/{init_hour_str}z/0p25/{stream_dir}"

        filename = (
            f"{init_time_str}{init_hour_str}0000-{lead_time_hour_str}h-{file_kind}"
        )
        return f"{base_url}/{directory_path}/{filename}"

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

    def resolve_data_vars(self) -> MarsSourceFileCoord:
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

    def get_url(self, source: MarsSource = "s3-source-coop") -> str:
        match source:
            case "s3-source-coop":
                base_url = DYNAMICAL_MARS_GRIB_BASE_URL
            case _ as unreachable:
                assert_never(unreachable)
        return f"{base_url}/{self._date_str()}/{self.request_type}.grib"

    def get_index_url(self, source: MarsSource = "s3-source-coop") -> str:
        match source:
            case "s3-source-coop":
                base_url = DYNAMICAL_MARS_GRIB_BASE_URL
            case _ as unreachable:
                assert_never(unreachable)
        return f"{base_url}/{self._date_str()}/{self.request_type}.grib.idx"

    def out_loc(self) -> Mapping[Dim, CoordinateValue]:
        return {
            "init_time": self.init_time,
            "lead_time": self.lead_time,
            "ensemble_member": self.ensemble_member,
        }


type IfsEnsSourceFileCoord = OpenDataSourceFileCoord | MarsSourceFileCoord
