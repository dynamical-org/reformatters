from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path

import pandas as pd
import xarray as xr

from reformatters.common.download import (
    http_download_to_disk,
)
from reformatters.common.iterating import digest, item
from reformatters.common.region_job import (
    CoordinateValueOrRange,
)
from reformatters.common.types import (
    Dim,
)
from reformatters.noaa.gfs.region_job import (
    NoaaGfsCommonRegionJob,
    NoaaGfsSourceFileCoord,
)
from reformatters.noaa.models import NoaaDataVar
from reformatters.noaa.noaa_grib_index import (
    grib_message_byte_ranges_from_index,
)
from reformatters.noaa.noaa_utils import has_hour_0_values


class NoaaGfsForecastSourceFileCoord(NoaaGfsSourceFileCoord):
    """Coordinates of a single source file to process for forecast dataset."""

    def out_loc(self) -> Mapping[Dim, CoordinateValueOrRange]:
        return {"init_time": self.init_time, "lead_time": self.lead_time}


class NoaaGfsForecastRegionJob(NoaaGfsCommonRegionJob[NoaaGfsForecastSourceFileCoord]):
    def download_file(self, coord: NoaaGfsForecastSourceFileCoord) -> Path:
        """Download the file for the given coordinate and return the local path.

        This method shadows the parent's implementation to use module-local imports
        for testability (tests can monkeypatch this module's imports).
        """
        # Download grib index file
        idx_url = f"{coord.get_url()}.idx"
        idx_local_path = http_download_to_disk(idx_url, self.dataset_id)

        # Download the grib messages for the data vars in the coord using byte ranges
        starts, ends = grib_message_byte_ranges_from_index(
            idx_local_path, coord.data_vars, coord.init_time, coord.lead_time
        )
        vars_suffix = digest(f"{s}-{e}" for s, e in zip(starts, ends, strict=True))
        return http_download_to_disk(
            coord.get_url(),
            self.dataset_id,
            byte_ranges=(starts, ends),
            local_path_suffix=f"-{vars_suffix}",
        )

    def generate_source_file_coords(
        self, processing_region_ds: xr.Dataset, data_var_group: Sequence[NoaaDataVar]
    ) -> Sequence[NoaaGfsForecastSourceFileCoord]:
        """Return a sequence of coords, one for each source file required to process the data covered by processing_region_ds."""
        var_has_hour_0_values = item({has_hour_0_values(v) for v in data_var_group})
        if not var_has_hour_0_values:
            processing_region_ds = processing_region_ds.sel(lead_time=slice("1h", None))

        return [
            NoaaGfsForecastSourceFileCoord(
                init_time=pd.Timestamp(init_time),
                lead_time=pd.Timedelta(lead_time),
                data_vars=data_var_group,
            )
            for init_time, lead_time in product(
                processing_region_ds["init_time"].values,
                processing_region_ds["lead_time"].values,
            )
        ]

    def update_template_with_results(
        self, process_results: Mapping[str, Sequence[NoaaGfsForecastSourceFileCoord]]
    ) -> xr.Dataset:
        """Update template dataset based on processing results."""
        return super().update_template_with_results(process_results)
