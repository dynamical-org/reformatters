from pathlib import Path

import pandas as pd

from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_data_var_configs,
)
from reformatters.noaa.noaa_utils import has_hour_0_values
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index


def test_grib_index_geavg_happy_path() -> None:
    fixtures_dir = Path(__file__).parent / "fixtures"
    idx_path = fixtures_dir / "geavg.t00z.pgrb2s.0p25.f000.idx"

    # Per instructions: use init_time 2025-08-01T00 when not clear from filename
    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("0h")

    # dummy chunk/shard sizes
    chunks = (1, 1, 1, 1)
    shards = (1, 1, 1, 1)

    data_vars = list(get_shared_data_var_configs(chunks, shards))

    # For lead_time == 0, filter out vars that do not have hour-0 values
    if lead_time == pd.Timedelta("0h"):
        data_vars = [v for v in data_vars if has_hour_0_values(v)]

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
