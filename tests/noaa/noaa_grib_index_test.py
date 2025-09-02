from pathlib import Path

import pandas as pd
import pytest

from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_data_var_configs,
)
from reformatters.noaa.noaa_grib_index import grib_message_byte_ranges_from_index
from reformatters.noaa.noaa_utils import has_hour_0_values

IDX_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# dummy chunk/shard sizes
CHUNKS = (1, 1, 1, 1)
SHARDS = (1, 1, 1, 1)


def test_grib_index_geavg_s_f000() -> None:
    idx_path = IDX_FIXTURES_DIR / "geavg.t00z.pgrb2s.0p25.f000.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("0h")

    # Get GEFS variables expected in an "s" file at hour 0
    data_vars = [
        v
        for v in get_shared_data_var_configs(CHUNKS, SHARDS)
        if has_hour_0_values(v) and v.internal_attrs.gefs_file_type == "s+a"
    ]

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_geavg_s_f009() -> None:
    idx_path = IDX_FIXTURES_DIR / "geavg.t00z.pgrb2s.0p25.f009.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("9h")

    # Get GEFS variables expected in an "s" file
    data_vars = [
        v
        for v in get_shared_data_var_configs(CHUNKS, SHARDS)
        if v.internal_attrs.gefs_file_type == "s+a"
    ]

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))
