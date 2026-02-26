from pathlib import Path

import pandas as pd
import pytest

from reformatters.common.pydantic import replace
from reformatters.noaa.gefs.common_gefs_template_config import (
    get_shared_data_var_configs,
)
from reformatters.noaa.gfs.forecast.template_config import (
    NoaaGfsForecastTemplateConfig,
)
from reformatters.noaa.hrrr.forecast_48_hour.template_config import (
    NoaaHrrrForecast48HourTemplateConfig,
)
from reformatters.noaa.noaa_grib_index import (
    _lead_time_str,
    grib_message_byte_ranges_from_index,
)
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
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))

    # First match is "PRES:surface:anl:ens mean"
    assert starts[0] == 1969991
    assert ends[0] == 2768859

    # Last match is "PRMSL:mean sea level:anl:ens mean"
    # We don't know the end byte so we add 10 GiB to the start byte to get the rest of the file
    assert starts[-1] == 13675595
    assert ends[-1] == 13675595 + 10 * (2**30)


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
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_gep01_s_f015() -> None:
    idx_path = IDX_FIXTURES_DIR / "gep01.t00z.pgrb2s.0p25.f015.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("15h")

    # Get GEFS variables expected in an "s" file
    data_vars = [
        v
        for v in get_shared_data_var_configs(CHUNKS, SHARDS)
        if v.internal_attrs.gefs_file_type == "s+a"
    ]
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_gec00_b_f003() -> None:
    idx_path = IDX_FIXTURES_DIR / "gec00.t00z.pgrb2b.0p50.f003.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("3h")

    # Get GEFS variables expected in a "b" file
    data_vars = [
        v
        for v in get_shared_data_var_configs(CHUNKS, SHARDS)
        if "b" in v.internal_attrs.gefs_file_type
    ]
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_gec00_a_f432() -> None:
    idx_path = IDX_FIXTURES_DIR / "gec00.t00z.pgrb2a.0p50.f432.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("432h")

    # Get GEFS variables expected in an "a" file
    data_vars = [
        v
        for v in get_shared_data_var_configs(CHUNKS, SHARDS)
        if "a" in v.internal_attrs.gefs_file_type
    ]
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_gfs_f000() -> None:
    idx_path = IDX_FIXTURES_DIR / "gfs.t00z.pgrb2.0p25.f000.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("0h")

    data_vars = [
        v for v in NoaaGfsForecastTemplateConfig().data_vars if has_hour_0_values(v)
    ]
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_gfs_f007() -> None:
    idx_path = IDX_FIXTURES_DIR / "gfs.t00z.pgrb2.0p25.f007.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("7h")

    data_vars = NoaaGfsForecastTemplateConfig().data_vars

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))

    # First match is "PRES:surface:7 hour fcst:"
    assert starts[0] == 409856805
    assert ends[0] == 410705050

    # Last match is "PRMSL:mean sea level:7 hour fcst:"
    # It's actually the first index row
    assert starts[-1] == 0
    assert ends[-1] == 895140


def test_grib_index_hrrr_f00() -> None:
    idx_path = IDX_FIXTURES_DIR / "hrrr.t00z.wrfsfcf00.grib2.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("0h")

    cfg = NoaaHrrrForecast48HourTemplateConfig()
    data_vars = [v for v in cfg.data_vars if has_hour_0_values(v)]
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_hrrr_f08() -> None:
    idx_path = IDX_FIXTURES_DIR / "hrrr.t00z.wrfsfcf08.grib2.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("8h")

    cfg = NoaaHrrrForecast48HourTemplateConfig()
    data_vars = list(cfg.data_vars)
    assert len(data_vars) > 0

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars, init_time, lead_time
    )

    assert len(starts) == len(data_vars)
    assert len(ends) == len(data_vars)
    assert all(isinstance(s, int) and s >= 0 for s in starts)
    assert all(isinstance(e, int) and e > 0 for e in ends)
    assert all(start < stop for start, stop in zip(starts, ends, strict=True))


def test_grib_index_hrrr_grib_element_alternatives() -> None:
    """Test that grib_element_alternatives matches when the primary element isn't in the index."""
    idx_path = IDX_FIXTURES_DIR / "hrrr.t00z.wrfsfcf00.grib2.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("0h")

    cfg = NoaaHrrrForecast48HourTemplateConfig()
    # The fixture has MSLMA. Switch the primary to PRMSL (not in index) with MSLMA as alternative.
    mslma_var = next(
        v for v in cfg.data_vars if v.internal_attrs.grib_element == "MSLMA"
    )
    prmsl_primary_var = replace(
        mslma_var,
        internal_attrs=replace(
            mslma_var.internal_attrs,
            grib_element="PRMSL",
            grib_element_alternatives=("MSLMA",),
        ),
    )

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, [prmsl_primary_var], init_time, lead_time
    )

    assert len(starts) == 1
    assert len(ends) == 1
    # MSLMA is at byte offset 26433488 in the fixture
    assert starts[0] == 26433488


def test_grib_index_skips_missing_vars() -> None:
    """Test that variables not present in the index are skipped instead of causing an error."""
    idx_path = IDX_FIXTURES_DIR / "hrrr.t00z.wrfsfcf00.grib2.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("0h")

    cfg = NoaaHrrrForecast48HourTemplateConfig()
    hour_0_vars = [v for v in cfg.data_vars if has_hour_0_values(v)]
    assert len(hour_0_vars) > 1

    # Add a var with a bogus element name that won't be in the index
    bogus_var = replace(
        hour_0_vars[0],
        name="bogus_var",
        internal_attrs=replace(
            hour_0_vars[0].internal_attrs,
            grib_element="BOGUS_ELEMENT",
            grib_element_alternatives=(),
        ),
    )

    data_vars_with_bogus = [hour_0_vars[0], bogus_var, hour_0_vars[1]]

    starts, ends = grib_message_byte_ranges_from_index(
        idx_path, data_vars_with_bogus, init_time, lead_time
    )

    # Bogus var skipped, so only 2 results instead of 3
    assert len(starts) == 2
    assert len(ends) == 2


def test_grib_index_raises_when_no_matches() -> None:
    """Test that a ValueError is raised when no variables match any index entry."""
    idx_path = IDX_FIXTURES_DIR / "hrrr.t00z.wrfsfcf00.grib2.idx"

    init_time = pd.Timestamp("2025-08-01T00")
    lead_time = pd.Timedelta("0h")

    cfg = NoaaHrrrForecast48HourTemplateConfig()
    bogus_var = replace(
        cfg.data_vars[0],
        name="bogus_var",
        internal_attrs=replace(
            cfg.data_vars[0].internal_attrs,
            grib_element="BOGUS_ELEMENT",
            grib_element_alternatives=(),
        ),
    )

    with pytest.raises(ValueError, match="No GRIB index matches found"):
        grib_message_byte_ranges_from_index(idx_path, [bogus_var], init_time, lead_time)


class TestLeadTimeStr:
    def setup_method(self) -> None:
        cfg = NoaaHrrrForecast48HourTemplateConfig()
        vars_by_name = {v.name: v for v in cfg.data_vars}
        self.instant_var = vars_by_name["temperature_2m"]
        self.accum_var = vars_by_name["precipitation_surface"]

    def test_analysis_hour(self) -> None:
        assert _lead_time_str(self.instant_var, lead_hours=0) == "anl"

    def test_instant_forecast(self) -> None:
        assert _lead_time_str(self.instant_var, lead_hours=8) == "8 hour fcst"

    def test_accum_hour_0(self) -> None:
        assert _lead_time_str(self.accum_var, lead_hours=0) == "0-0 day acc fcst"

    def test_accum_forecast(self) -> None:
        assert _lead_time_str(self.accum_var, lead_hours=8) == "7-8 hour acc fcst"

    def test_accum_at_reset_boundary(self) -> None:
        # At reset boundary (1h reset freq), reset_hour = lead_hours - reset_hours
        assert _lead_time_str(self.accum_var, lead_hours=1) == "0-1 hour acc fcst"

    def test_unhandled_step_type_raises(self) -> None:
        var_with_avg = replace(
            self.instant_var,
            attrs=replace(self.instant_var.attrs, step_type="avg"),
        )
        with pytest.raises(ValueError, match="Unhandled grib lead/accumulation hours"):
            _lead_time_str(var_with_avg, lead_hours=5)
