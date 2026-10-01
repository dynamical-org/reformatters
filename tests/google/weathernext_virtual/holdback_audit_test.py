from itertools import product
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pytest
import zarr

from reformatters.google.weathernext_virtual.holdback_audit import (
    forbidden_chunk_keys,
    run_audit,
)


@pytest.mark.parametrize("spatial", [("y", "x"), ("latitude", "longitude")])
@pytest.mark.parametrize(
    "extra_dims", [(), ("statistic",), ("statistic", "pressure_level")]
)
def test_audit_all_nonspatial_dimensions(
    spatial: tuple[str, str], extra_dims: tuple[str, ...], tmp_path: Path
) -> None:
    repo = icechunk.Repository.create(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w")
    root.create_array(
        "init_time",
        data=np.array([0, 3600], dtype="int64"),
        dimension_names=("init_time",),
        attributes={"units": "seconds since 1970-01-01 00:00:00"},
    )
    root.create_array(
        "lead_time",
        data=np.array([3600, 7200], dtype="float64"),
        dimension_names=("lead_time",),
        attributes={"units": "seconds"},
    )
    sizes = {"statistic": 6, "pressure_level": 2}
    chunks = {"statistic": 2, "pressure_level": 1}
    root.create_array(
        "temperature",
        shape=(2, 2, 2, 2, *(sizes[dim] for dim in extra_dims)),
        chunks=(1, 1, 1, 2, *(chunks[dim] for dim in extra_dims)),
        dtype="float32",
        dimension_names=("lead_time", spatial[0], "init_time", spatial[1], *extra_dims),
    )
    planted_key = (
        "temperature/c/1/1/1/0"
        + "/2" * ("statistic" in extra_dims)
        + "/1" * ("pressure_level" in extra_dims)
    )
    session.store.set_virtual_ref(
        planted_key,
        "https://example.com/unread-chunk",
        offset=0,
        length=16,
        validate_container=False,
    )
    snapshot = session.commit("plant forbidden reference")
    cutoff = pd.Timestamp("1970-01-01T02:00")
    expected = {
        "temperature/c/1/" + str(y) + "/1/0" + "".join(f"/{i}" for i in indices)
        for y in range(2)
        for indices in product(
            *(range(sizes[dim] // chunks[dim]) for dim in extra_dims)
        )
    }
    group = zarr.open_group(repo.readonly_session(snapshot_id=snapshot).store, mode="r")
    assert {chunk.key for chunk in forbidden_chunk_keys(group, cutoff)} == expected

    result = run_audit(repo, "fixture", cutoff, snapshot, None, tmp_path)

    assert result.total_keys == len(expected)
    assert result.present_keys == 1
    assert result.present_steps == {
        (pd.Timestamp("1970-01-01T01:00"), pd.Timedelta("2h"))
    }
    assert result.max_present_valid_time == pd.Timestamp("1970-01-01T03:00")
    assert result.last_age_out == pd.Timestamp("1970-01-01T04:00")
    assert f"- Total keys probed: {len(expected)}" in result.report_path.read_text()
    assert repo.lookup_branch("main") == snapshot
