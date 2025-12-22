from collections.abc import Sequence
from os import PathLike

import numpy as np
import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.ecmwf.ecmwf_config_models import (
    EcmwfInternalAttrs,
)


def get_message_byte_ranges_from_index(
    index_local_path: PathLike[str],
    data_vars: Sequence[DataVar[EcmwfInternalAttrs]],
    ensemble_member: int,
) -> tuple[list[int], list[int]]:
    """
    Given an ECMWF grib index file, returns the byte ranges the given data var(s) & ensemble member can be found within.

    Returns
    -------
    tuple[list[int], list[int]]
        list of byte range starts & a list of byte range ends. Elements of each list in order of data vars.
    """
    byte_range_starts: list[int] = []
    byte_range_ends: list[int] = []
    index_file_df = _parse_index_file(index_local_path)
    for data_var in data_vars:
        level_selector = (
            slice(None)
            if np.isnan(level_value := data_var.internal_attrs.grib_index_level_value)
            else level_value
        )
        row: pd.Series | pd.DataFrame = index_file_df.loc[
            (
                ensemble_member,
                data_var.internal_attrs.grib_index_param,
                data_var.internal_attrs.grib_index_level_type,
                level_selector,
            ),
            ["_offset", "_length"],
        ]
        if isinstance(row, pd.DataFrame):
            if len(row) == 1:
                row = row.iloc[0]
            else:
                raise AssertionError(f"Expected exactly one match, but found: {row}")
        assert isinstance(row, pd.Series)
        start, length = row.values
        byte_range_starts.append(int(start))
        byte_range_ends.append(int(start + length))
    return byte_range_starts, byte_range_ends


def _parse_index_file(index_local_path: PathLike[str]) -> pd.DataFrame:
    """
    Parses an ECMWF index file into a pandas dataframe containing that information.

    For an example snippet of the contents of an index file, see:
        tests/ecmwf/ifs_ens/forecast_15_day_0_25_degree/region_job_test.py::test_region_job_download_file

    Returns
    -------
    pd.DataFrame
        DataFrame representing the index file.
        Has a MultiIndex of (number, param) representing ensemble member & data variable short name.
        Useful columns include: type (control/perturbed), _offset (start of byte window within grib file), _length (length of byte window)
        Additional columns include: levtype (sfc/pl/sol/...), levelist, domain, date, time, step, expver, class, stream
    """
    df = pd.read_json(index_local_path, lines=True)

    # Control members by default don't have "number" field. We fill with 0
    df["number"] = df["number"].fillna(0).astype(int)
    # Ensure that every row we filled with number=0 was indeed type "cf" (control forecast)
    assert (df[df["number"] == 0]["type"] == "cf").all(), (
        "Parsed row as control member that didn't have type='cf'"
    )
    return df.set_index(["number", "param", "levtype", "levelist"]).sort_index()
