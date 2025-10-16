from collections.abc import Sequence
from os import PathLike

import pandas as pd

from reformatters.common.config_models import DataVar
from reformatters.common.iterating import item
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.template_config import (
    EcmwfIfsEnsInternalAttrs,
)


def get_message_byte_ranges_from_index(
    index_local_path: PathLike[str],
    data_vars: Sequence[DataVar[EcmwfIfsEnsInternalAttrs]],
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
        rows = index_file_df.loc[
            (ensemble_member, data_var.internal_attrs.grib_index_param),
            ["_offset", "_length"],
        ]
        start, length = item(rows.values)
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
    assert all(df[df["number"] == 0]["type"] == "cf"), (
        "Parsed row as control member that didn't have type='cf'"
    )

    return df.set_index(["number", "param"]).sort_index()
