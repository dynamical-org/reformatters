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
    ensemble_member: int | None = None,
    *,
    step: int | None = None,
) -> tuple[list[int], list[int]]:
    """
    Given an ECMWF grib index file, returns the byte ranges the given data var(s) can be found within.

    Returns
    -------
    tuple[list[int], list[int]]
        list of byte range starts & a list of byte range ends. Elements of each list in order of data vars.
    """
    byte_range_starts: list[int] = []
    byte_range_ends: list[int] = []
    index_file_df = _parse_index_file(
        index_local_path, ensemble=ensemble_member is not None, step=step
    )
    for data_var in data_vars:
        level_selector = (
            slice(None)
            if np.isnan(level_value := data_var.internal_attrs.grib_index_level_value)
            else level_value
        )
        # Use MARS param name when filtering by step (MARS source)
        if (
            step is not None
            and data_var.internal_attrs.mars_grib_index_param is not None
        ):
            param = data_var.internal_attrs.mars_grib_index_param
        else:
            param = data_var.internal_attrs.grib_index_param

        if ensemble_member is not None:
            loc_key = (
                ensemble_member,
                param,
                data_var.internal_attrs.grib_index_level_type,
                level_selector,
            )
        else:
            loc_key = (
                param,
                data_var.internal_attrs.grib_index_level_type,
                level_selector,
            )
        row: pd.Series | pd.DataFrame = index_file_df.loc[
            loc_key,
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


def _parse_index_file(
    index_local_path: PathLike[str], *, ensemble: bool, step: int | None = None
) -> pd.DataFrame:
    """
    Parses an ECMWF index file into a pandas dataframe containing that information.

    For an example snippet of the contents of an index file, see:
        tests/ecmwf/ifs_ens/forecast_15_day_0_25_degree/region_job_test.py::test_region_job_download_file

    Parameters
    ----------
    step : int | None
        If provided, filter the index to only rows matching this forecast step.
        MARS indexes contain all steps in one file, so this is needed for MARS sources.

    Returns
    -------
    pd.DataFrame
        DataFrame representing the index file.
        Has a MultiIndex of (number, param, levtype, levelist) for ensemble data or
        (param, levtype, levelist) for deterministic data.
    """
    df = pd.read_json(index_local_path, lines=True)

    if step is not None:
        df = df[df["step"] == step]

    if "levelist" not in df.columns:
        df["levelist"] = np.nan

    index_cols = ["param", "levtype", "levelist"]

    if ensemble:
        index_cols = ["number", *index_cols]

        # Control-only indexes (e.g. MARS cf_sfc) may not have a "number" column at all
        if "number" not in df.columns:
            df["number"] = 0
        else:
            df["number"] = df["number"].fillna(0).astype(int)
        # Ensure that every row we filled with number=0 was indeed type "cf" (control forecast)
        assert (df[df["number"] == 0]["type"] == "cf").all(), (
            "Parsed row as control member that didn't have type='cf'"
        )

    return df.set_index(index_cols).sort_index()
