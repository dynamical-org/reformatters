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
    steps: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """
    Given an ECMWF grib index file, returns the byte ranges the given data var(s) can be found within.

    When steps is provided, byte ranges are returned for each (step, data_var) combination,
    ordered as: all data_vars for step[0], then all data_vars for step[1], etc.

    Returns
    -------
    tuple[list[int], list[int]]
        list of byte range starts & a list of byte range ends.
    """
    byte_range_starts: list[int] = []
    byte_range_ends: list[int] = []
    index_file_df = _parse_index_file(
        index_local_path, ensemble=ensemble_member is not None, steps=steps
    )

    iterate_steps = steps if steps is not None else [None]
    for step in iterate_steps:
        for data_var in data_vars:
            level_selector = (
                slice(None)
                if np.isnan(
                    level_value := data_var.internal_attrs.grib_index_level_value
                )
                else level_value
            )
            param = data_var.internal_attrs.grib_index_param

            loc_key: tuple[object, ...] = ()
            if step is not None:
                loc_key += (step,)
            if ensemble_member is not None:
                loc_key += (ensemble_member,)
            loc_key += (
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
                    raise AssertionError(
                        f"Expected exactly one match, but found: {row}"
                    )
            assert isinstance(row, pd.Series)
            start, length = row.values
            byte_range_starts.append(int(start))
            byte_range_ends.append(int(start + length))
    return byte_range_starts, byte_range_ends


def _parse_index_file(
    index_local_path: PathLike[str],
    *,
    ensemble: bool,
    steps: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Parses an ECMWF index file into a pandas dataframe containing that information.

    For an example snippet of the contents of an index file, see:
        tests/ecmwf/ifs_ens/forecast_15_day_0_25_degree/region_job_test.py::test_region_job_download_file

    Returns
    -------
    pd.DataFrame
        DataFrame representing the index file.
        When steps is provided, the MultiIndex includes step as the first level.
    """
    df = pd.read_json(index_local_path, lines=True)

    if steps is not None:
        df = df[df["step"].isin(steps)]

    if "levelist" not in df.columns:
        df["levelist"] = np.nan

    index_cols: list[str] = []

    if steps is not None:
        index_cols.append("step")

    if ensemble:
        index_cols.append("number")
        # Control-only indexes (e.g. MARS cf_sfc) may not have a "number" column at all
        if "number" not in df.columns:
            df["number"] = 0
        else:
            df["number"] = df["number"].fillna(0).astype(int)
        # Ensure that every row we filled with number=0 was indeed type "cf" (control forecast)
        assert (df[df["number"] == 0]["type"] == "cf").all(), (
            "Parsed row as control member that didn't have type='cf'"
        )

    index_cols.extend(["param", "levtype", "levelist"])

    return df.set_index(index_cols).sort_index()
