from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from reformatters.common.dynamical_dataset import DynamicalDataset
from reformatters.common.iterating import group_by, split_groups
from reformatters.common.region_job import SourceFileCoord
from reformatters.common.virtual_region_job import VirtualRegionJob
from reformatters.contrib.noaa.ndvi_cdr.analysis.region_job import (
    NoaaNdviCdrAnalysisRegionJob,
)
from reformatters.dwd.icon_eu.forecast_5_day.region_job import (
    DwdIconEuForecast5DaySourceFileCoord,
)
from reformatters.eccc.hrdps.forecast.region_job import EcccHrdpsForecastSourceFileCoord
from reformatters.ecmwf.aifs_single.forecast.region_job import (
    EcmwfAifsSingleForecastSourceFileCoord,
)
from reformatters.ecmwf.archive_gribs.archive import format_init_time
from reformatters.ecmwf.archive_gribs.forecast_46_day_archiver import (
    ARCHIVE_BASE_URL,
    ECDS_VARIABLES,
)
from reformatters.ecmwf.archive_gribs.grib_inventory import INDEX_SUFFIX
from reformatters.ecmwf.archive_gribs.request_shards import initialization_selections
from reformatters.ecmwf.ifs_ens.forecast_15_day_0_25_degree.source_file_coord import (
    OpenDataSourceFileCoord,
)
from reformatters.google.weathernext_virtual.listing import PROXY_LOCATION_PREFIX
from reformatters.google.weathernext_virtual.region_job import (
    WeatherNextVirtualRegionJob,
)
from reformatters.noaa.gefs.gefs_config_models import (
    GEFS_PRE_EXTENSION_MAX,
    NoaaGefsVirtualDataVar,
)
from reformatters.noaa.gfs.region_job import NoaaGfsSourceFileCoord
from reformatters.noaa.hrrr.region_job import NoaaHrrrSourceFileCoord

from .sources import TimestampSource, canonical_url


@dataclass(frozen=True)
class Unit:
    anchor: pd.Timestamp
    tier: str
    files: tuple[frozenset[str], ...]
    anchor_minus_label: float


def source_gap(dataset_id: str) -> str | None:
    if dataset_id.startswith("nasa-imerg"):
        return "PPS timestamps require credentials"
    if dataset_id.startswith("nasa-smap"):
        return "protected source; public production dates include reprocessing"
    if dataset_id == "u-arizona-swann-analysis":
        return "source mtimes rewritten in batches; not first-publication evidence"
    if "weathernext2-forecast-historical" in dataset_id:
        return "historical product has no operational arrival units"
    return None


def anchor_spacing(dataset: DynamicalDataset[Any, Any]) -> pd.Timedelta:
    if dataset.dataset_id.startswith(("noaa-gfs-analysis", "noaa-gefs-analysis")):
        return pd.Timedelta("6h")
    return dataset.template_config.append_dim_frequency


def usable_file(coord: SourceFileCoord) -> frozenset[str]:
    if isinstance(coord, DwdIconEuForecast5DaySourceFileCoord):
        return frozenset([coord.get_fallback_url()])
    if isinstance(coord, EcccHrdpsForecastSourceFileCoord):
        return frozenset([coord.get_datamart_url()])
    if isinstance(
        coord, (EcmwfAifsSingleForecastSourceFileCoord, OpenDataSourceFileCoord)
    ):
        return frozenset(
            [
                canonical_url(coord.get_url("gcs")),
                canonical_url(coord.get_index_url("gcs")),
            ]
        )
    url = canonical_url(coord.get_url())
    if isinstance(coord, (NoaaGfsSourceFileCoord, NoaaHrrrSourceFileCoord)):
        index_url = coord.get_idx_url()
    elif type(coord).get_index_url is not SourceFileCoord.get_index_url:
        index_url = coord.get_index_url()
    else:
        return frozenset([url])
    return frozenset([url, canonical_url(index_url)])


def timing_variables(dataset: DynamicalDataset[Any, Any]) -> list[Any]:
    variables = list(dataset.template_config.data_vars)
    if isinstance(variables[0], NoaaGefsVirtualDataVar):
        variables = [
            group[0]
            for group in group_by(
                variables,
                lambda var: (
                    var.internal_attrs.source_file_types,
                    var.has_hour_0_values(),
                    var.internal_attrs.available_from,
                ),
            )
        ]
    return variables


def archive_units(
    dataset: DynamicalDataset[Any, Any],
    anchors: pd.DatetimeIndex,
    label_offset: pd.Timedelta,
    source: TimestampSource,
) -> Iterator[Unit]:
    if "ifs-ens-forecast-46-day" in dataset.dataset_id:
        files = [
            selection.file_name
            for selection in initialization_selections(ECDS_VARIABLES)
        ]
        for anchor in anchors:
            base = f"{ARCHIVE_BASE_URL}/{format_init_time(anchor)}/"
            yield Unit(
                anchor,
                "archive complete",
                tuple(
                    frozenset(
                        [
                            canonical_url(base + name),
                            canonical_url(base + name + INDEX_SUFFIX),
                        ]
                    )
                    for name in files
                ),
                0,
            )
        return
    if issubclass(dataset.region_job_class, NoaaNdviCdrAnalysisRegionJob):
        for anchor in anchors:
            label = anchor - label_offset
            listing = source.list(f"s3://noaa-cdr-ndvi-pds/data/{label.year}/")
            urls = [
                url
                for url in listing.objects
                if f"_{label:%Y%m%d}_" in url and url.endswith(".nc")
            ]
            yield Unit(
                anchor,
                "position",
                (frozenset([max(urls)]),)
                if urls
                else (
                    frozenset(
                        [
                            f"s3://noaa-cdr-ndvi-pds/data/{label.year}/missing-{label:%Y%m%d}"
                        ]
                    ),
                ),
                label_offset.total_seconds(),
            )
        return


def expected_units(
    dataset: DynamicalDataset[Any, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: TimestampSource,
) -> Iterator[Unit]:
    dataset_id = dataset.dataset_id
    config = dataset.template_config
    spacing = anchor_spacing(dataset)
    frequency = config.append_dim_frequency
    observational = config.append_dim == "time" and not dataset_id.startswith(
        ("noaa-gfs", "noaa-gefs", "noaa-hrrr")
    )
    label_offset = (
        frequency
        if observational and frequency >= pd.Timedelta("1d")
        else pd.Timedelta(0)
    )
    anchors = pd.date_range(start.ceil(spacing), end, freq=spacing, inclusive="left")
    if "ifs-ens-forecast-46-day" in dataset_id or issubclass(
        dataset.region_job_class, NoaaNdviCdrAnalysisRegionJob
    ):
        yield from archive_units(dataset, anchors, label_offset, source)
        return
    coords = config.dimension_coordinates()
    variables = timing_variables(dataset)
    for anchor in anchors:
        if config.append_dim == "init_time":
            times = pd.DatetimeIndex([anchor])
        elif spacing > frequency:
            # The boundary supplies accumulated values for the cycle's last position.
            times = pd.date_range(anchor, anchor + spacing, freq=frequency)
        else:
            times = pd.DatetimeIndex([anchor - label_offset])
        processing = xr.Dataset(coords={**coords, config.append_dim: times})
        tree = xr.DataTree(processing)
        job = dataset.region_job_class(
            tmp_store=Path(),
            template_ds=tree,
            data_vars=variables,
            append_dim=config.append_dim,
            region=slice(0, len(times)),
            reformat_job_name="timing-profile",
        )
        groups = job.source_file_var_groups(job.data_vars)
        groups = split_groups(
            groups,
            getattr(job, "max_vars_per_download_group", None) or len(job.data_vars),
        )
        tiers: dict[str, set[frozenset[str]]] = {}
        for group in groups:
            for coord in job.generate_source_file_coords(processing, group):
                if (
                    spacing > frequency
                    and getattr(coord, "init_time", anchor) != anchor
                ):
                    continue
                tier = "run" if config.append_dim == "init_time" else "position"
                if dataset_id.startswith("noaa-gefs-forecast-35-day"):
                    tier = (
                        "tier 1 (through f384)"
                        if coord.lead_time <= GEFS_PRE_EXTENSION_MAX
                        else "tier 2 (extension)"
                    )
                if isinstance(job, WeatherNextVirtualRegionJob):
                    tier = f"{coord.data_vars[0].path} f{int(coord.lead_time.total_seconds() / 3600):03}"
                    files = frozenset(
                        canonical_url(chunk.location)
                        for chunk in job._source_chunks(coord)  # noqa: SLF001
                    )
                    # Proxy URLs map directly to GCS object keys.
                    bucket = (
                        "weathernext3_statistics_spatial"
                        if "weathernext3" in dataset_id
                        else "weathernext"
                    )
                    files = frozenset(
                        url.replace(PROXY_LOCATION_PREFIX, f"gs://{bucket}/")
                        for url in files
                    )
                else:
                    files = usable_file(coord)
                tiers.setdefault(tier, set()).add(files)
        for tier, files in tiers.items():
            yield Unit(
                anchor,
                tier,
                tuple(sorted(files, key=sorted)),
                label_offset.total_seconds(),
            )


def is_virtual(dataset: DynamicalDataset[Any, Any]) -> bool:
    return issubclass(dataset.region_job_class, VirtualRegionJob)
