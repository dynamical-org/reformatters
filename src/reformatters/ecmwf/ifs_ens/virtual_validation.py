from collections.abc import Sequence
from typing import Any, ClassVar

import xarray as xr
import zarr
from icechunk.store import IcechunkStore
from zarr.core.metadata import ArrayV3Metadata

from reformatters.common.config_models import ROOT
from reformatters.common.region_job import SourceFileCoord
from reformatters.common.validation import (
    CheckVirtualDecodeHealth,
    ValidationContext,
    ValidationResult,
    Validator,
)
from reformatters.common.virtual_region_job import VirtualRegionJob, _exists_many

from .virtual_region_job import (
    EcmwfIfsEnsForecastVirtualRegionJob,
    EcmwfIfsEnsForecastVirtualSourceFileCoord,
)


class EcmwfIfsEnsVirtualManifestCompleteness(Validator):
    """Require every expected message across the ingested update window."""

    requires_virtual_dataset: ClassVar[bool] = True

    def check(self, context: ValidationContext) -> ValidationResult:
        job = context.virtual_region_job()
        assert isinstance(job, EcmwfIfsEnsForecastVirtualRegionJob)
        store = context.store
        assert isinstance(store, IcechunkStore)
        through = context.ds.get_index(job.append_dim).max()
        candidates = [c for c in job.source_file_coords() if c.init_time <= through]
        if not candidates:
            return ValidationResult(
                passed=False,
                checked_count=0,
                message="No source files in the init_time window to check",
            )
        group = zarr.open_group(store, mode="r")
        missing: list[str] = []
        checked = 0
        for var in job.data_vars:
            coords = [c for c in candidates if var in c.data_vars]
            if not coords:
                continue
            array = group[var.path]
            assert isinstance(array, zarr.Array)
            metadata = array.metadata
            assert isinstance(metadata, ArrayV3Metadata)
            assert metadata.dimension_names
            assert metadata.dimension_names[0] == job.append_dim
            encode = metadata.chunk_key_encoding.encode_chunk_key
            for init_time in sorted({c.init_time for c in coords}):
                locations = [
                    loc
                    for coord in coords
                    if coord.init_time == init_time
                    for loc in job.message_locations(coord, var)
                ]
                indices = job._resolve_chunk_keys([(loc, var) for loc in locations])  # noqa: SLF001
                assert all(index is not None for index in indices)
                expected = {
                    f"{array.path}/{encode(index)}"
                    for index in indices
                    if index is not None
                }
                present = _exists_many(store, list(expected))
                absent = {key for key in expected if not present[key]}
                checked += len(expected)
                if absent:
                    missing.append(
                        f"{var.path} at {init_time}: {len(absent)}/{len(expected)} message references missing"
                    )
        return ValidationResult(
            passed=not missing,
            checked_count=checked,
            message="\n".join(missing)
            if missing
            else f"All {checked} expected message references present",
        )


class EcmwfIfsEnsVirtualDecodeHealth(CheckVirtualDecodeHealth):
    def _decode_coord(
        self,
        coord: SourceFileCoord,
        region_job: VirtualRegionJob[Any, Any],
        store: IcechunkStore,
        group: zarr.Group,
        ds: xr.Dataset,
    ) -> tuple[list[tuple[str, float, str | None]], set[str]]:
        assert isinstance(coord, EcmwfIfsEnsForecastVirtualSourceFileCoord)
        results: list[tuple[str, float, str | None]] = []
        skipped: set[str] = set()
        for var in coord.data_vars:
            selection: dict[str, Sequence[int]] = {
                "ensemble_member": list(coord.members_for(var))
            }
            if var.group is not ROOT:
                selection["pressure_level"] = list(coord.pressure_levels)
            var_coord = coord.model_copy(update={"data_vars": [var]})
            decoded, absent = super()._decode_coord(
                var_coord, region_job, store, group, ds.sel(selection)
            )
            results.extend(decoded)
            skipped.update(absent)
        return results, skipped
