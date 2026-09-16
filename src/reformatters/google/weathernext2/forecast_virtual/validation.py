from typing import ClassVar

from icechunk.store import IcechunkStore

from reformatters.common.validation import (
    ValidationContext,
    ValidationResult,
    Validator,
)

from .region_job import GoogleWeathernext2ForecastVirtualRegionJob

_REPORTED_STEPS = 20


class CheckNoRefsInsideHoldback(Validator):
    """Check representative refs for held-back steps in the operational window."""

    requires_virtual_dataset: ClassVar[bool] = True

    def check(self, context: ValidationContext) -> ValidationResult:
        region_job = context.virtual_region_job()
        assert isinstance(region_job, GoogleWeathernext2ForecastVirtualRegionJob)
        store = context.store
        assert isinstance(store, IcechunkStore)
        append_dim = region_job.append_dim

        ingested_through = context.ds.get_index(append_dim).max()
        template_through = (
            region_job.template_ds.to_dataset().get_index(append_dim).max()
        )
        if ingested_through > template_through:
            return ValidationResult(
                passed=False,
                message=(
                    f"The store's {append_dim} extends to {ingested_through}, past the "
                    f"update job's template ({template_through}); its held-back steps "
                    "cannot be probed"
                ),
            )

        candidates = [
            coord
            for coord in region_job.held_back_source_file_coords()
            if coord.init_time <= ingested_through
        ]
        missing = {
            id(coord) for coord in region_job.filter_already_present(candidates, store)
        }
        present = sorted(
            (coord.init_time, coord.lead_time, coord.data_vars[0].path)
            for coord in candidates
            if id(coord) not in missing
        )
        cutoff = region_job.publication_cutoff
        if not present:
            return ValidationResult(
                passed=True,
                message=(
                    f"None of the {len(candidates)} steps with a valid time after the "
                    f"publication cutoff {cutoff} has a representative ref"
                ),
                checked_count=len(candidates),
            )

        present_inits = {init_time for init_time, _, _ in present}
        sample = present[:_REPORTED_STEPS]
        return ValidationResult(
            passed=False,
            message=(
                f"{len(present)} of {len(candidates)} steps with a valid time after "
                f"the publication cutoff {cutoff} have representative refs, across "
                f"{len(present_inits)} initialization(s) "
                f"({min(present_inits)} to {max(present_inits)}). First {len(sample)}:\n"
                + "\n".join(
                    f"- {append_dim}={init_time} lead_time={lead_time} {var_path}"
                    for init_time, lead_time, var_path in sample
                )
            ),
            checked_count=len(candidates),
        )
