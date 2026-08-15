import numpy as np

from reformatters.common import validation

# For regions outside of CONUS, the values in this dataset are expected
# to be NaNs. We sampled various times across the dataset and determined
# the expected fraction of NaNs to be ~0.46425.
EXPECTED_NAN_FRACTION = 0.46425
MAX_NAN_FRACTION = EXPECTED_NAN_FRACTION + 0.00001


class CheckRandomTimeWithinLastYearNans(validation.Validator):
    """
    Check NaN fraction at a single random time within the last year.

    The operational update pulls a year's worth of data, so we want to verify
    older timesteps in that window remain healthy.
    """

    def check(
        self, context: validation.ValidationContext
    ) -> validation.ValidationResult:
        rng = np.random.default_rng()
        random_time_index = int(rng.integers(0, 365)) + 1
        sample_ds = context.ds.isel(time=[-random_time_index])

        problem_vars = []
        for var_name, da in sample_ds.data_vars.items():
            nan_fraction = float(da.isnull().mean().compute().item())
            if nan_fraction > MAX_NAN_FRACTION:
                problem_vars.append((str(var_name), nan_fraction))

        if problem_vars:
            message = "Excessive NaN fraction:\n"
            for var, fraction in problem_vars:
                message += f"- {var}: {fraction:.6f} NaN fraction\n"
            return validation.ValidationResult(passed=False, message=message)

        return validation.ValidationResult(
            passed=True,
            message=(
                f"All variables have NaN fraction <= {MAX_NAN_FRACTION:.6f} at a "
                "random time within the last year"
            ),
        )
