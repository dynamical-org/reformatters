from pathlib import Path

import eccodes
import eccodeslib


def test_loaded_library_is_bundled_with_pthread_support() -> None:
    library_path = Path(eccodes.codes_get_library_path()).resolve()
    bundle_path = Path(eccodeslib.__file__).resolve().parent

    assert library_path.is_relative_to(bundle_path)
    assert (
        "ECCODES_THREADS"
        in eccodes.codes_get_features(eccodes.CODES_FEATURES_ENABLED).split()
    )
