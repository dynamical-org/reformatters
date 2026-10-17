import obstore
from obstore.exceptions import BaseError

from reformatters.common.logging import get_logger
from reformatters.common.region_job import SOURCE_FILE_COORD

log = get_logger(__name__)


def discover_available_by_obstore_listing(
    pending: list[SOURCE_FILE_COORD],
    *,
    store: obstore.store.ObjectStore,
    location_prefix: str,
    require_index: bool,
) -> list[tuple[SOURCE_FILE_COORD, int]]:
    """The ready subset of `pending`, each paired with its data file's size in bytes.

    A file is ready once `store` lists its data object (`coord.get_url()`), plus, when
    `require_index`, its index (`coord.get_index_url()`) — the index lands alongside
    the data, so a source whose files have no sidecar index passes require_index=False.
    `location_prefix` is the url:// prefix `store` is rooted at, stripped from coord
    URLs to form store-relative keys. Returns the same coord objects from `pending`.
    """
    by_key = {coord.get_url().removeprefix(location_prefix): coord for coord in pending}
    prefixes = sorted({key.rsplit("/", 1)[0] + "/" for key in by_key})
    listed = _list_objects(store, prefixes)
    return [
        (coord, listed[key])
        for key, coord in by_key.items()
        if key in listed
        and (
            not require_index
            or coord.get_index_url().removeprefix(location_prefix) in listed
        )
    ]


def _list_objects(
    store: obstore.store.ObjectStore, prefixes: list[str]
) -> dict[str, int]:
    """All object keys under each prefix in `store`, mapped to size in bytes."""
    listed = {}
    for prefix in prefixes:
        try:
            for batch in obstore.list(store, prefix=prefix, chunk_size=10_000):
                listed.update((meta["path"], meta["size"]) for meta in batch)
        except OSError, BaseError:
            log.exception(f"Could not finish listing source prefix {prefix}")
    return listed
