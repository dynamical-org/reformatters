import httpx
import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> object:
    try:
        return (yield)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        if item.get_closest_marker("slow") is None:
            raise
        pytest.skip(f"CHIRPS source unreachable: {e!r}")
