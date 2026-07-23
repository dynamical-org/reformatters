import httpx
import pytest

from scripts.validation import upload


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None


def test_trigger_pages_deploy_is_noop_when_hook_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PAGES_DEPLOY_HOOK_URL", raising=False)
    posted: list[str] = []

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        posted.append(url)
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    upload._trigger_pages_deploy()
    assert posted == []


def test_trigger_pages_deploy_posts_hook_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAGES_DEPLOY_HOOK_URL", "https://pages.example/deploy/abc")
    posted: list[str] = []

    def fake_post(url: str, **kwargs: object) -> _FakeResponse:
        posted.append(url)
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    upload._trigger_pages_deploy()
    assert posted == ["https://pages.example/deploy/abc"]
