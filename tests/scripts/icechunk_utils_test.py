import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import icechunk_utils


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], {"dry_run": True}),
        (
            [
                "--max-snapshots-in-memory",
                "16",
                "--max-compressed-manifest-mem-bytes",
                "134217728",
                "--max-concurrent-manifest-fetches",
                "32",
            ],
            {
                "dry_run": True,
                "max_snapshots_in_memory": 16,
                "max_compressed_manifest_mem_bytes": 134217728,
                "max_concurrent_manifest_fetches": 32,
            },
        ),
    ],
)
def test_gc_passes_requested_limits(
    monkeypatch: pytest.MonkeyPatch, options: list[str], expected: dict[str, int | bool]
) -> None:
    gc = Mock(
        return_value=SimpleNamespace(
            chunks_deleted=0,
            manifests_deleted=0,
            snapshots_deleted=0,
            attributes_deleted=0,
            transaction_logs_deleted=0,
            bytes_deleted=0,
        )
    )
    monkeypatch.setattr(
        icechunk_utils,
        "open_repo",
        lambda *args, **kwargs: SimpleNamespace(garbage_collect=gc),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "icechunk_utils.py",
            "--repo",
            "s3://bucket/repo",
            "garbage-collect",
            "--older-than",
            "2026-09-20T14:30:00+00:00",
            *options,
        ],
    )

    icechunk_utils.main()

    gc.assert_called_once_with(datetime(2026, 9, 20, 14, 30, tzinfo=UTC), **expected)


def test_icechunk_utils_parses_k8s_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], icechunk_utils.AuthMode, str]] = []

    def fake_count_snapshots(
        repos: list[str], *, auth: icechunk_utils.AuthMode, k8s_secret: str
    ) -> None:
        calls.append((repos, auth, k8s_secret))

    monkeypatch.setattr(icechunk_utils, "count_snapshots", fake_count_snapshots)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "icechunk_utils.py",
            "--repo",
            "s3://bucket/repo.icechunk",
            "--k8s-secret",
            "weathernext2-storage-options-key",
            "count",
        ],
    )

    icechunk_utils.main()

    assert calls == [
        (
            ["s3://bucket/repo.icechunk"],
            "secret",
            "weathernext2-storage-options-key",
        )
    ]
