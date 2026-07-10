#!/usr/bin/env python3
# ruff: noqa: T201
"""List, count, expire, and garbage collect icechunk repositories."""

import argparse
import datetime
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import icechunk

from reformatters.common.kubernetes import load_secret

DEFAULT_STAC_CATALOG_URL = "https://stac.dynamical.org/catalog.json"

K8S_SECRET_NAME = "aws-open-data-icechunk-storage-options-key"  # noqa: S105

AuthMode = Literal["secret", "env", "anonymous"]


def parse_older_than(value: str) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt


def load_storage_options() -> dict[str, Any]:
    return load_secret(K8S_SECRET_NAME)


def open_repo(s3_uri: str, auth: AuthMode) -> icechunk.Repository:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    match auth:
        case "anonymous":
            storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, anonymous=True)
        case "env":
            storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
        case "secret":
            storage = icechunk.s3_storage(
                bucket=bucket, prefix=prefix, **load_storage_options()
            )
    return icechunk.Repository.open(storage)


def load_icechunk_repos_from_stac(catalog_url: str) -> list[str]:
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        catalog = client.get(catalog_url).raise_for_status().json()
        repos = []
        for link in catalog["links"]:
            if link["rel"] != "child":
                continue
            collection = client.get(link["href"]).raise_for_status().json()
            repos.append(collection["assets"]["icechunk"]["href"])
    return repos


def get_repos(repo_uri: str | None, catalog_url: str) -> list[str]:
    if repo_uri is not None:
        return [repo_uri]
    return load_icechunk_repos_from_stac(catalog_url)


def expire(
    repos: list[str], older_than: datetime.datetime, force: bool, *, auth: AuthMode
) -> None:
    if not force:
        print(
            "Dry run mode. Would expire snapshots older than "
            f"{older_than.isoformat()} for the following repos:"
        )
        for uri in repos:
            print(f"  {uri}")
        print("\nPass --force to actually expire.")
        return

    for uri in repos:
        print(f"\n{'=' * 60}")
        print(f"Expiring snapshots: {uri}")
        print(f"Older than: {older_than.isoformat()}")
        print(f"{'=' * 60}")

        repo = open_repo(uri, auth)
        expired = repo.expire_snapshots(older_than)
        print(f"Expired {len(expired)} snapshots")


def garbage_collect(
    repos: list[str], older_than: datetime.datetime, force: bool, *, auth: AuthMode
) -> None:
    for uri in repos:
        print(f"\n{'=' * 60}")
        print(f"Garbage collecting: {uri}")
        print(f"Deleting objects older than: {older_than.isoformat()}")
        print(f"{'=' * 60}")

        repo = open_repo(uri, auth)
        dry_run = not force
        summary = repo.garbage_collect(older_than, dry_run=dry_run)
        prefix = "Would delete" if dry_run else "Deleted"
        print(
            f"{prefix}: {summary.chunks_deleted} chunks, {summary.manifests_deleted} manifests, "
            f"{summary.snapshots_deleted} snapshots, {summary.attributes_deleted} attributes, "
            f"{summary.transaction_logs_deleted} transaction logs"
        )
        print(
            f"{'Would free' if dry_run else 'Freed'}: {summary.bytes_deleted / 1024 / 1024:.1f} MB"
        )

        if dry_run:
            print("Dry run (pass --force to actually delete).")


def count_snapshots(repos: list[str], *, auth: AuthMode) -> None:
    for uri in repos:
        repo = open_repo(uri, auth)
        count = sum(1 for _ in repo.ancestry(branch="main"))
        print(f"{count:>8,}  {uri}")


def list_snapshots(repos: list[str], *, verbose: bool, auth: AuthMode) -> None:
    for uri in repos:
        repo = open_repo(uri, auth)
        snapshots = list(repo.ancestry(branch="main"))
        print(f"{uri}  ({len(snapshots)} snapshots)")
        if verbose:
            for snap in snapshots:
                print(f"  {snap.written_at.isoformat()}  {snap.message}")


def resolve_auth(args: argparse.Namespace) -> AuthMode:
    if args.anonymous:
        return "anonymous"
    if args.auth_from_env:
        return "env"
    return "secret"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List, count, expire, and garbage collect icechunk repositories."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="S3 URI of a single icechunk repo to process (default: all repos from STAC catalog)",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=DEFAULT_STAC_CATALOG_URL,
        help=f"STAC catalog URL to discover icechunk repos from (default: {DEFAULT_STAC_CATALOG_URL})",
    )
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument(
        "--auth-from-env",
        action="store_true",
        help="Use AWS credentials from environment instead of k8s secret",
    )
    auth_group.add_argument(
        "--anonymous",
        action="store_true",
        help="Use anonymous (public read-only) S3 access instead of k8s secret",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    expire_parser = subparsers.add_parser("expire", help="Expire old snapshots")
    expire_parser.add_argument(
        "--older-than",
        type=parse_older_than,
        required=True,
        help="ISO 8601 datetime cutoff — affect snapshots/objects strictly older than this (e.g. 2026-04-01T00:00+00:00)",
    )
    expire_parser.add_argument(
        "--force", action="store_true", help="Actually expire (default is dry run)"
    )

    gc_parser = subparsers.add_parser(
        "garbage-collect", help="Garbage collect unreachable objects"
    )
    gc_parser.add_argument(
        "--older-than",
        type=parse_older_than,
        required=True,
        help="ISO 8601 datetime cutoff — affect snapshots/objects strictly older than this (e.g. 2026-04-01T00:00+00:00)",
    )
    gc_parser.add_argument(
        "--force", action="store_true", help="Actually delete (default is dry run)"
    )

    subparsers.add_parser("count", help="Count snapshots on the main branch")

    list_parser = subparsers.add_parser(
        "list", help="List snapshots on the main branch"
    )
    list_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also print each snapshot's timestamp and commit message",
    )

    args = parser.parse_args()

    auth = resolve_auth(args)
    repos = get_repos(args.repo, args.catalog)

    match args.command:
        case "expire":
            expire(repos, args.older_than, args.force, auth=auth)
        case "garbage-collect":
            garbage_collect(repos, args.older_than, args.force, auth=auth)
        case "count":
            count_snapshots(repos, auth=auth)
        case "list":
            list_snapshots(repos, verbose=args.verbose, auth=auth)


if __name__ == "__main__":
    main()
