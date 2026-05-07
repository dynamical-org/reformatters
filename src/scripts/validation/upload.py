import mimetypes
import os
from pathlib import Path

import boto3
import typer

from reformatters.common.logging import get_logger
from scripts.validation.render import REPORT_FILENAME, render_report

log = get_logger(__name__)

BUCKET = "dataset-validation-reports"
PUBLIC_BASE_URL = "https://dataset-validation-reports.dynamical.org"


def _content_type(path: Path) -> str:
    ct, _ = mimetypes.guess_type(path.name)
    return ct or "application/octet-stream"


def upload(run_dir: Path, publish: bool) -> str:
    dataset_id = run_dir.parent.name
    version_ts = run_dir.name
    if publish:
        prefixes = [
            f"{dataset_id}/latest",
            f"{dataset_id}/published/{version_ts}",
        ]
    else:
        prefixes = [f"{dataset_id}/drafts/{version_ts}"]

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_VALIDATION_REPORTS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_VALIDATION_REPORTS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_VALIDATION_REPORTS_SECRET_ACCESS_KEY"],
    )
    files = [f for f in sorted(run_dir.rglob("*")) if f.is_file()]
    for prefix in prefixes:
        for f in files:
            key = f"{prefix}/{f.relative_to(run_dir).as_posix()}"
            client.upload_file(
                str(f),
                BUCKET,
                key,
                ExtraArgs={"ContentType": _content_type(f)},
            )
            log.info(f"uploaded {key}")
    return f"{PUBLIC_BASE_URL}/{prefixes[0]}/{REPORT_FILENAME}"


run_dir_argument = typer.Argument(..., help="Path to the run directory")
publish_option = typer.Option(
    False,
    "--publish",
    help="Upload to <dataset-id>/latest/ and <dataset-id>/published/<version_ts>/. "
    "Without this flag, uploads to <dataset-id>/drafts/<version_ts>/ only.",
)


def upload_command(
    run_dir: Path = run_dir_argument,
    publish: bool = publish_option,
) -> None:
    render_report(run_dir)
    url = upload(run_dir, publish)
    log.info(f"Done: {url}")
    typer.echo(url)
