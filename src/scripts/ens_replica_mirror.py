"""Operator-gated ENS replica verification and handoff; see docs/ops_card.md."""

import json
from pathlib import Path

import typer

from reformatters.__main__ import DYNAMICAL_DATASETS
from reformatters.common import replica_mirror as mirror
from reformatters.common.storage import StoreFactory

app = typer.Typer()


def factory() -> StoreFactory:
    return next(
        dataset.store_factory
        for dataset in DYNAMICAL_DATASETS
        if dataset.dataset_id == "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
    )


@app.command()
def pending(handoff_id: str) -> None:
    mirror.begin_handoff(factory(), handoff_id)


@app.command()
def adoption_dry_run(snapshot: str, report: Path) -> None:
    report.write_text(
        json.dumps(mirror.adoption_dry_run(factory(), snapshot), indent=2)
    )


@app.command()
def adopt(
    snapshot: str,
    inventory_sha256: str,
    rehearsal: str,
    drained_writers: str,
    max_read_bytes: int = 0,
) -> None:
    mirror.adopt(
        factory(),
        snapshot,
        inventory_sha256=inventory_sha256,
        rehearsal=rehearsal,
        drained_writers=drained_writers,
        max_read_bytes=max_read_bytes,
    )


@app.command()
def epoch_dry_run(before: str, after: str, report: Path) -> None:
    report.write_text(
        json.dumps(mirror.epoch_dry_run(factory(), before, after), indent=2)
    )


@app.command()
def prepare_epoch(
    before: str, after: str, inventory_sha256: str, max_read_bytes: int = 0
) -> None:
    mirror.prepare_epoch(
        factory(),
        before,
        after,
        inventory_sha256=inventory_sha256,
        max_read_bytes=max_read_bytes,
    )


@app.command()
def cancel_epoch(before: str, after: str, evidence: str) -> None:
    mirror.cancel_epoch(factory(), before=before, after=after, evidence=evidence)


@app.command()
def recover_lock(owner: str, attestation: Path) -> None:
    mirror.recover_abandoned_lock(
        factory(),
        owner,
        evidence=mirror.DeadWriterAttestation.model_validate_json(
            attestation.read_bytes()
        ),
    )


@app.command()
def retry() -> None:
    mirror.retry_mirror(factory())


if __name__ == "__main__":
    app()
