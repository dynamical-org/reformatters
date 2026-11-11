import logging
import runpy
from pathlib import Path
from typing import Any

import pytest
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from reformatters import __main__ as main
from reformatters.common.config import Config


@pytest.mark.parametrize("cluster", [None, "original-deployment"])
def test_sentry_cluster_identity(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cluster: str | None,
) -> None:
    if cluster is None:
        monkeypatch.delenv("DYNAMICAL_CLUSTER", raising=False)
    else:
        monkeypatch.setenv("DYNAMICAL_CLUSTER", cluster)
    monkeypatch.setattr(Config, "sentry_dsn", "https://key@example.invalid/1")
    envelopes: list[Envelope] = []

    class RecordingTransport(Transport):
        def capture_envelope(self, envelope: Envelope) -> None:
            envelopes.append(envelope)

    def init(**options: Any) -> None:  # noqa: ANN401
        sdk_client = sentry_sdk.Client(**options, transport=RecordingTransport())
        sentry_sdk.get_current_scope().set_client(sdk_client)

    monkeypatch.setattr(sentry_sdk, "init", init)
    caplog.set_level(logging.INFO)
    with sentry_sdk.isolation_scope(), sentry_sdk.new_scope():
        runpy.run_path(str(Path(main.__file__)))
        sdk_client = sentry_sdk.get_client()
        try:
            sentry_sdk.capture_message("cluster event")
            logging.getLogger(__name__).info("cluster log")
            sdk_client.flush()
        finally:
            sdk_client.close()

    events = [
        payload
        for envelope in envelopes
        for item in envelope.items
        if item.type == "event"
        if (payload := item.payload.json) is not None
    ]
    logs = [
        log
        for envelope in envelopes
        for item in envelope.items
        if item.type == "log"
        if (payload := item.payload.json) is not None
        for log in payload["items"]
    ]
    event = next(event for event in events if event["message"] == "cluster event")
    log = next(log for log in logs if log["body"] == "cluster log")
    assert event["environment"] == Config.env.value
    if cluster is None:
        assert "cluster" not in event.get("tags", {})
        assert "cluster" not in log["attributes"]
    else:
        assert event["tags"]["cluster"] == cluster
        assert log["attributes"]["cluster"]["value"] == cluster
