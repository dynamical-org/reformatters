import os
import re
import subprocess

import pandas as pd

from reformatters.common.logging import get_logger

log = get_logger(__name__)


def build_and_push_image() -> str:
    """Returns the image tag"""
    docker_repo = os.environ["DOCKER_REPOSITORY"]
    assert re.fullmatch(r"[0-9a-zA-Z_\.\-\/]{1,1000}", docker_repo)
    job_timestamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H-%M-%SZ")
    image_tag = f"{docker_repo}:{job_timestamp}"

    subprocess.run(  # noqa: S603  allow passing variable to subprocess, it's realtively sanitized above
        [
            "/usr/bin/docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64,linux/arm64",
            "--push",
            "--file",
            "deploy/Dockerfile",
            "--tag",
            image_tag,
            ".",
        ],
        check=True,
    )
    log.info(f"Pushed {image_tag}")

    return image_tag
