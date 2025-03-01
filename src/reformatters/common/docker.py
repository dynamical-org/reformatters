import logging
import os
import re
import subprocess

import pandas as pd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_and_push_image() -> str:
    """Returns the image tag"""
    docker_repo = os.environ["DOCKER_REPOSITORY"]
    assert re.fullmatch(r"[0-9a-zA-Z_\.\-\/]{1,1000}", docker_repo)
    job_timestamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H-%M-%SZ")
    image_tag = f"{docker_repo}:{job_timestamp}"

    # If depot cli is installed where expected, use it to accelerate builds
    if os.path.exists("/usr/bin/depot"):
        builder = ["/usr/bin/depot"]
    else:
        builder = ["/usr/bin/docker", "buildx"]

    subprocess.run(  # noqa: S603  allow passing variable to subprocess, it's realtively sanitized above
        [
            *builder,
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
    logger.info(f"Pushed {image_tag}")

    return image_tag
