#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPOSITORY=$(cat "$RUNNER_TEMP/shared-image/repository")
SOURCE_DIGEST=$(cat "$RUNNER_TEMP/shared-image/digest")
[[ "$IMAGE_SHA" =~ ^[a-f0-9]{40}$ ]]
[[ "$IMAGE_DIGEST" =~ ^sha256:[a-f0-9]{64}$ ]]
[[ "$SOURCE_DIGEST" == "$IMAGE_DIGEST" ]]
[[ "$SOURCE_REPOSITORY" =~ ^[0-9]{12}\.dkr\.ecr\.us-west-2\.amazonaws\.com/[a-z0-9/_-]+$ ]]
[[ "$DOCKER_REPOSITORY" =~ ^[0-9]{12}\.dkr\.ecr\.us-east-1\.amazonaws\.com/[a-z0-9/_-]+$ ]]

umask 077
AUTH_DIR=$(mktemp -d "$RUNNER_TEMP/ecr-auth.XXXXXX")
trap 'rm -rf "$AUTH_DIR"' EXIT
aws ecr get-login-password --region us-west-2 | skopeo login --authfile "$AUTH_DIR/source.json" --username AWS --password-stdin "${SOURCE_REPOSITORY%%/*}"
aws ecr get-login-password --region us-east-1 | skopeo login --authfile "$AUTH_DIR/target.json" --username AWS --password-stdin "${DOCKER_REPOSITORY%%/*}"
skopeo copy --all --preserve-digests \
  --src-authfile "$AUTH_DIR/source.json" --dest-authfile "$AUTH_DIR/target.json" \
  --digestfile "$AUTH_DIR/copied-digest" \
  "docker://$SOURCE_REPOSITORY@$IMAGE_DIGEST" "docker://$DOCKER_REPOSITORY:$IMAGE_SHA"
[[ "$(cat "$AUTH_DIR/copied-digest")" == "$IMAGE_DIGEST" ]]
TARGET_DIGEST="sha256:$(skopeo inspect --authfile "$AUTH_DIR/target.json" --raw "docker://$DOCKER_REPOSITORY:$IMAGE_SHA" | sha256sum | cut -d ' ' -f 1)"
[[ "$TARGET_DIGEST" == "$IMAGE_DIGEST" ]]
