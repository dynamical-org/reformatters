#!/bin/bash
# Create an R2 bucket, mint an R2 API token scoped to it, and write the kubernetes
# secret a dataset's StorageConfig reads via `k8s_secret_name`. Safe to re-run: an
# existing bucket is reused and the secret is applied, rotating its credentials.
set -euo pipefail

BUCKET_NAME=${1:-}
K8S_SECRET_NAME=${2:-}

if [ -z "$BUCKET_NAME" ] || [ -z "$K8S_SECRET_NAME" ]; then
  cat <<USAGE
Usage: $0 <bucket-name> <k8s-secret-name>

Required env:
  CLOUDFLARE_ACCOUNT_ID
  CLOUDFLARE_API_TOKEN    R2 admin permissions, plus "API Tokens Write" to mint
                          the bucket scoped token

Optional env:
  R2_ACCESS_KEY_ID        use these credentials instead of minting a new token
  R2_SECRET_ACCESS_KEY
USAGE
  exit 1
fi

: "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID}"
: "${CLOUDFLARE_API_TOKEN:?set CLOUDFLARE_API_TOKEN}"

for cmd in npx kubectl jq curl sha256sum uv; do
  command -v "$cmd" > /dev/null || { echo "Error: $cmd not found"; exit 1; }
done

cd "$(git rev-parse --show-toplevel)"

WRANGLER="npx --yes wrangler@4"

if $WRANGLER r2 bucket info "$BUCKET_NAME" > /dev/null 2>&1; then
  echo "Bucket $BUCKET_NAME already exists"
else
  echo "Creating bucket $BUCKET_NAME..."
  $WRANGLER r2 bucket create "$BUCKET_NAME"
fi

if [ -z "${R2_ACCESS_KEY_ID:-}" ] || [ -z "${R2_SECRET_ACCESS_KEY:-}" ]; then
  echo "Minting an R2 API token scoped to $BUCKET_NAME..."
  # Object Read & Write on this one bucket. Resource name format and permission
  # group id: https://developers.cloudflare.com/r2/api/tokens/
  token_request=$(jq -n \
    --arg name "$K8S_SECRET_NAME" \
    --arg resource "com.cloudflare.edge.r2.bucket.${CLOUDFLARE_ACCOUNT_ID}_default_${BUCKET_NAME}" \
    '{
      name: $name,
      policies: [{
        effect: "allow",
        resources: {($resource): "*"},
        permission_groups: [{id: "2efd5506f9c8494dacb1fa10a3e7d5b6"}]
      }]
    }')
  token_response=$(curl -sS -X POST \
    "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/tokens" \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "$token_request")

  if [ "$(jq -r '.success' <<< "$token_response")" != "true" ]; then
    echo "Token creation failed:"
    jq -r '.errors' <<< "$token_response"
    exit 1
  fi

  # The S3 access key id is the token id and the secret is the SHA-256 of the
  # token value, see https://developers.cloudflare.com/r2/api/tokens/
  R2_ACCESS_KEY_ID=$(jq -r '.result.id' <<< "$token_response")
  R2_SECRET_ACCESS_KEY=$(jq -r '.result.value' <<< "$token_response" \
    | tr -d '\n' | sha256sum | cut -d' ' -f1)
fi

export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY BUCKET_NAME
export R2_ENDPOINT_URL="https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"

echo "Verifying credentials against $BUCKET_NAME..."
# A freshly minted token takes a few seconds to become usable.
for attempt in 1 2 3 4 5 6; do
  if uv run python - <<'PYTHON'
import os

import boto3

boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",  # R2 only accepts its own region names
).head_bucket(Bucket=os.environ["BUCKET_NAME"])
PYTHON
  then
    break
  fi
  if [ "$attempt" = 6 ]; then
    echo "Error: credentials never became usable for $BUCKET_NAME"
    exit 1
  fi
  sleep 5
done

echo "Applying kubernetes secret $K8S_SECRET_NAME..."
# The contents are the kwargs `icechunk.s3_storage` is called with, see
# reformatters.common.storage.
jq -n '{
  region: "auto",
  endpoint_url: env.R2_ENDPOINT_URL,
  access_key_id: env.R2_ACCESS_KEY_ID,
  secret_access_key: env.R2_SECRET_ACCESS_KEY,
  force_path_style: true
}' | kubectl create secret generic "$K8S_SECRET_NAME" \
  --namespace default \
  --from-file=contents=/dev/stdin \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Done! Bucket $BUCKET_NAME and secret $K8S_SECRET_NAME are ready."
