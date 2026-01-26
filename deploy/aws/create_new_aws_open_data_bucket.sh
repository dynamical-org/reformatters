#!/bin/bash
set -e

MODEL_ID=${1:-}

if [ -z "$MODEL_ID" ]; then
  echo "Usage: $0 <model-id>"
  exit 1
fi

# Check that model-id doesn't start with "dynamical"
if [[ "$MODEL_ID" == dynamical* ]]; then
  echo "Error: model-id should not start with 'dynamical' (it will be added automatically)"
  exit 1
fi

# Check that model-id is lowercase alphanumeric with hyphens only
if [[ ! "$MODEL_ID" =~ ^[a-z0-9-]+$ ]]; then
  echo "Error: model-id must contain only lowercase letters, numbers, and hyphens"
  exit 1
fi

BUCKET_NAME="dynamical-${MODEL_ID}"
STACK_NAME="${BUCKET_NAME}-pds-bucket"

# Make sure we're logged in
aws login

# Step 0-1: Deploy CloudFormation stack
echo "Creating CloudFormation stack..."
aws cloudformation create-stack \
  --stack-name "$STACK_NAME" \
  --template-url https://s3-us-west-2.amazonaws.com/opendata.aws/pds-bucket-cf.yml \
  --parameters ParameterKey=DataSetName,ParameterValue="$BUCKET_NAME" \
  --region us-west-2

# Wait for stack creation
echo "Waiting for stack creation to complete..."
aws cloudformation wait stack-create-complete \
  --stack-name "$STACK_NAME" \
  --region us-west-2

# Step 2: Enable versioning
echo "Enabling versioning..."
aws s3api put-bucket-versioning \
  --bucket "$BUCKET_NAME" \
  --versioning-configuration Status=Enabled

# Step 3: Set all lifecycle rules
echo "Setting lifecycle rules..."
aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET_NAME" \
  --lifecycle-configuration '{
    "Rules": [
      {
        "ID": "IntelligentTieringRule",
        "Status": "Enabled",
        "Filter": {},
        "Transitions": [
          {
            "Days": 0,
            "StorageClass": "INTELLIGENT_TIERING"
          }
        ]
      },
      {
        "ID": "AbortIncompleteMultipartUploadRule",
        "Status": "Enabled",
        "Filter": {},
        "AbortIncompleteMultipartUpload": {
          "DaysAfterInitiation": 7
        }
      },
      {
        "ID": "DeleteOldVersions",
        "Status": "Enabled",
        "Filter": {},
        "NoncurrentVersionExpiration": {
          "NoncurrentDays": 7
        },
        "Expiration": {
          "ExpiredObjectDeleteMarker": true
        }
      }
    ]
  }'

# Step 4: Enable access logging
echo "Enabling access logging..."
aws s3api put-bucket-logging \
  --bucket "$BUCKET_NAME" \
  --bucket-logging-status '{
    "LoggingEnabled": {
      "TargetBucket": "dynamical-org-open-data-s3-access-logs",
      "TargetPrefix": "",
      "TargetObjectKeyFormat": {
        "PartitionedPrefix": {
          "PartitionDateSource": "EventTime"
        }
      }
    }
  }'

# Step 5: Copy index.html
echo "Copying index.html..."
aws s3 cp s3://dynamical-noaa-hrrr/index.html s3://${BUCKET_NAME}/index.html

echo "Done! Bucket ${BUCKET_NAME} created and configured."
