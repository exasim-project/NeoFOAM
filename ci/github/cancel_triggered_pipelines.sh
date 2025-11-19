#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
# ---------------------------------------------------------------------------------------
# Cancel running/pending LRZ GitLab CI pipelines on TUM COMA cluster for a given branch.
# Only cancel if the pipeline does not have the NEON_BRANCH variable set.
#
# Uses environment variables defined in the GitHub Actions job:
#   LRZ_GROUP, LRZ_HOST, REPO_NAME, LRZ_GITLAB_PROJECT_TOKEN
#
# Usage:
#   ./ci/github/scripts/cancel_triggered_pipelines.sh <branch>
# ---------------------------------------------------------------------------------------

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <branch>"
  exit 1
fi

BRANCH=$1
GROUP="${LRZ_GROUP:?LRZ_GROUP not set}"
PROJECT="${REPO_NAME:?REPO_NAME not set}"
TOKEN="${LRZ_GITLAB_PROJECT_TOKEN:?LRZ_GITLAB_PROJECT_TOKEN not set}"
HOST="${LRZ_HOST:?LRZ_HOST not set}"

echo "Checking running/pending CI pipelines for branch: $BRANCH in project: $GROUP/$PROJECT"

# Fetch pipelines for the branch
response=$(curl -s -w "%{http_code}" -o response.json \
  --header "PRIVATE-TOKEN: ${TOKEN}" \
  "https://${HOST}/api/v4/projects/${GROUP}%2F${PROJECT}/pipelines?ref=${BRANCH}&order_by=id&sort=desc")

http_code="${response:(-3)}"

if [[ "$http_code" != "200" ]]; then
  echo "GitLab API request failed with HTTP status $http_code"
  cat response.json
  exit 1
fi

# Ensure response is a JSON array
if ! jq -e 'type=="array"' response.json >/dev/null 2>&1; then
  echo "Unexpected response from GitLab API (not a JSON array)"
  cat response.json
  exit 1
fi

# Extract running/pending pipeline IDs
pipeline_ids=$(jq -r '.[] | select(.status=="running" or .status=="pending") | .id' response.json)
if [ -z "$pipeline_ids" ]; then
  echo "No running/pending CI pipelines to check"
else
  for id in $pipeline_ids; do
    echo "Checking pipeline $id for TRIGGER_SOURCE variable..."
    vars=$(curl -s \
      --header "PRIVATE-TOKEN: ${TOKEN}" \
      "https://${HOST}/api/v4/projects/${GROUP}%2F${PROJECT}/pipelines/$id/variables")

    trigger_source=$(echo "$vars" | jq -r '.[] | select(.key=="TRIGGER_SOURCE") | .value' || true)

    if [ "$trigger_source" == "$PROJECT" ]; then
      echo "Canceling pipeline $id (TRIGGER_SOURCE matches $PROJECT)"
      curl -s --request POST \
        --header "PRIVATE-TOKEN: ${TOKEN}" \
        "https://${HOST}/api/v4/projects/${GROUP}%2F${PROJECT}/pipelines/$id/cancel" >/dev/null
    else
      if [ -z "$trigger_source" ]; then
        echo "Keeping pipeline $id (TRIGGER_SOURCE not set)"
      else
        echo "Keeping pipeline $id (TRIGGER_SOURCE=$trigger_source)"
      fi
    fi
  done
fi

echo "Done."
