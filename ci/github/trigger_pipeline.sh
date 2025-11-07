#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
# -----------------------------------------------------------------------------
# Trigger a new LRZ GitLab CI pipeline on TUM COMA cluster for a given branch.
#
# Uses environment variables defined in the GitHub Actions job:
#   LRZ_GROUP, LRZ_HOST, REPO_NAME, LRZ_GITLAB_TRIGGER_TOKEN
#
# Usage:
#   ./ci/github/trigger_pipeline.sh <branch>
# -----------------------------------------------------------------------------

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <branch>"
  exit 1
fi

BRANCH=$1
GROUP="${LRZ_GROUP:?LRZ_GROUP not set}"
PROJECT="${REPO_NAME:?REPO_NAME not set}"
TRIGGER_TOKEN="${LRZ_GITLAB_TRIGGER_TOKEN:?LRZ_GITLAB_TRIGGER_TOKEN not set}"
HOST="${LRZ_HOST:?LRZ_HOST not set}"
shift 1
VARIABLES="$@"     # Optional extra variables in the form: "variables[KEY]=VALUE"

# -----------------------------------------------------------------------------
# Determine NeoN branch on LRZ GitLab
# -----------------------------------------------------------------------------
NEON_PROJECT="NeoN"

echo "Checking if NeoN branch '${BRANCH}' exists on LRZ GitLab..."

# Encode branch name for safe URL use
ENCODED_BRANCH=$(printf '%s' "$BRANCH" | jq -sRr @uri)

# Query the NeoN repo branch endpoint on LRZ GitLab
status_code=$(curl -s -o /dev/null -w "%{http_code}" \
  "https://${HOST}/api/v4/projects/${GROUP}%2F${NEON_PROJECT}/repository/branches/${ENCODED_BRANCH}")


if [ "$status_code" -eq 200 ]; then
  NEON_BRANCH="$BRANCH"
  echo "NeoN branch '${BRANCH}' exists on LRZ GitLab."
else
  NEON_BRANCH="develop"
  echo "NeoN branch '${BRANCH}' does not exist on LRZ GitLab. Using '${NEON_BRANCH}'."
fi

# -----------------------------------------------------------------------------
# Trigger FoamAdapter pipeline
# -----------------------------------------------------------------------------
echo "Triggering a new CI pipeline on branch $BRANCH in project: $GROUP/$PROJECT"
echo "Use NeoN branch: $NEON_BRANCH"

# Prepare curl form data for variables
FORM_DATA="--form ref=$BRANCH --form token=$TRIGGER_TOKEN"
FORM_DATA="$FORM_DATA --form variables[NEON_BRANCH]=$NEON_BRANCH"

for var in $VARIABLES; do
  FORM_DATA="$FORM_DATA --form $var"
done

response=$(curl -s --request POST $FORM_DATA \
  "https://${HOST}/api/v4/projects/${GROUP}%2F${PROJECT}/trigger/pipeline")

echo "$response" | jq .

pipeline_id=$(echo "$response" | jq -r '.id')
if [ "$pipeline_id" = "null" ] || [ -z "$pipeline_id" ]; then
  echo "Failed to trigger LRZ CI pipeline"
  exit 1
fi

echo "Successfully triggered pipeline: $pipeline_id"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "pipeline_id=$pipeline_id" >> "$GITHUB_OUTPUT"
fi