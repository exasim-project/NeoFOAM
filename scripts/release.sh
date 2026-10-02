#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense

## Prepare and publish a NeoFOAM release.
##
## The release is a two-step process because `main` is a protected branch and needs an
## approving review, so the merge itself stays a human action:
##
##   scripts/release.sh prepare v0.3.0rc1   bump versions, open the release PR against main
##   <review and merge the PR on GitHub>
##   scripts/release.sh tag v0.3.0rc1       tag the merge commit and create the GitHub release
##
## Pushing the tag is what triggers everything downstream: build_doc.yaml publishes the
## documentation and conda_packages.yaml builds and publishes the conda packages to the
## prefix.dev channel, which is how NeoFOAM is distributed (`pixi add neofoam`). Those
## workflows take the version from the tag name, not from pyproject.toml.
##
##   scripts/release.sh check v0.3.0rc1     run the preflight checks only
##
## Release candidates are versions with a PEP 440 a/b/rc suffix, e.g. 0.3.0rc1. They keep
## the CHANGELOG section marked unreleased and are marked as a prerelease on GitHub, so
## the final 0.3.0 release still dates the same section.
##
## Add --dry-run to any subcommand to print the mutating commands instead of running them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REMOTE="origin"
SOURCE_BRANCH="develop"
TARGET_BRANCH="main"
CONDA_PACKAGE="neofoam"
# Published to a prefix.dev channel and consumed with pixi; see recipe/recipe.yaml.
CONDA_CHANNEL="${NEOFOAM_CONDA_CHANNEL:-greole/exasim-project}"

DRY_RUN=0

err() {
    echo "error: $*" >&2
    exit 1
}

info() { echo "==> $*"; }

# Run a mutating command, or print it under --dry-run.
run() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf 'dry-run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

usage() {
    sed -n 's/^## \{0,1\}//p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

# v0.3.0rc1 -> 0.3.0rc1
strip_v() { echo "${1#v}"; }

# 0.3.0rc1 -> 0.3.0
base_version() { echo "${1%%[a-z]*}"; }

# A prerelease carries an a/b/rc suffix, e.g. 0.3.0rc1.
is_prerelease() { [[ "$1" != "$(base_version "$1")" ]]; }

check_version_format() {
    local version="$1"
    # PEP 440 subset. Conda needs the dash-free form in particular, because a package
    # filename is name-version-build and a dash inside the version breaks that; the v* tag
    # glob in build_doc.yaml matches the corresponding tag.
    if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
        err "'${version}' is not a supported version. Use 0.3.0 or a PEP 440 prerelease like 0.3.0rc1 (no dash)."
    fi
}

require_tools() {
    command -v git >/dev/null || err "git not found"
    command -v gh >/dev/null || err "gh not found; install the GitHub CLI"
    command -v python3 >/dev/null || err "python3 not found"
    command -v curl >/dev/null || err "curl not found"
    gh auth status >/dev/null 2>&1 || err "gh is not authenticated; run 'gh auth login'"
}

fetch() {
    git fetch --quiet "${REMOTE}" "$@" \
        || err "could not fetch ${*} from ${REMOTE}; check your network and git credentials"
}

check_clean_tree() {
    git diff --quiet && git diff --cached --quiet \
        || err "working tree has uncommitted changes; commit or stash them first"
}

check_tag_unused() {
    local tag="$1"
    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
        err "tag ${tag} already exists locally"
    fi
    if gh api "repos/{owner}/{repo}/git/ref/tags/${tag}" >/dev/null 2>&1; then
        err "tag ${tag} already exists on ${REMOTE}"
    fi
}

# A published conda version is permanent: prefix.dev will not accept a re-upload of the same
# name-version-build, so a tag must not be created for a version already in the channel.
check_channel_unused() {
    local version="$1" result
    result="$(
        CONDA_CHANNEL="${CONDA_CHANNEL}" CONDA_PACKAGE="${CONDA_PACKAGE}" VERSION="${version}" \
            python3 - <<'PYEOF'
import json, os, sys, urllib.error, urllib.request

channel = os.environ["CONDA_CHANNEL"]
package = os.environ["CONDA_PACKAGE"]
version = os.environ["VERSION"]

found = []
for subdir in ("linux-64", "osx-arm64", "noarch"):
    url = f"https://repo.prefix.dev/{channel}/{subdir}/repodata.json"
    # An explicit User-Agent is required: prefix.dev answers 403 to urllib's default
    # "Python-urllib/3.x", which would make this guard fail closed on every release.
    request = urllib.request.Request(url, headers={"User-Agent": "neofoam-release-script"})
    try:
        with urllib.request.urlopen(request, timeout=30) as fh:
            data = json.load(fh)
    except urllib.error.HTTPError as exc:
        # A subdir with nothing published in it does not exist yet; that is not an error.
        if exc.code == 404:
            continue
        print(f"UNREACHABLE {subdir} HTTP {exc.code}")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - network failures of any shape
        print(f"UNREACHABLE {subdir} {type(exc).__name__}")
        sys.exit(0)

    packages = {**data.get("packages", {}), **data.get("packages.conda", {})}
    found += [
        name
        for name, meta in packages.items()
        if meta.get("name") == package and meta.get("version") == version
    ]

print("FOUND " + " ".join(sorted(found)) if found else "FREE")  # noqa: E501
PYEOF
    )"

    case "${result}" in
    FREE) ;;
    FOUND*) err "${CONDA_PACKAGE} ${version} is already published in ${CONDA_CHANNEL} and cannot be re-uploaded; pick a new version (${result#FOUND })" ;;
    # Fail closed. A tag and GitHub release are permanent, so proceeding on an inconclusive
    # answer risks burning a version the channel may later refuse. Override deliberately
    # with SKIP_CHANNEL_CHECK=1.
    *)
        if [[ "${SKIP_CHANNEL_CHECK:-0}" == "1" ]]; then
            echo "warning: could not verify ${version} against ${CONDA_CHANNEL} (${result}); continuing because SKIP_CHANNEL_CHECK=1" >&2
        else
            err "could not verify ${version} is unused in ${CONDA_CHANNEL} (${result}); re-run when the channel responds, or set SKIP_CHANNEL_CHECK=1 to override"
        fi
        ;;
    esac
}

check_source_branch() {
    local current
    current="$(git rev-parse --abbrev-ref HEAD)"
    [[ "${current}" == "${SOURCE_BRANCH}" ]] \
        || err "expected to be on ${SOURCE_BRANCH}, but on ${current}"

    fetch "${SOURCE_BRANCH}" "${TARGET_BRANCH}"
    [[ -z "$(git rev-list "${REMOTE}/${SOURCE_BRANCH}..HEAD")" ]] \
        || err "${SOURCE_BRANCH} has commits not pushed to ${REMOTE}"
    [[ -z "$(git rev-list "HEAD..${REMOTE}/${SOURCE_BRANCH}")" ]] \
        || err "${SOURCE_BRANCH} is behind ${REMOTE}/${SOURCE_BRANCH}; pull first"
}

# The CHANGELOG section for this release, used as the GitHub release notes.
changelog_section() {
    local base="$1"
    python3 - "${base}" <<'PY'
import re
import sys
from pathlib import Path

base = sys.argv[1]
text = Path("CHANGELOG.md").read_text(encoding="utf-8")
match = re.search(
    rf"^# Version {re.escape(base)}\b.*?$(.*?)(?=^# Version |\Z)",
    text,
    re.MULTILINE | re.DOTALL,
)
if not match:
    raise SystemExit(f"No '# Version {base}' section found in CHANGELOG.md")
print(match.group(1).strip())
PY
}

cmd_check() {
    local version="$1" tag="v$1"
    require_tools
    check_version_format "${version}"
    check_clean_tree
    check_tag_unused "${tag}"
    check_channel_unused "${version}"
    changelog_section "$(base_version "${version}")" >/dev/null
    info "preflight checks passed for ${tag}"
}

cmd_prepare() {
    local version="$1" tag="v$1"
    local base branch today
    base="$(base_version "${version}")"
    branch="release/${tag}"

    check_source_branch
    cmd_check "${version}"

    today="$(date +%Y-%m-%d)"

    info "creating ${branch}"
    run git switch --create "${branch}"

    info "stamping pyproject.toml to ${version}"
    run python3 scripts/set_package_version.py "${version}"

    info "updating CITATION.cff"
    run python3 - "${tag}" "${today}" <<'PY'
import re
import sys
from pathlib import Path

tag, today = sys.argv[1], sys.argv[2]
path = Path("CITATION.cff")
text = path.read_text(encoding="utf-8")
text, n = re.subn(r"(?m)^version: .*$", f"version: {tag}", text)
if n != 1:
    raise SystemExit("Could not update 'version:' in CITATION.cff")
text, n = re.subn(r"(?m)^date-released: .*$", f"date-released: '{today}'", text)
if n != 1:
    raise SystemExit("Could not update 'date-released:' in CITATION.cff")
path.write_text(text, encoding="utf-8")
PY

    # A prerelease keeps the CHANGELOG section open; only a final release dates it.
    if is_prerelease "${version}"; then
        info "keeping the CHANGELOG ${base} section marked unreleased (${version} is a prerelease)"
    else
        info "dating the CHANGELOG ${base} section"
        run python3 - "${base}" <<'PY'
import re
import sys
from datetime import date
from pathlib import Path

base = sys.argv[1]
path = Path("CHANGELOG.md")
text = path.read_text(encoding="utf-8")
text, n = re.subn(
    rf"(?m)^# Version {re.escape(base)} \(unreleased\)$",
    f"# Version {base} ({date.today():%Y.%m.%d})",
    text,
)
if n != 1:
    raise SystemExit(f"Could not date the '# Version {base} (unreleased)' header")
path.write_text(text, encoding="utf-8")
PY
    fi

    echo
    info "review the staged version bump:"
    run git --no-pager diff -- pyproject.toml CITATION.cff CHANGELOG.md

    run git commit --all --message "release: ${tag}"
    run git push --set-upstream "${REMOTE}" "${branch}"

    info "opening the release PR against ${TARGET_BRANCH}"
    run gh pr create \
        --base "${TARGET_BRANCH}" \
        --head "${branch}" \
        --title "Release ${tag}" \
        --body "Release ${tag}.

Merging this PR brings \`${TARGET_BRANCH}\` up to \`${SOURCE_BRANCH}\` and stamps the release version.
After it is merged, run \`scripts/release.sh tag ${tag}\` to tag the merge commit and publish."

    echo
    info "next: get the PR approved and merged, then run: scripts/release.sh tag ${tag}"
}

cmd_tag() {
    local version="$1" tag="v$1"
    local base sha notes prerelease_flag

    require_tools
    check_version_format "${version}"
    check_clean_tree
    check_channel_unused "${version}"

    base="$(base_version "${version}")"
    fetch "${TARGET_BRANCH}"
    sha="$(git rev-parse "${REMOTE}/${TARGET_BRANCH}")"

    # Guard against tagging a main that never received the release commit.
    local main_pyproject main_version
    main_pyproject="$(git show "${sha}:pyproject.toml" 2>/dev/null)" || err \
        "${REMOTE}/${TARGET_BRANCH} has no pyproject.toml; is the release PR merged?"
    main_version="$(echo "${main_pyproject}" | sed -n 's/^version = "\(.*\)"$/\1/p' | head -1)"
    [[ "${main_version}" == "${version}" ]] || err \
        "${REMOTE}/${TARGET_BRANCH} has pyproject version '${main_version}', expected '${version}'; is the release PR merged?"

    notes="$(changelog_section "${base}")"

    prerelease_flag=()
    if is_prerelease "${version}"; then
        prerelease_flag=(--prerelease)
        info "${version} is a prerelease; the GitHub release will be marked as such"
    fi

    # Resumable: the tag is pushed before the GitHub release is created, so a transient
    # failure in `gh release create` used to leave `tag` unable to re-run — the tag now
    # existed and check_tag_unused aborted. Accept an existing tag only when it already
    # points at the SHA we are about to tag, and skip straight to creating the release.
    local remote_tag remote_peeled
    remote_tag="$(git ls-remote --tags "${REMOTE}" "refs/tags/${tag}" | cut -f1)"
    if [[ -n "${remote_tag}" ]]; then
        # ^{} is the peeled ref: the commit an annotated tag points at. Read it from the
        # remote rather than locally, so `tag` can be resumed from a different checkout.
        remote_peeled="$(git ls-remote --tags "${REMOTE}" "refs/tags/${tag}^{}" | cut -f1)"
        [[ -n "${remote_peeled}" ]] || remote_peeled="${remote_tag}"
        if [[ "${remote_peeled}" == "${sha}" ]]; then
            info "tag ${tag} already points at ${sha}; resuming"
        else
            err "tag ${tag} already exists and does not point at ${sha}; delete it or pick a new version"
        fi
    else
        if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
            err "tag ${tag} exists locally but not on ${REMOTE}; delete it or push it deliberately"
        fi
        info "tagging ${sha} as ${tag}"
        run git tag --annotate "${tag}" "${sha}" --message "NeoFOAM ${tag}"
        run git push "${REMOTE}" "refs/tags/${tag}"
    fi

    if gh release view "${tag}" >/dev/null 2>&1; then
        info "GitHub release ${tag} already exists; nothing further to do"
        return 0
    fi

    info "creating the GitHub release"
    run gh release create "${tag}" \
        --title "NeoFOAM ${tag}" \
        --notes "${notes}" \
        --verify-tag \
        "${prerelease_flag[@]}"

    echo
    info "the tag push triggers the documentation workflow; watch it with:"
    echo "  gh run list --workflow build_doc.yaml"
}

main() {
    local args=()
    for arg in "$@"; do
        case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h | --help) usage 0 ;;
        -*) err "unknown option ${arg}" ;;
        *) args+=("${arg}") ;;
        esac
    done

    [[ ${#args[@]} -eq 2 ]] || usage 1

    local subcommand version
    subcommand="${args[0]}"
    version="$(strip_v "${args[1]}")"

    case "${subcommand}" in
    check) cmd_check "${version}" ;;
    prepare) cmd_prepare "${version}" ;;
    tag) cmd_tag "${version}" ;;
    *) err "unknown subcommand '${subcommand}'" ;;
    esac
}

main "$@"
