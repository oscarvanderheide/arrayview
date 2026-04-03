#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# release.sh — bump version, commit, tag, push, create GitHub release
# ---------------------------------------------------------------------------

BUMP="minor"
DRY_RUN=true

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --bump {major,minor,patch}   Version bump type (default: minor)
  --execute                    Actually run (default is dry-run)
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bump)   BUMP="$2"; shift 2 ;;
        --execute) DRY_RUN=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ ! "$BUMP" =~ ^(major|minor|patch)$ ]]; then
    echo "Error: --bump must be major, minor, or patch (got '$BUMP')"
    exit 1
fi

# --- Guard: clean working tree on main ---
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "main" ]]; then
    echo "Error: must be on main (currently on '$branch')"
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree is dirty — commit or stash first"
    exit 1
fi

# --- Bump version ---
uv version --bump "$BUMP"
VERSION=$(uv version --short)
TAG="v${VERSION}"

echo "Version bumped to $VERSION (tag: $TAG)"

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi
    "$@"
}

# --- Commit, tag, push, release ---
run git add pyproject.toml
run git commit -m "release: $TAG"
run git push origin main
run git tag "$TAG"
run git push origin "$TAG"
run gh release create "$TAG" \
    --title "[Pre-release] $TAG" \
    --generate-notes \
    --prerelease

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "(dry-run — re-run with --execute to apply)"
    git checkout pyproject.toml
fi
