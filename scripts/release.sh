#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# release.sh — bump version, commit, tag, push, create GitHub release
# ---------------------------------------------------------------------------

BUMP="minor"
DRY_RUN=true
NO_AI=false
AI_TOOL="codex"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --bump {major,minor,patch}   Version bump type (default: minor)
  --ai {claude,codex}          AI tool for release notes (default: codex)
  --execute                    Actually run (default is dry-run)
  --no-ai                      Skip AI release notes, use GitHub's --generate-notes
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bump)   BUMP="$2"; shift 2 ;;
        --ai)     AI_TOOL="$2"; shift 2 ;;
        --execute) DRY_RUN=false; shift ;;
        --no-ai)  NO_AI=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ ! "$BUMP" =~ ^(major|minor|patch)$ ]]; then
    echo "Error: --bump must be major, minor, or patch (got '$BUMP')"
    exit 1
fi

if [[ ! "$AI_TOOL" =~ ^(claude|codex)$ ]]; then
    echo "Error: --ai must be claude or codex (got '$AI_TOOL')"
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

uv lock --quiet

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi
    "$@"
}

# --- Generate release notes ---
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude 2>/dev/null || echo claude)}"
CODEX_BIN="${CODEX_BIN:-$(command -v codex 2>/dev/null || echo codex)}"
PREV_TAG=$(git describe --tags --abbrev=0 HEAD 2>/dev/null || echo "")
NOTES=""

if [[ -n "$PREV_TAG" ]]; then
    COMMITS=$(git log "${PREV_TAG}..HEAD" --oneline)
else
    COMMITS=$(git log --oneline -20)
fi

if [[ "$NO_AI" == false ]]; then
    PROMPT="You are writing release notes for arrayview $TAG (a Python array/image viewer).

Here are the commits since the last release ($PREV_TAG):

$COMMITS

Write concise, user-friendly release notes in this exact format:

## What's new in $TAG

- **Feature name**: one-sentence description

Rules:
- 5-10 bullet points max — group related commits into one bullet
- Write for end-users, not developers (no commit hashes, no file names)
- Use past tense (\"added\", \"fixed\", \"improved\")
- Skip pure refactors/docs unless they affect user experience
- Bold the feature name, keep the description to one sentence"

    case "$AI_TOOL" in
        claude)
            if command -v "$CLAUDE_BIN" &>/dev/null; then
                echo "Generating release notes with Claude..."
                NOTES=$("$CLAUDE_BIN" -p "$PROMPT" 2>/dev/null) || true
            fi
            ;;
        codex)
            if command -v "$CODEX_BIN" &>/dev/null; then
                echo "Generating release notes with Codex..."
                tmpfile=$(mktemp)
                errfile=$(mktemp)
                trap 'rm -f "$tmpfile" "$errfile"' EXIT
                if printf '%s\n' "$PROMPT" | "$CODEX_BIN" exec --output-last-message "$tmpfile" - >/dev/null 2>"$errfile"; then
                    NOTES=$(<"$tmpfile")
                elif [[ -s "$errfile" ]]; then
                    echo "Codex release notes failed:"
                    sed 's/^/  /' "$errfile"
                fi
            fi
            ;;
    esac
fi

if [[ -z "$NOTES" ]]; then
    if [[ "$NO_AI" == false ]]; then
        echo "AI notes unavailable, falling back to commit-based notes"
    fi
    NOTES="## What's new in $TAG"
    if [[ -n "$COMMITS" ]]; then
        while IFS= read -r commit; do
            [[ -z "$commit" ]] && continue
            message="${commit#* }"
            NOTES+=$'\n'
            NOTES+="- ${message}"
        done <<< "$COMMITS"
    else
        NOTES+=$'\n'
        NOTES+="- Maintenance release."
    fi
fi

if [[ "$DRY_RUN" == true && -n "$NOTES" ]]; then
    echo ""
    echo "--- Release notes preview ---"
    echo "$NOTES"
    echo "-----------------------------"
    echo ""
fi

# --- Commit, tag, push, release ---
run git add pyproject.toml uv.lock
run git commit -m "release: $TAG"
run git push origin main
run git tag "$TAG"
run git push origin "$TAG"

run gh release create "$TAG" \
    --title "$TAG" \
    --notes "$NOTES" \
    --prerelease

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "(dry-run — re-run with --execute to apply)"
    git checkout pyproject.toml uv.lock
fi
