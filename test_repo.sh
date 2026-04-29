#!/usr/bin/env bash
# =============================================================================
# test_repo.sh — sabench local CI gate
#
# Mirrors every check run by .github/workflows/ci.yml so that a clean run here
# guarantees a clean run on GitHub Actions.
#
# USAGE
#   ./test_repo.sh                  # run all stages (prepare + check)
#   ./test_repo.sh --clean          # nuke env + caches, then run all stages
#   ./test_repo.sh --prepare-only   # format / auto-fix code only, no checks
#   ./test_repo.sh --check-only     # lint/test/build only, no auto-fixing
#   ./test_repo.sh --no-build       # skip the sdist/wheel build step
#   ./test_repo.sh --no-notebook    # skip notebook execution
#
# PREREQUISITES
#   pixi (https://pixi.sh)  — install with:
#     curl -fsSL https://pixi.sh/install.sh | bash
#
# EXIT CODES
#   0  all checks passed
#   1  one or more checks failed  (details printed inline)
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}  $*"; }
header(){ echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; \
          echo -e "${BOLD}${CYAN}  $*${RESET}"; \
          echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; }

# ── Flag parsing ─────────────────────────────────────────────────────────────
DO_CLEAN=false
DO_PREPARE=true
DO_CHECK=true
DO_BUILD=true
DO_NOTEBOOK=true

for arg in "$@"; do
  case "$arg" in
    --clean)         DO_CLEAN=true ;;
    --prepare-only)  DO_CHECK=false; DO_BUILD=false; DO_NOTEBOOK=false ;;
    --check-only)    DO_PREPARE=false ;;
    --no-build)      DO_BUILD=false ;;
    --no-notebook)   DO_NOTEBOOK=false ;;
    --help|-h)
      sed -n '/^# USAGE/,/^# PREREQUISITES/p' "$0" | head -n -1
      exit 0 ;;
    *)
      echo "Unknown flag: $arg  (use --help)"; exit 1 ;;
  esac
done

# ── Ensure we're at the repo root ────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "pyproject.toml" ]]; then
  fail "Not in the repo root (pyproject.toml not found). Aborting."
  exit 1
fi

# ── Track failures ────────────────────────────────────────────────────────────
FAILURES=()
record() {
  local label=$1; shift
  if "$@"; then
    ok "$label"
  else
    fail "$label"
    FAILURES+=("$label")
    return 1
  fi
}

# ── Require pixi ─────────────────────────────────────────────────────────────
if ! command -v pixi &>/dev/null; then
  fail "pixi not found in PATH."
  echo
  echo "  Install with:"
  echo "    curl -fsSL https://pixi.sh/install.sh | bash"
  echo "  Then restart your shell and re-run this script."
  exit 1
fi
PIXI="pixi run -e ci"
PIXI_NOTEBOOK="pixi run"

# =============================================================================
# STAGE 0 — CLEAN
# =============================================================================
if $DO_CLEAN; then
  header "STAGE 0 — Clean environment & caches"

  info "Removing pixi environment (.pixi/)"
  rm -rf .pixi

  info "Removing build artefacts (dist/, build/, *.egg-info)"
  find . -type d \( -name "dist" -o -name "build" -o -name "*.egg-info" \) \
    -not -path "./.git/*" -not -path "./.pixi/*" \
    -prune -exec rm -rf {} +

  info "Removing Python caches"
  find . -type d -name "__pycache__" -not -path "./.pixi/*" | xargs rm -rf
  find . -name "*.pyc" -o -name "*.pyo" | xargs rm -f 2>/dev/null || true

  info "Removing test/coverage artefacts"
  rm -f .coverage coverage.xml .coverage.*
  rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov

  info "Removing transient bundle/platform artefacts"
  find . -type d -name "__MACOSX" \
    -not -path "./.git/*" -not -path "./.pixi/*" \
    -prune -exec rm -rf {} +
  find . -type f \( -name ".DS_Store" -o -name "._*" -o -name "diff.txt" -o -name "*.zip" -o -name "*.tar.gz" \) \
    -not -path "./.git/*" -not -path "./.pixi/*" \
    -delete

  info "Removing generated notebook outputs"
  rm -f notebooks/demo_executed.ipynb notebooks/demo.html
  rm -f notebooks/*.png

  ok "Clean complete"
fi

# =============================================================================
# STAGE 1 — INSTALL / SYNC ENVIRONMENT
# =============================================================================
header "STAGE 1 — Install / sync pixi environment"

info "Running: pixi install --locked -e ci"
pixi install --locked -e ci
ok "pixi ci environment ready"

if $DO_NOTEBOOK; then
  info "Running: pixi install --locked (default environment for notebook smoke test)"
  pixi install --locked
  ok "pixi default environment ready"
fi

info "Installing package in editable mode into the ci environment"
$PIXI pip install -e ".[dev]" --quiet
ok "sabench installed (editable)"

# =============================================================================
# STAGE 2 — PREPARE  (auto-fix: format, sort imports)
# =============================================================================
if $DO_PREPARE; then
  header "STAGE 2 — Prepare (auto-format & auto-fix)"

  # ── ruff: fix safe lint violations + sort imports ─────────────────────────
  info "ruff --fix: auto-fixing safe violations and sorting imports"
  $PIXI ruff check sabench tests --fix --select I,UP,C4,F401 || true
  ok "ruff auto-fix applied"

  # ── ruff format: reformat all Python source ───────────────────────────────
  info "ruff format: reformatting sabench/"
  record "ruff format (source)" $PIXI ruff format sabench tests

  # ── ruff format: reformat test suite ─────────────────────────────────────
  info "ruff format: reformatting tests (for consistency)"
  $PIXI ruff format tests || true

  # ── Strip notebook outputs (keeps git diffs clean) ───────────────────────
  if command -v nbstripout &>/dev/null || $PIXI python -c "import nbstripout" 2>/dev/null; then
    info "Stripping notebook outputs (nbstripout)"
    find notebooks -name "*.ipynb" | xargs $PIXI nbstripout
    ok "Notebook outputs stripped"
  else
    warn "nbstripout not available — skipping (install pre-commit hooks to automate)"
  fi

  ok "Prepare stage complete"
fi

# =============================================================================
# STAGE 3 — CHECK  (mirrors GitHub Actions CI exactly)
# =============================================================================
if $DO_CHECK; then
  header "STAGE 3 — Check (mirrors GitHub Actions CI)"

  # ── 3a. ruff lint ────────────────────────────────────────────────────────
  info "ruff check sabench  (lint — mirrors 'lint' job)"
  record "ruff lint" $PIXI lint

  # ── 3b. ruff format --check ───────────────────────────────────────────────
  info "ruff format --check sabench  (format guard — mirrors 'lint' job)"
  record "ruff format --check" $PIXI fmt-check

  # ── 3c. mypy ──────────────────────────────────────────────────────────────
  info "mypy sabench  (type check — mirrors 'typecheck' job)"
  record "mypy" $PIXI typecheck

  # ── 3d. pytest + coverage ─────────────────────────────────────────────────
  info "pytest with coverage  (mirrors 'test' matrix job)"
  record "pytest + coverage" $PIXI test-cov

  # ── 3e. YAML / TOML / JSON validation (pre-commit hooks do this on CI) ───
  info "Validating pyproject.toml"
  record "pyproject.toml valid" $PIXI python -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    d = tomllib.load(f)
assert d['project']['name'] == 'sabench', 'Bad project name'
assert d['project']['requires-python'], 'Missing requires-python'
print('  name:', d['project']['name'], '  version:', d['project']['version'])
" 2>/dev/null || \
  record "pyproject.toml valid" $PIXI python -c "
import tomli
with open('pyproject.toml', 'rb') as f:
    d = tomli.load(f)
assert d['project']['name'] == 'sabench'
print('  name:', d['project']['name'], '  version:', d['project']['version'])
" 2>/dev/null || \
  # fallback: basic syntax check via python-toml parse
  record "pyproject.toml valid (fallback)" $PIXI python -c "
data = open('pyproject.toml').read()
assert '[project]' in data and 'hatchling' in data, 'pyproject structure error'
print('  pyproject.toml structure OK (fallback check)')
"

  info "Validating CITATION.cff"
  record "CITATION.cff exists" test -f CITATION.cff

  info "Validating .zenodo.json"
  record ".zenodo.json valid" $PIXI python -c "
import json
with open('.zenodo.json') as f:
    d = json.load(f)
assert 'title' in d and 'creators' in d, 'Missing required zenodo fields'
print('  .zenodo.json OK')
"

  # ── 3f. trailing whitespace check ─────────────────────────────────────────
  info "Checking for trailing whitespace in Python source"
  TW_COUNT=$( (grep -rl " $" sabench/ tests/ --include="*.py" 2>/dev/null || true) | wc -l | tr -d '[:space:]' )
  if [[ "$TW_COUNT" -gt 0 ]]; then
    fail "Trailing whitespace found in $TW_COUNT files"
    FAILURES+=("trailing whitespace")
  else
    ok "trailing whitespace: none"
  fi

  # ── 3g. check for merge conflict markers ──────────────────────────────────
  info "Checking for merge conflict markers"
  MC=$(grep -RIl --include="*.py" -E '^<<<<<<< |^=======$|^>>>>>>> ' sabench/ tests/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "${MC:-0}" -ne 0 ]; then
    fail "merge conflict markers found"
    grep -RIn --include="*.py" -E '^<<<<<<< |^=======$|^>>>>>>> ' sabench/ tests/ || true
    FAILURES+=("merge conflict markers")
  else
    ok "merge conflict markers: none"
  fi

  # ── 3h. notebook execution (light smoke test) ─────────────────────────────
  if $DO_NOTEBOOK; then
    info "Running demo notebook end-to-end (smoke test)"
    (
      cd "$REPO_ROOT"
      MPLBACKEND=Agg $PIXI_NOTEBOOK python - << 'NBRUN'
import json, sys, traceback
with open("notebooks/demo.ipynb") as f:
    nb = json.load(f)
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
g = {}
failed = []
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    try:
        exec(src, g)
    except Exception as e:
        failed.append((i, type(e).__name__, str(e)))
if failed:
    for i, ename, emsg in failed:
        print(f"  Cell {i} FAILED: {ename}: {emsg}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"  All {len(cells)} notebook cells executed successfully")
NBRUN
    ) && ok "notebook smoke test" || { fail "notebook smoke test"; FAILURES+=("notebook smoke test"); }
    # Clean up generated PNGs from notebook run
    rm -f notebooks/*.png
  fi
fi

# =============================================================================
# STAGE 4 — BUILD (mirrors 'build' job)
# =============================================================================
if $DO_BUILD; then
  header "STAGE 4 — Build distribution (mirrors 'build' job)"

  info "Cleaning old build artefacts"
  find . -type d \( -name "dist" -o -name "build" -o -name "*.egg-info" \) \
    -not -path "./.git/*" -not -path "./.pixi/*" \
    -prune -exec rm -rf {} +

  info "Building sdist and wheel"
  record "python -m build" $PIXI build

  info "twine check --strict dist/*"
  record "twine check" $PIXI twine-check

  if [[ -d dist ]]; then
    echo
    info "Built artefacts:"
    ls -lh dist/
  fi
fi

# =============================================================================
# STAGE 5 — PRE-COMMIT HOOKS (optional, requires hooks installed)
# =============================================================================
header "STAGE 5 — Pre-commit hooks"

if $PIXI python -c "import pre_commit" 2>/dev/null; then
  if [[ -f .git/hooks/pre-commit ]]; then
    info "Running pre-commit on all files"
    record "pre-commit --all-files" $PIXI pre-commit run --all-files
  else
    warn "Pre-commit hooks not installed in .git/hooks."
    warn "Run:  pixi run pre-commit-install"
    warn "Then commit files normally — hooks fire automatically."
  fi
else
  warn "pre-commit not installed in this env. Run: pixi run pre-commit-install"
fi

# =============================================================================
# SUMMARY
# =============================================================================
header "SUMMARY"

if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}"
  echo "  ✓  All checks passed."
  echo "  ✓  This repo is ready to push to GitHub."
  echo -e "${RESET}"
  echo "  Next steps:"
  echo "    git add -p && git commit -m 'your message'"
  echo "    git push"
  echo
  echo "  To publish a release:"
  echo "    git tag v0.3.0 && git push --tags"
  echo "    (GitHub Actions will build and publish to PyPI automatically)"
  exit 0
else
  echo -e "${RED}${BOLD}"
  echo "  ✗  ${#FAILURES[@]} check(s) failed:"
  for f in "${FAILURES[@]}"; do
    echo "       • $f"
  done
  echo -e "${RESET}"
  echo "  Fix the issues above and re-run:  ./test_repo.sh --check-only"
  exit 1
fi
