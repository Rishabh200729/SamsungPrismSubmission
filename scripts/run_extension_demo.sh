#!/usr/bin/env bash
# Extension demo: in-car navigation agent with mid-route destination correction.
#
#   ./scripts/run_extension_demo.sh          offline, deterministic, no API keys (default)
#   ./scripts/run_extension_demo.sh demo     same as above (alias)
#   ./scripts/run_extension_demo.sh test     extension unit tests
#   ./scripts/run_extension_demo.sh live     LiveKit voice agent (needs .env with LiveKit + Gemini keys)
set -euo pipefail

# Move to repo root regardless of where the script is called from.
cd "$(dirname "$0")/.."

# Auto-activate virtual environment if present.
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Ensure `rich` is available for the beautiful terminal dashboard (soft-install).
python -c "import rich" 2>/dev/null || python -m pip install rich --quiet

case "${1:-demo}" in
  demo|offline)
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║       TRAX In-Car Navigation — Offline Deterministic Demo           ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    python -m agent.extension_incar --demo
    ;;
  test)
    echo ""
    echo "Running TRAX Extension Unit Tests..."
    python -m unittest prism.tests.test_extension_incar -v
    ;;
  live)
    echo ""
    echo "Starting TRAX In-Car Navigation — Live LiveKit Voice Agent..."
    echo "(Requires LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, GOOGLE_API_KEY in .env)"
    echo ""
    python -m agent.extension_incar dev
    ;;
  *)
    echo "usage: $0 [demo|test|live]" >&2
    echo ""
    echo "  demo   - Run the offline deterministic walkthrough (default, no API keys needed)"
    echo "  test   - Run all extension unit tests"
    echo "  live   - Start the live LiveKit voice agent"
    exit 2
    ;;
esac
