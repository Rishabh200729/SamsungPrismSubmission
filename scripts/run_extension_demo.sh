#!/usr/bin/env bash
# Extension demo: in-car navigation agent with mid-route destination correction.
#   ./scripts/run_extension_demo.sh          offline, deterministic, no API keys (default)
#   ./scripts/run_extension_demo.sh test     extension unit tests
#   ./scripts/run_extension_demo.sh live     LiveKit voice agent (needs .env with LiveKit + Gemini keys)
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-offline}" in
  offline) python -m agent.extension_incar --demo ;;
  test)    python -m unittest prism.tests.test_extension_incar -v ;;
  live)    python -m agent.extension_incar dev ;;
  *) echo "usage: $0 [offline|test|live]" >&2; exit 2 ;;
esac
