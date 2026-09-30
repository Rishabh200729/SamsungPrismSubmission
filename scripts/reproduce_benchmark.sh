#!/usr/bin/env bash
# TRAX - one-command reproduction of Full-Duplex-Bench v3 (install -> configure -> infer -> evaluate).
#
#   ./scripts/reproduce_benchmark.sh              full run, all 100 recordings
#   ./scripts/reproduce_benchmark.sh --smoke      2 recordings: validates the whole pipeline first
#   ./scripts/reproduce_benchmark.sh --eval-only  re-score existing result files, no inference
#   ./scripts/reproduce_benchmark.sh --skip-install   reuse the current .venv as is
#
# Declared agent: TRAX (prism/ + agent/trax_agent.py) on Gemini 2.5 Live
#   (gemini-2.5-flash-native-audio-preview-12-2025, temperature 0.0) through LiveKit Cloud.
#   LK_PROVIDER=gpt_realtime switches to GPT-Realtime (needs OPENAI_API_KEY).
# Needs: Linux, NVIDIA GPU + CUDA (stock FDB-v3 ASR is NeMo parakeet), ffmpeg, git, python3.10+,
#        credentials in .env (see .env.example). Output: results/<run_id>/
set -euo pipefail

FDB_REPO="https://github.com/DanielLin94144/Full-Duplex-Bench"
FDB_COMMIT="3e799c45a045256f47d5f1c9cda90157e2d2ec9e"   # pinned upstream commit (2026-05-20)
FDB_DATA_GDRIVE_ID="1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz"  # released data, from the v3 README

MODE="full"; SKIP_INSTALL=0
for a in "$@"; do
  case "$a" in
    --smoke) MODE="smoke" ;;
    --eval-only) MODE="eval" ;;
    --skip-install) SKIP_INSTALL=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# 1. Install ---------------------------------------------------------------------------------
log "1/6 Python environment"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
if [ "$SKIP_INSTALL" -eq 0 ]; then
  pip install -q -U pip
  pip install -q -r requirements-benchmark.txt
fi
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found (apt install ffmpeg)"
if [ "$MODE" != "eval" ]; then
  command -v nvidia-smi >/dev/null || echo "WARNING: no NVIDIA GPU detected; stock FDB-v3 ASR calls .cuda() and will fail"
fi

# 2. Configure -------------------------------------------------------------------------------
log "2/6 Configuration"
for f in .env .env.local; do
  if [ -f "$f" ]; then set -a; . "./$f"; set +a; fi
done
need() { [ -n "${!1:-}" ] || die "$1 is not set (see .env.example)"; }
export LK_PROVIDER="${LK_PROVIDER:-gemini2_5}"
if [ "$MODE" != "eval" ]; then
  need LIVEKIT_URL; need LIVEKIT_API_KEY; need LIVEKIT_API_SECRET
  case "$LK_PROVIDER" in
    gemini2_5) need GOOGLE_API_KEY; MODEL="${GOOGLE_MODEL:-gemini-2.5-flash-native-audio-preview-12-2025}" ;;
    gpt_realtime) need OPENAI_API_KEY; MODEL="${OPENAI_MODEL:-gpt-4o-realtime-preview}" ;;
    *) die "unsupported LK_PROVIDER=$LK_PROVIDER (TRAX agent supports gemini2_5, gpt_realtime)" ;;
  esac
else
  MODEL="n/a (eval-only)"
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then LLM_FLAG="--use-llm"; JUDGE="gpt-4o LLM judge (stock FDB-v3)"
else LLM_FLAG=""; JUDGE="exact-match only (OPENAI_API_KEY not set; not comparable with the official judge)"; fi
echo "provider=$LK_PROVIDER model=$MODEL judge=$JUDGE mode=$MODE"

# 3. Benchmark code + data -------------------------------------------------------------------
log "3/6 FDB-v3 code (pinned $FDB_COMMIT) and data"
if [ ! -d external/FDB-v3/.git ]; then
  mkdir -p external
  git clone -q "$FDB_REPO" external/FDB-v3
fi
git -C external/FDB-v3 checkout -q "$FDB_COMMIT" || die "cannot check out pinned FDB-v3 commit"
git -C external/FDB-v3 diff --quiet || echo "WARNING: external/FDB-v3 has local modifications (should be untouched upstream)"
V3="external/FDB-v3/v3"
DATA="$V3/fdb_v3_data_released"
if [ ! -d "$DATA" ]; then
  ZIP="${FDB_DATA_ZIP:-$V3/fdb_v3_data.download}"
  if [ ! -f "$ZIP" ]; then
    gdown "https://drive.google.com/uc?id=$FDB_DATA_GDRIVE_ID" -O "$ZIP" \
      || die "data download failed; download it manually from the v3 README and rerun with FDB_DATA_ZIP=/path/to/archive"
  fi
  TMP="$(mktemp -d)"
  python - "$ZIP" "$TMP" <<'PY'
import shutil, sys, tarfile, zipfile
src, dst = sys.argv[1:3]
if zipfile.is_zipfile(src): zipfile.ZipFile(src).extractall(dst)
elif tarfile.is_tarfile(src): tarfile.open(src).extractall(dst)
else: sys.exit("unrecognised archive format: " + src)
PY
  FOUND="$(find "$TMP" -type d -name fdb_v3_data_released | head -1)"
  mv "${FOUND:-$TMP}" "$DATA"
fi
N_REC="$(find "$DATA" -maxdepth 1 -mindepth 1 -type d | grep -cE '_[0-9a-f]{24}$' || true)"
echo "recordings found: $N_REC (expected 100)"
[ "$N_REC" -gt 0 ] || die "no recordings under $DATA"

RESULTS_REL="fdb_v3_data_released"
if [ "$MODE" = "smoke" ]; then
  RESULTS_REL=".smoke_data"
  rm -rf "$V3/$RESULTS_REL"; mkdir -p "$V3/$RESULTS_REL"
  find "$DATA" -maxdepth 1 -mindepth 1 -type d | grep -E '_[0-9a-f]{24}$' | sort | head -2 \
    | while read -r d; do ln -s "$ROOT/$d" "$V3/$RESULTS_REL/$(basename "$d")"; done
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${LK_PROVIDER}_${MODE}"
OUT="results/$RUN_ID"; mkdir -p "$OUT"
echo "run id: $RUN_ID"

# 4. Agent + inference -----------------------------------------------------------------------
AGENT_PID=""
cleanup() {
  if [ -n "$AGENT_PID" ] && kill -0 "$AGENT_PID" 2>/dev/null; then
    pkill -TERM -P "$AGENT_PID" 2>/dev/null || true
    kill -INT "$AGENT_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$AGENT_PID" 2>/dev/null || break; sleep 1; done
    kill -TERM "$AGENT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [ "$MODE" != "eval" ]; then
  log "4/6 Start TRAX agent and stream recordings"
  rm -f /tmp/agent_tool_calls.log /tmp/agent_heartbeat.log      # fresh telemetry; nothing carried over
  python -m agent.run start >"$OUT/agent.log" 2>&1 &
  AGENT_PID=$!
  for _ in $(seq 1 90); do
    grep -q "registered worker" "$OUT/agent.log" 2>/dev/null && break
    kill -0 "$AGENT_PID" 2>/dev/null || { tail -20 "$OUT/agent.log" >&2; die "agent exited during start-up"; }
    sleep 1
  done
  grep -q "registered worker" "$OUT/agent.log" || echo "WARNING: worker registration not confirmed in agent.log; continuing"
  ( cd "$V3" && python run_tool_benchmark_all_released.py --provider "$LK_PROVIDER" \
      --root_dir "$RESULTS_REL" --force ) 2>&1 | tee "$OUT/inference.log"
  cleanup; AGENT_PID=""
else
  log "4/6 Skipped (eval-only)"
fi

# 5. Evaluate (stock FDB-v3 scorers, unmodified) ---------------------------------------------
log "5/6 Evaluation - judge: $JUDGE"
(
  cd "$V3"
  python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir "$RESULTS_REL" \
    --provider "$LK_PROVIDER" --output "$ROOT/$OUT/tool_accuracy_report.json" $LLM_FLAG
  python evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir "$RESULTS_REL" \
    --provider "$LK_PROVIDER" --output "$ROOT/$OUT/pass_rate_report.json" $LLM_FLAG
  python analyze_tool_latency.py --results-dir "$RESULTS_REL" --provider "$LK_PROVIDER" \
    || echo "WARNING: latency analysis failed (non-fatal)"
) 2>&1 | tee "$OUT/evaluation.log"
[ -f "$V3/${LK_PROVIDER}_latency_report.json" ] && cp "$V3/${LK_PROVIDER}_latency_report.json" "$OUT/latency_report.json" || true

# 6. Collect logs and configuration ----------------------------------------------------------
log "6/6 Collect logs -> $OUT"
mkdir -p "$OUT/per_recording"
find "$V3/$RESULTS_REL"/ -maxdepth 2 -name "result_${LK_PROVIDER}.json" 2>/dev/null | while read -r f; do
  d="$(basename "$(dirname "$f")")"; mkdir -p "$OUT/per_recording/$d"; cp "$f" "$OUT/per_recording/$d/result.json"
done
[ -f /tmp/agent_tool_calls.log ] && cp /tmp/agent_tool_calls.log "$OUT/agent_tool_calls.log" || true
pip freeze > "$OUT/pip_freeze.txt"
RUN_ID="$RUN_ID" MODE="$MODE" JUDGE="$JUDGE" MODEL="$MODEL" FDB_COMMIT="$FDB_COMMIT" python - "$OUT" <<'PY'
import datetime, json, os, platform, subprocess, sys
def sh(*a):
    try: return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
cfg = {
    "run_id": os.environ["RUN_ID"], "mode": os.environ["MODE"],
    "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "agent": "TRAX (prism/ + agent/trax_agent.py)", "provider": os.environ["LK_PROVIDER"],
    "model": os.environ["MODEL"], "temperature": 0.0,
    "seed": None, "seed_note": "hosted realtime APIs expose no seed; determinism relies on temperature 0.0",
    "judge": os.environ["JUDGE"], "fdb_commit": os.environ["FDB_COMMIT"],
    "submission_commit": sh("git", "rev-parse", "HEAD"), "python": platform.python_version(),
    "platform": platform.platform(),
}
json.dump(cfg, open(os.path.join(sys.argv[1], "run_config.json"), "w"), indent=2)
PY
log "Done. Reports and logs: $OUT/"
ls -1 "$OUT"
