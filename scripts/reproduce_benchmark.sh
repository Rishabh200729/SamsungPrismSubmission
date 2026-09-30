#!/usr/bin/env bash
# TRAX - one-command reproduction of Full-Duplex-Bench v3 without modifying the pinned source checkout or data.
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
#        credentials in .env (see .env.example).

set -euo pipefail

FDB_REPO="https://github.com/DanielLin94144/Full-Duplex-Bench"
FDB_COMMIT="3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
FDB_DATA_GDRIVE_ID="1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz"
MODE="full"; SKIP_INSTALL=0; LOCAL_EXACT_MATCH="${LOCAL_EXACT_MATCH:-0}"

for arg in "$@"; do
  case "$arg" in
    --smoke) MODE="smoke" ;;
    --eval-only) MODE="eval" ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --local-exact-match) LOCAL_EXACT_MATCH=1 ;;
    -h|--help) sed -n '1,17p' "$0"; exit 0 ;;
    *) echo "error: unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
ROOT="$(pwd)"; PYTHON_BIN="${PYTHON_BIN:-python3}"
case "$(uname -s)" in
  Linux) ;;
  *)
    echo "error: this benchmark runner requires Linux or WSL2 with a Linux distribution." >&2
    echo "       Install Ubuntu with: wsl --install -d Ubuntu" >&2
    exit 1 ;;
esac
command -v "$PYTHON_BIN" >/dev/null || { echo "error: Python 3 is required" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { echo "error: $*" >&2; exit 1; }
need() { [ -n "${!1:-}" ] || die "$1 is required; copy .env.example to .env or .env.local"; }

"$PYTHON_BIN" - <<'PY' || die "Python 3.10 through 3.12 is required"
import sys
raise SystemExit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))
PY

log "1/7 Python environment"
[ -d .venv ] || "$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
if [ "$SKIP_INSTALL" -eq 0 ]; then
  python -m pip install --upgrade pip
  python -m pip install --requirement requirements-benchmark.txt
fi
command -v ffmpeg >/dev/null || die "ffmpeg is required"
if [ "$MODE" != "eval" ]; then command -v nvidia-smi >/dev/null || die "an NVIDIA GPU is required by stock FDB-v3 ASR"; fi

log "2/7 Configuration"
for env_file in .env .env.local; do
  if [ -f "$env_file" ]; then set -a; eval "$(tr -d '\r' < "$env_file")"; set +a; fi
done
export LK_PROVIDER="${LK_PROVIDER:-gemini2_5}"
if [ "$MODE" != "eval" ]; then
  need LIVEKIT_URL; need LIVEKIT_API_KEY; need LIVEKIT_API_SECRET
  case "$LK_PROVIDER" in
    gemini2_5) need GOOGLE_API_KEY; MODEL="${GOOGLE_MODEL:-gemini-2.5-flash-native-audio-preview-12-2025}" ;;
    gpt_realtime) need OPENAI_API_KEY; MODEL="${OPENAI_MODEL:-gpt-4o-realtime-preview}" ;;
    *) die "unsupported LK_PROVIDER=$LK_PROVIDER (TRAX agent supports gemini2_5, gpt_realtime)" ;;
  esac
else MODEL="n/a (evaluation only)"; fi
if [ "$LOCAL_EXACT_MATCH" -eq 1 ]; then
  JUDGE="local exact-match diagnostic only"; LLM_FLAG=""
else
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "error: OPENAI_API_KEY is not set in .env." >&2
    echo "       The official FDB-v3 benchmark scorer requires an OpenAI API key for its LLM judge." >&2
    echo "       If you don't have an OpenAI key, run with local exact-match diagnostic by adding the flag:" >&2
    echo "         --local-exact-match" >&2
    exit 1
  fi
  JUDGE="official LLM judge"; LLM_FLAG="--use-llm"
fi
echo "provider=$LK_PROVIDER model=$MODEL judge=$JUDGE mode=$MODE"

log "3/7 Pinned FDB-v3 source and data"
FDB_ROOT="$ROOT/external/FDB-v3"
if [ ! -d "$FDB_ROOT/.git" ]; then mkdir -p "$ROOT/external"; git clone "$FDB_REPO" "$FDB_ROOT"; fi
git -C "$FDB_ROOT" fetch --quiet origin
git -C "$FDB_ROOT" checkout --quiet "$FDB_COMMIT"
git -C "$FDB_ROOT" diff --quiet || die "pinned FDB checkout has tracked modifications"
SOURCE_DATA="$FDB_ROOT/v3/fdb_v3_data_released"
if [ ! -d "$SOURCE_DATA" ]; then
  ARCHIVE="${FDB_DATA_ZIP:-$ROOT/external/fdb_v3_data.download}"
  if [ ! -f "$ARCHIVE" ]; then
    gdown "https://drive.google.com/uc?id=$FDB_DATA_GDRIVE_ID" --output "$ARCHIVE" || die "FDB data download failed; set FDB_DATA_ZIP"
  fi
  TEMP_EXTRACT="$(mktemp -d)"
  python - "$ARCHIVE" "$TEMP_EXTRACT" "$SOURCE_DATA" <<'PY'
import pathlib, shutil, sys, tarfile, zipfile
source, destination, target = sys.argv[1:]
if zipfile.is_zipfile(source): zipfile.ZipFile(source).extractall(destination)
elif tarfile.is_tarfile(source): tarfile.open(source).extractall(destination)
else: raise SystemExit("unrecognised FDB data archive")
found = next((p for p in pathlib.Path(destination).rglob('fdb_v3_data_released') if p.is_dir()), None)
if found is None: raise SystemExit("archive does not contain fdb_v3_data_released")
shutil.move(str(found), target)
PY
fi
[ -d "$SOURCE_DATA" ] || die "FDB data directory is missing"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_${LK_PROVIDER}_${MODE}"
OUT="$ROOT/benchmark-runs/$RUN_ID"
mkdir -p "$OUT"/{config,logs,results,telemetry}
WORKTREE="$OUT/fdb-worktree"
git -C "$FDB_ROOT" worktree add --detach "$WORKTREE" "$FDB_COMMIT"
WORK_V3="$WORKTREE/v3"; RUN_DATA="$WORK_V3/fdb_v3_data_released"
mkdir -p "$RUN_DATA"
limit=0; [ "$MODE" = "smoke" ] && limit=2
count=0
while IFS= read -r -d '' source_dir; do
  name="$(basename "$source_dir")"; mkdir -p "$RUN_DATA/$name"
  ln -s "$source_dir/input.wav" "$RUN_DATA/$name/input.wav"
  [ -f "$source_dir/metadata.json" ] && cp "$source_dir/metadata.json" "$RUN_DATA/$name/metadata.json"
  count=$((count + 1)); [ "$limit" -gt 0 ] && [ "$count" -ge "$limit" ] && break
done < <(find "$SOURCE_DATA" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
[ "$count" -gt 0 ] || die "no released FDB recordings found"
[ "$MODE" != "full" ] || [ "$count" -eq 100 ] || die "expected 100 recordings, found $count"

log "4/7 Run-scoped telemetry and LiveKit agent"
TELEMETRY="$OUT/telemetry/tool_calls.jsonl"; HEARTBEAT="$OUT/telemetry/agent_heartbeat.log"
rm -f /tmp/agent_tool_calls.log /tmp/agent_heartbeat.log
ln -s "$TELEMETRY" /tmp/agent_tool_calls.log; ln -s "$HEARTBEAT" /tmp/agent_heartbeat.log
export PRISM_TOOL_LOG_PATH="$TELEMETRY" PRISM_HEARTBEAT_PATH="$HEARTBEAT"
AGENT_PID=""
cleanup() { if [ -n "$AGENT_PID" ] && kill -0 "$AGENT_PID" 2>/dev/null; then kill -INT "$AGENT_PID" 2>/dev/null || true; wait "$AGENT_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
if [ "$MODE" != "eval" ]; then
  python -m agent.run start >"$OUT/logs/agent.log" 2>&1 & AGENT_PID=$!
  for _ in $(seq 1 90); do
    grep -q "registered worker" "$OUT/logs/agent.log" 2>/dev/null && break
    kill -0 "$AGENT_PID" 2>/dev/null || { tail -30 "$OUT/logs/agent.log" >&2; die "agent exited during startup"; }
    sleep 1
  done
  grep -q "registered worker" "$OUT/logs/agent.log" || die "agent registration was not observed within 90 seconds"
  (cd "$WORK_V3" && python run_tool_benchmark_all_released.py --provider "$LK_PROVIDER" --root_dir fdb_v3_data_released --force) 2>&1 | tee "$OUT/logs/inference.log"
  cleanup; AGENT_PID=""
fi

log "5/7 Stock FDB-v3 scoring"
(cd "$WORK_V3" && python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released --provider "$LK_PROVIDER" --output "$OUT/results/tool_accuracy_report.json" $LLM_FLAG) 2>&1 | tee "$OUT/logs/tool-scoring.log"
(cd "$WORK_V3" && python evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released --provider "$LK_PROVIDER" --output "$OUT/results/pass_rate_report.json" $LLM_FLAG) 2>&1 | tee "$OUT/logs/pass-scoring.log"
(cd "$WORK_V3" && python analyze_tool_latency.py --results-dir fdb_v3_data_released --provider "$LK_PROVIDER") 2>&1 | tee "$OUT/logs/latency.log" || die "stock latency analysis failed"
find "$RUN_DATA" -name "result_${LK_PROVIDER}.json" -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/results"
[ "$MODE" = "eval" ] || [ -s "$TELEMETRY" ] || die "no tool telemetry was captured"
[ "$MODE" = "eval" ] || [ -s "$HEARTBEAT" ] || die "no structured latency records were captured"

log "6/7 Leakage audit"
python "$ROOT/scripts/audit_benchmark_leakage.py" --benchmark "$WORK_V3/benchmark_data_v2.json" --reference "$WORK_V3/lk_agent_tool.py" > "$OUT/logs/leakage-audit.txt"
if grep -E '^(LEAK|SPEECH)' "$OUT/logs/leakage-audit.txt"; then die "leakage audit failed"; fi

log "7/7 Capture reproducibility artefacts"
python -m pip freeze > "$OUT/config/pip-freeze.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/config/submission-commit.txt"
git -C "$FDB_ROOT" rev-parse HEAD > "$OUT/config/fdb-commit.txt"
env | grep -E '^(LK_PROVIDER|GOOGLE_MODEL|OPENAI_MODEL)=' > "$OUT/config/model.env" || true
printf '%s\n' "$JUDGE" > "$OUT/config/judge.txt"; printf '%s\n' "$count" > "$OUT/config/recording-count.txt"
RUN_ID="$RUN_ID" MODE="$MODE" JUDGE="$JUDGE" MODEL="$MODEL" FDB_COMMIT="$FDB_COMMIT" python - "$OUT/config" <<'PY'
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
echo "Completed. Artefacts: $OUT"
