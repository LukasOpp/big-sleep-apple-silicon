#!/usr/bin/env bash
# Unattended Big Sleep batch on a RunPod GPU pod.
#
# Installs everything, generates COUNT images, uploads them to your cloud folder
# every few minutes, sends phone notifications, then TERMINATES the pod, so the
# billing stops without you having to remember.
#
# Pod start command (RunPod PyTorch template):
#   bash -c "curl -fsSL https://raw.githubusercontent.com/LukasOpp/big-sleep-apple-silicon/main/runpod/run.sh -o /workspace/run.sh && bash /workspace/run.sh"
# Give the pod a volume disk (/workspace) of ~20 GB so a restarted pod resumes instead of starting over.
#
# Or from the pod's web terminal (survives closing the tab; runs unattended):
#   export RCLONE_CONFIG_DEST_TYPE=drive
#   export RCLONE_CONFIG_DEST_TOKEN='{"access_token":...}'     <- SINGLE quotes, or the JSON breaks
#   nohup bash /workspace/run.sh > /dev/null 2>&1 &
#   tail -f /workspace/run_*.log
# Running it in the foreground of a terminal is "interactive": it never deletes the pod
# because of a setup error, only after a finished run.
#
# Tip: store the token as a RunPod Secret and set the env var to {{ RUNPOD_SECRET_<name> }}.
#
# Pod environment variables:
#   RCLONE_CONFIG_DEST_TYPE   e.g. dropbox or drive          (required)
#   RCLONE_CONFIG_DEST_TOKEN  the token JSON from `rclone authorize dropbox` on your Mac  (required)
#   NTFY_TOPIC     ntfy.sh topic to get push notifications on your phone (optional, recommended)
#   COUNT          images to make                    (default 1000)
#   MANIFEST       one image per <img> tag: a path in the cloud remote (e.g. big-sleep/manifest.json,
#                  made by extract_img_manifest.py) or an https URL. Overrides COUNT and CANVAS.
#   SHARD          split one job over several pods: "1/3" on the first pod, "2/3" on the second, ...
#                  (use the same MANIFEST and DEST_DIR on all of them)
#   SCALE          manifest sizes x this (default 1; 2 = retina)
#   PROMPTS        garbage | alt | mix   (alt = use the tag's alt text when it has one; default garbage)
#   CANVAS         output size: 1920x1080 | a comma list to pick from per image | random   (default 512x512)
#   DETAIL         128px cells on the short side before resizing to CANVAS (default 4 = 512px;
#                  time and VRAM grow with the cell count; use WORKERS=1 for big grids)
#   WORKERS        parallel jobs on the GPU          (default 2; ~8-10 GB VRAM each)
#   MAX_HOURS      hard stop no matter what          (default 30)
#   RUN_SEED       same seed = same prompts/settings (default: random)
#   DEST_DIR       folder inside the remote          (default big-sleep/<date>)
#   KEEP_POD=1     don't terminate at the end (for debugging)
set -uo pipefail

COUNT="${COUNT:-1000}"
WORKERS="${WORKERS:-2}"
MAX_HOURS="${MAX_HOURS:-30}"
POD="${RUNPOD_POD_ID:-unknown}"
INTERACTIVE=0; [ -t 0 ] && INTERACTIVE=1
mkdir -p /workspace
# Persist run id + seed so a restarted container resumes the same run.
[ -f /workspace/.run_id ] || { date -u +%Y-%m-%d_%H%M > /workspace/.run_id; echo "${RUN_SEED:-$RANDOM$RANDOM}" > /workspace/.run_seed; }
RUN_ID="$(cat /workspace/.run_id)"
RUN_SEED="$(cat /workspace/.run_seed)"
DEST="dest:${DEST_DIR:-big-sleep/$RUN_ID}"
RAW_BASE="${RAW_BASE:-https://raw.githubusercontent.com/LukasOpp/big-sleep-apple-silicon/main/runpod}"
OUT="/workspace/out/$RUN_ID"
LOG="/workspace/run_$RUN_ID.log"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

notify() {
  echo "[notify] $1"
  [ -n "${NTFY_TOPIC:-}" ] && curl -fsS -m 20 -H "Title: big sleep" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
}

upload() {
  rclone copy "$OUT" "$DEST" --exclude "*.tmp.png" --transfers 8 --retries 5 --low-level-retries 10 -q
}

terminate_pod() {
  if [ "${KEEP_POD:-0}" = "1" ]; then notify "KEEP_POD=1, leaving pod running. Remember to kill it!"; return; fi
  local how="$1"  # remove | stop
  sleep 5
  runpodctl "$how" pod "$POD" && return
  # Fallback: GraphQL API with the pod's own key
  local m; [ "$how" = remove ] && m="podTerminate(input:{podId:\\\"$POD\\\"})" || m="podStop(input:{podId:\\\"$POD\\\"}){id}"
  curl -fsS -m 30 -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { $m }\"}" "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY:-}" && return
  notify "⚠️ Could not $how pod $POD automatically. Kill it in the RunPod console NOW."
  sleep infinity   # don't exit: an exited container may get restarted and re-run
}

FINISHED=0
GENERATING=0
finish() {
  [ "$FINISHED" = 1 ] && return; FINISHED=1
  if [ "$GENERATING" = 0 ]; then
    # Failed during setup: nothing was made, so there's nothing to upload or protect.
    if [ "$INTERACTIVE" = 1 ]; then
      echo "Setup failed (see above). Pod left running; fix it and re-run."
    else
      notify "Setup failed on pod $POD (see log). Terminating so it doesn't bill idle."
      terminate_pod remove
    fi
    return
  fi
  kill "${UPLOADER_PID:-}" "${PROGRESS_PID:-}" 2>/dev/null
  local made; made=$(ls "$OUT"/*.png 2>/dev/null | grep -vc '\.tmp\.png$')
  cp "$LOG" "$OUT/run.log" 2>/dev/null
  local ok=0
  for attempt in 1 2 3; do upload && ok=1 && break; sleep 30; done
  local remote; remote=$(rclone lsf "$DEST" --include "*.png" 2>/dev/null | wc -l)
  touch /workspace/.done
  if [ "$made" = 0 ]; then
    notify "Nothing was generated (see log above). Terminating pod."
    terminate_pod remove
  elif [ "$ok" = 1 ] && [ "$remote" -ge "$made" ]; then
    notify "✅ Done: $made images uploaded to ${DEST#dest:}. Terminating pod."
    terminate_pod remove
  else
    # Don't destroy the only copy: stop the GPU (billing drops to disk only), keep /workspace.
    notify "⚠️ Upload incomplete ($remote/$made on remote). Stopping pod but keeping its volume. Images in $OUT."
    terminate_pod stop
  fi
}
trap finish EXIT INT TERM

if [ -f /workspace/.done ]; then
  FINISHED=1
  if [ "$INTERACTIVE" = 1 ]; then
    echo "A run already finished on this volume. To start a new one: rm /workspace/.done /workspace/.run_id /workspace/.run_seed"
    exit 0
  fi
  notify "Pod $POD restarted after the run already finished. Terminating."; terminate_pod remove; exit 0
fi

[ -n "${RCLONE_CONFIG_DEST_TYPE:-}" ] && [ -n "${RCLONE_CONFIG_DEST_TOKEN:-}" ] || {
  notify "RCLONE_CONFIG_DEST_TYPE / RCLONE_CONFIG_DEST_TOKEN not set (did you 'export' them?). Aborting."; exit 1; }
python -c 'import json,os; json.loads(os.environ["RCLONE_CONFIG_DEST_TOKEN"])' 2>/dev/null || {
  notify "RCLONE_CONFIG_DEST_TOKEN isn't valid JSON. In a shell, wrap it in single quotes: export RCLONE_CONFIG_DEST_TOKEN='{...}'"; exit 1; }
notify "Pod $POD starting: $COUNT images, $WORKERS workers, run seed $RUN_SEED."

# ---- setup -------------------------------------------------------------------
command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | bash >/dev/null
# Newer GPUs (RTX 5090 / PRO 4000 / PRO 6000 / B200 = Blackwell) need torch built for CUDA 12.8+;
# older templates (e.g. PyTorch 2.4) fail with "no kernel image". Upgrade only if needed.
if ! python -c "import torch; (torch.ones(2, device='cuda') @ torch.ones(2, device='cuda')).item()" 2>/dev/null; then
  echo "torch can't run on this GPU, upgrading to a CUDA 12.8 build"
  pip install -q -U torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    || { notify "torch upgrade for this GPU failed"; exit 1; }
fi
pip install -q big-sleep==0.9.1 || { notify "pip install failed"; exit 1; }
curl -fsSL "$RAW_BASE/generate.py" -o /workspace/generate.py || { notify "could not fetch generate.py"; exit 1; }
MANIFEST_ARGS=()
if [ -n "${MANIFEST:-}" ]; then
  case "$MANIFEST" in
    http*) curl -fsSL "$MANIFEST" -o /workspace/manifest.json ;;
    *)     rclone copyto "dest:$MANIFEST" /workspace/manifest.json ;;
  esac || { notify "could not fetch manifest $MANIFEST"; exit 1; }
  COUNT=$(python -c 'import json; print(len(json.load(open("/workspace/manifest.json"))))') \
    || { notify "manifest is not valid JSON"; exit 1; }
  MANIFEST_ARGS=(--manifest /workspace/manifest.json --scale "${SCALE:-1}")
  notify "manifest: $COUNT images"
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # old checkpoints vs torch>=2.6 defaults
rclone mkdir "$DEST" && echo "run seed $RUN_SEED, count $COUNT, canvas ${CANVAS:-512x512}, detail ${DETAIL:-4}" | rclone rcat "$DEST/_started.txt" \
  || { notify "Can't write to cloud folder (check rclone token). Aborting."; exit 1; }

# ---- generate ----------------------------------------------------------------
GENERATING=1
( while sleep 180; do upload; done ) & UPLOADER_PID=$!
( while sleep 7200; do
    n=$(ls "$OUT"/*.png 2>/dev/null | grep -vc '\.tmp\.png$'); notify "progress: $n/$COUNT"
  done ) & PROGRESS_PID=$!

SHARD="${SHARD:-1/1}"; SHARD_K="${SHARD%/*}"; SHARD_N="${SHARD#*/}"
pids=()
for ((w = 0; w < WORKERS; w++)); do
  timeout "${MAX_HOURS}h" python -u /workspace/generate.py --count "$COUNT" --out "$OUT" \
    --worker "$(( (SHARD_K - 1) * WORKERS + w ))" --workers "$(( SHARD_N * WORKERS ))" --run-seed "$RUN_SEED" \
    --canvas "${CANVAS:-512x512}" --detail "${DETAIL:-4}" \
    --prompt-source "${PROMPTS:-garbage}" "${MANIFEST_ARGS[@]}" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
kill "$PROGRESS_PID" 2>/dev/null
[ "$fail" = 1 ] && notify "A worker exited with an error or hit MAX_HOURS (see run.log in the folder)."
exit 0   # trap -> final upload -> terminate
