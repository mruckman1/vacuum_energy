#!/bin/bash
# Control the Layer-5 S8 job pools (papers/interacting-vacua-first-numbers).
#
#   scripts/s8_pool.sh status    what is running, what is done, what a hard stop would cost
#   scripts/s8_pool.sh pause     suspend everything in place (instant, loses nothing)
#   scripts/s8_pool.sh resume    continue a paused pool, or relaunch it after a reboot
#   scripts/s8_pool.sh drain     let running jobs finish and persist, start no new ones
#   scripts/s8_pool.sh stop      kill everything (survives a reboot; loses in-flight jobs)
#
# The pools are idempotent at job granularity: every finished job is one file under
# data/jobs/s8_<tag>.json and is never recomputed, so `resume` always picks up where it
# left off.  There is no checkpointing *inside* a job, so `stop` discards the elapsed
# time of whatever is running; `pause` and `drain` do not.
set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
NB="$REPO/papers/interacting-vacua-first-numbers/notebook_qei_exact.py"
LOGS="$HOME/s8_logs"                      # outside /tmp: survives a reboot
PARENT_PAT="notebook_qei_exact.py --diagnostics --only s8"
WORKER_PAT="notebook_qei_exact.py --job"

parents() { pgrep -f "$PARENT_PAT" 2>/dev/null; }
workers() { pgrep -f "$WORKER_PAT" 2>/dev/null; }
n_of()    { local c; c=$(echo "${1:-}" | grep -c . ); echo "$c"; }

progress() {
  "$PY" - "$REPO" <<'PY'
import glob, json, os, sys
d = os.path.join(sys.argv[1], "papers/interacting-vacua-first-numbers/data")
res = {}
for f in glob.glob(d + "/jobs/s8_*.json"):
    if f.endswith(".spec.json"):
        continue
    try:
        r = json.load(open(f))["row"]
    except Exception:
        continue
    if "time_s" in r:
        res[r.get("tag")] = r["time_s"]
try:
    jobs = json.load(open(d + "/s8_plan.json"))["jobs"]
except Exception:
    print("  (no plan file; %d results on disk)" % len(res)); raise SystemExit
rat = sorted(res[j["tag"]] / j["est_wall_s"] for j in jobs
             if j["tag"] in res and j.get("est_wall_s"))
cal = rat[len(rat) // 2] if rat else 1.0
rem = [j for j in jobs if j["tag"] not in res]
h = sum(j.get("est_wall_s", 0) for j in rem) * cal / 3600
print("  jobs      %d of %d done" % (len(res), len(jobs)))
print("  remaining %d jobs, %.0f h of job wall (calibrated x%.2f on measured runs)"
      % (len(rem), h, cal))
big = sorted(rem, key=lambda j: -j.get("est_wall_s", 0))[:3]
if big and big[0].get("est_wall_s", 0) * cal / 3600 > 10:
    print("  of which  %.0f h is in the %d longest: %s"
          % (sum(j["est_wall_s"] for j in big) * cal / 3600, len(big),
             ", ".join(j["tag"] for j in big)))
PY
}

# Consumed CPU time, not elapsed time: a job suspended by `pause` keeps ageing on the
# wall clock but does no work, so etime would overstate what a hard stop discards.
# Jobs run on 2 BLAS threads, so CPU seconds are divided by 2 to read as work done.
at_risk() {
  local ps_out
  ps_out=$(for p in $(workers); do ps -o time= -p "$p" 2>/dev/null; done)
  echo "$ps_out" | tr -d ' ' | awk -F'[-:]' '
    NF { n = NF; s = 0
         if (n >= 1) s += $n          # seconds, may carry a fraction
         if (n >= 2) s += $(n-1) * 60
         if (n >= 3) s += $(n-2) * 3600
         if (n >= 4) s += $(n-3) * 86400
         t += s }
    END { printf "%.1f", t / 2 / 3600 }'
}

launch() {
  mkdir -p "$LOGS"
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
  cd "$REPO" || exit 1
  # S8 TRIAGE 2026-09-12: part "plane" is RETIRED (weight 0.5 + exact SVD; see the
  # MANIFEST "S8 triage" section) and replaced by "plane2" (sketched SVD, weight an
  # anchored axis) plus "wt" (the exact-ED weight ladder every plane2 column rests on).
  nohup "$PY" "$NB" --diagnostics --only s8 \
        --s8-parts lam01,tau0,plane2,wt,mpo_anchor,keff --workers 5 >> "$LOGS/pool_a.log" 2>&1 &
  echo "  pool A (plane2, weight ladder, anchors, tau0) started, pid $!"
  nohup "$PY" "$NB" --diagnostics --only s8 --s8-parts ed --workers 1 >> "$LOGS/pool_b.log" 2>&1 &
  echo "  pool B (exact diagonalization) started, pid $!"
  echo "  logs: $LOGS/pool_a.log, $LOGS/pool_b.log"
}

case "${1:-status}" in

  status)
    np=$(n_of "$(parents)"); nw=$(n_of "$(workers)")
    state=$(for p in $(parents); do ps -o state= -p "$p"; done | tr -d ' ' | sort -u | tr '\n' ' ')
    echo "S8 job pools"
    if [ "$np" -eq 0 ]; then
      echo "  pools     not running (resume relaunches them)"
    elif echo "$state" | grep -q T; then
      echo "  pools     PAUSED ($np parents, $nw workers suspended, holding memory)"
    else
      echo "  pools     running ($np parents, $nw workers)"
    fi
    progress
    if [ "$nw" -gt 0 ]; then
      echo "  in flight $nw jobs, $(at_risk) h of compute a hard stop would discard:"
      for p in $(workers); do
        ps -o etime= -p "$p" | tr -d ' ' | tr '\n' ' '
        ps -o command= -p "$p" | sed 's/.*jobs\///; s/\.spec\.json.*//'
      done | sed 's/^/    /'
    fi
    ;;

  pause)
    if [ "$(n_of "$(parents)")" -eq 0 ]; then echo "pools are not running"; exit 0; fi
    pkill -STOP -f "$WORKER_PAT"
    pkill -STOP -f "$PARENT_PAT"
    echo "paused: $(n_of "$(workers)") workers suspended in place, nothing lost."
    echo "ten cores are free now. Memory stays held (macOS will swap it out under pressure)."
    echo "a reboot while paused loses the in-flight jobs only; finished jobs are safe."
    echo "resume with: scripts/s8_pool.sh resume"
    ;;

  resume)
    if [ "$(n_of "$(parents)")" -gt 0 ]; then
      pkill -CONT -f "$WORKER_PAT"
      pkill -CONT -f "$PARENT_PAT"
      echo "resumed in place: $(n_of "$(workers)") workers continue from where they stopped."
    else
      echo "no pools found (rebooted, or stopped); relaunching, finished jobs are skipped:"
      launch
    fi
    progress
    ;;

  drain)
    if [ "$(n_of "$(parents)")" -eq 0 ]; then echo "pools are not running"; exit 0; fi
    pkill -STOP -f "$PARENT_PAT"
    echo "draining: no new job will start; the $(n_of "$(workers)") running jobs finish and are saved."
    echo "watch with: pgrep -f '$WORKER_PAT'   (empty means quiesced, then stop is free)"
    ;;

  stop)
    r=$(at_risk)
    pkill -f "$PARENT_PAT"; pkill -CONT -f "$WORKER_PAT" 2>/dev/null; pkill -f "$WORKER_PAT"
    echo "stopped. $r h of in-flight compute discarded; every finished job is kept."
    echo "restart with: scripts/s8_pool.sh resume"
    ;;

  *) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
