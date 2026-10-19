#!/usr/bin/env bash
# Operator controls for the loop.
#   harness/ctl.sh start | stop | pause | resume | status | tail | review-log N | kill | dashboard
set -uo pipefail
# shellcheck source=harness/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
cd "$ROOT" || exit 1

case "${1:-status}" in
  start)
    rm -f STOP PAUSE
    pm2 start harness/ecosystem.config.js --only perturb26-loop && pm2 save
    ;;
  stop) touch STOP; echo "STOP set: the loop exits after the current iteration (pm2 keeps it stopped)" ;;
  kill) touch STOP; pm2 stop perturb26-loop ;;
  pause) touch PAUSE; echo "PAUSE set: the loop idles before the next iteration" ;;
  resume) rm -f PAUSE; echo "resumed" ;;
  dashboard) pm2 start harness/ecosystem.config.js --only perturb26-dashboard && pm2 save ;;
  status)
    pm2 describe perturb26-loop 2>/dev/null | grep -E "status|restarts|uptime" || echo "perturb26-loop not under pm2"
    [[ -f STOP ]] && echo "STOP is set"; [[ -f PAUSE ]] && echo "PAUSE is set"
    echo "--- best"; cat status/best.json 2>/dev/null || echo "(none)"; echo
    echo "--- ledger (last 10)"; tail -n 10 status/results.tsv 2>/dev/null | column -t -s $'\t'
    echo "--- iterations (last 5)"; tail -n 5 status/iterations.tsv 2>/dev/null | column -t -s $'\t'
    echo "--- spend: $(awk -F'\t' 'NR>1 && $6 ~ /^[0-9.]+$/ {s+=$6} END {printf "%.2f", s+0}' status/iterations.tsv 2>/dev/null) USD"
    ;;
  tail) tail -n 50 -f logs/harness.log ;;
  review-log) "$PY" -c "
import json,sys
for l in open(sys.argv[1],encoding='utf-8',errors='replace'):
    try: e=json.loads(l)
    except Exception: continue
    if e.get('type')=='assistant':
        for c in e['message'].get('content',[]):
            if c.get('type')=='text': print(c['text'])
" "logs/review-$(printf '%04d' "${2:?iteration}").jsonl" ;;
  *) echo "usage: $0 start|stop|kill|pause|resume|status|tail|review-log N|dashboard"; exit 1 ;;
esac
