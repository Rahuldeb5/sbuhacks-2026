#!/usr/bin/env bash
# Run a command and kill it (and its children) if its total RSS exceeds LIMIT_MB (default 6000).
# WSL has 15 GB and crashed twice when an experiment grew unbounded.  Usage: scripts/run_capped.sh cmd args...
LIMIT_MB=${LIMIT_MB:-6000}
"$@" &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= --ppid $pid -p $pid 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
  [ "${rss:-0}" -gt "$peak" ] && peak=$rss
  if [ "${rss:-0}" -gt "$LIMIT_MB" ]; then
    echo "run_capped: KILLED, RSS ${rss} MB > ${LIMIT_MB} MB" >&2
    pkill -9 -P $pid; kill -9 $pid; exit 137
  fi
  sleep 0.5
done
wait $pid; rc=$?
echo "run_capped: exit $rc, peak RSS ${peak} MB" >&2
exit $rc
