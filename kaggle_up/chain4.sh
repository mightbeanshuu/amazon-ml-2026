#!/bin/zsh
# round6 completes -> harvest its submission -> push v10-with-artifacts into the freed slot.
K=/Users/mac/amazon-ml-2026/kagglecli/bin/kaggle
log() { echo "$(date +%H:%M:%S) $*"; }
while true; do
  S=$($K kernels status anshuaman0904/round6-full 2>&1 | grep -o 'KernelWorkerStatus[.A-Za-z]*')
  log "round6: $S"
  echo "$S" | grep -qi complete && break
  echo "$S" | grep -qiE "error|cancel" && { log "ROUND6 FAILED"; break; }
  sleep 240
done
if echo "$S" | grep -qi complete; then
  rm -rf /Users/mac/amazon-ml-2026/runs/round6_out && mkdir -p /Users/mac/amazon-ml-2026/runs/round6_out
  $K kernels output anshuaman0904/round6-full -p /Users/mac/amazon-ml-2026/runs/round6_out 2>&1 | tail -2
  find /Users/mac/amazon-ml-2026/runs/round6_out -name "matching_results*" -o -name "*.tsv" | head -5
  log "ROUND6 HARVESTED"
fi
sleep 90
for i in 1 2 3 4 5; do
  R=$($K kernels push -p /Users/mac/amazon-ml-2026/kaggle_up/k-v6 2>&1 | tail -1)
  log "v10 push: $R"
  echo "$R" | grep -q "successfully" && break
  sleep 300
done
log "CHAIN4 DONE"
