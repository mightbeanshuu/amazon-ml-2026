#!/bin/zsh
# Dataset ready -> push stage1-all -> completes -> push round6-full. Logs every step.
K=/Users/mac/amazon-ml-2026/kagglecli/bin/kaggle
D=/Users/mac/amazon-ml-2026/kaggle_up
log() { echo "$(date +%H:%M:%S) $*"; }
# 1. dataset ready?
for i in {1..60}; do
  $K datasets files anshuaman0904/amc-2026 > /tmp/dsf.txt 2>&1
  if grep -q "student_resource" /tmp/dsf.txt; then log "dataset READY"; break; fi
  log "dataset processing... ($(head -c 80 /tmp/dsf.txt | tr '\n' ' '))"; sleep 30
done
grep -q "student_resource" /tmp/dsf.txt || { log "DATASET NEVER READY"; exit 1; }
# 2. push stage1
cd $D/k-stage1 && $K kernels push -p . 2>&1 | while read l; do log "stage1 push: $l"; done
# 3. wait for stage1 to finish
while true; do
  S=$($K kernels status anshuaman0904/stage1-all 2>&1)
  log "stage1: $S"
  echo "$S" | grep -qi "complete" && break
  echo "$S" | grep -qiE "error|failed|cancel" && { log "STAGE1 FAILED"; exit 1; }
  sleep 120
done
# 4. push round6
cd $D/k-round6 && $K kernels push -p . 2>&1 | while read l; do log "round6 push: $l"; done
log "round6 QUEUED — poll: $K kernels status anshuaman0904/round6-full"
