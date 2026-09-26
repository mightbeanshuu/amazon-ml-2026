#!/bin/zsh
# Push stage1 (offline wheels) -> verify its OUTPUT really exists -> push round6.
K=/Users/mac/amazon-ml-2026/kagglecli/bin/kaggle
D=/Users/mac/amazon-ml-2026/kaggle_up
log() { echo "$(date +%H:%M:%S) $*"; }
for i in {1..20}; do $K datasets files anshuaman0904/pip-wheels 2>/dev/null | grep -q whl && break; log "wheels processing"; sleep 20; done
cd $D/k-stage1 && $K kernels push -p . 2>&1 | while read l; do log "stage1 push: $l"; done
sleep 60
while true; do
  S=$($K kernels status anshuaman0904/stage1-all 2>&1)
  log "stage1: $S"
  echo "$S" | grep -qi "complete" && break
  echo "$S" | grep -qiE '"error|failed|cancel' && { log "STAGE1 FAILED"; exit 1; }
  sleep 180
done
rm -rf /tmp/s1chk && mkdir -p /tmp/s1chk
$K kernels output anshuaman0904/stage1-all -p /tmp/s1chk --force 2>&1 | tail -2 | while read l; do log "out: $l"; done
if ls /tmp/s1chk | grep -q "edges_train_q1.parquet"; then
  log "stage1 OUTPUT VERIFIED"
else
  log "STAGE1 COMPLETE BUT NO OUTPUT — files: $(ls /tmp/s1chk | head -5 | tr '\n' ' ')"; exit 1
fi
cd $D/k-round6 && $K kernels push -p . 2>&1 | while read l; do log "round6 push: $l"; done
log "round6 QUEUED"
