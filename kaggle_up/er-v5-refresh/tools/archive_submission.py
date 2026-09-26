"""Archive a run's leaderboard file as a numbered submission, so the best one can be re-packaged byte-identically.
Usage: python tools/archive_submission.py <run_out_dir> "<note>" [public_lb_score]"""
import hashlib, json, os, shutil, subprocess, sys, time
run, note = sys.argv[1], sys.argv[2]
score = sys.argv[3] if len(sys.argv) > 3 else None
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "submissions"))
os.makedirs(root, exist_ok=True)
n = len([d for d in os.listdir(root) if d[:2].isdigit()]) + 1
dst = os.path.join(root, f"{n:02d}_{time.strftime('%m%d_%H%M')}")
os.makedirs(os.path.join(dst, "output"))
# only the leaderboard file: candidate_pairs.tsv is ~1 GB at full scale and is rebuilt from test_<c>.parquet
shutil.copy(os.path.join(run, "output", "matching_results.tsv"), os.path.join(dst, "output"))
for f in ("run_log.json", "run_log_train.json", "run_log_final.json", "models/meta.json"):
    if os.path.exists(os.path.join(run, f)):
        shutil.copy(os.path.join(run, f), dst)
here = os.path.dirname(os.path.abspath(__file__))
git = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
md5 = hashlib.md5(open(os.path.join(dst, "output", "matching_results.tsv"), "rb").read()).hexdigest()
meta = dict(n=n, note=note, public_lb=score, git=git, git_dirty=bool(dirty), matching_md5=md5, run=os.path.abspath(run))
json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
print(dst, meta)
