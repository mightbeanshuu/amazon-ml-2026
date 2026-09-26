"""Build <team>_submission.zip from an archived submission. Fails if it is over 50 MB or the matching file differs.
Usage: python tools/package.py <submissions/NN_...> <team_name> <path to filled Documentation_template.md>"""
import hashlib, json, os, sys, zipfile
sub, team, doc = sys.argv[1], sys.argv[2], sys.argv[3]
code = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
meta = json.load(open(os.path.join(sub, "meta.json")))
out = os.path.join(os.path.dirname(sub), f"{team}_submission.zip")
skip_dirs = {".venv", "__pycache__", ".git", ".impeccable"}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        z.write(os.path.join(sub, "output", f), f"output/{f}")
    for dp, dns, fns in os.walk(code):
        dns[:] = [d for d in dns if d not in skip_dirs]
        for fn in fns:
            if fn.endswith((".pyc", ".DS_Store")):
                continue
            full = os.path.join(dp, fn)
            z.write(full, os.path.join("code", "business_entity_resolution", os.path.relpath(full, code)))
    z.write(doc, "Documentation_template.md")
size = os.path.getsize(out) / 1e6
with zipfile.ZipFile(out) as z:
    md5 = hashlib.md5(z.read("output/matching_results.tsv")).hexdigest()
assert md5 == meta["matching_md5"], "matching_results.tsv differs from the archived leaderboard upload"
assert size <= 50, f"zip is {size:.1f} MB > 50 MB"
print(out, f"{size:.2f} MB", "md5 ok")
