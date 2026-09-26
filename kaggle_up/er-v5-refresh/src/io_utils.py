"""TSV input/output and a local copy of every submission rule in the problem statement."""
import csv
import os

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    """Read a TSV exactly: tab separator, no quote processing (names can contain quotes), empty stays ''."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    with open(path, encoding="utf-8") as fh:
        n_lines = sum(1 for line in fh if line.strip())
    if len(df) != n_lines - 1:
        raise ValueError(f"{path}: parsed {len(df)} rows but file has {n_lines - 1} data lines")
    return df


def load_split(data_dir, split):
    d = os.path.join(data_dir, split)
    s = [read_tsv(os.path.join(d, f"{split}_source{i}.tsv")) for i in (1, 2, 3)]
    gt = None
    gp = os.path.join(d, f"{split}_ground_truth.tsv")
    if os.path.exists(gp):
        gt = read_tsv(gp)
    return s, gt


def write_lists(path, key_col, val_col, s1_ids, mapping):
    """One row per S1 in the given order. The ID list is comma-joined with no quoting and no duplicates."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"{key_col}\t{val_col}\n")
        for e in s1_ids:
            ids = sorted(set(mapping.get(e, ())))
            fh.write(f"{e}\t{','.join(ids)}\n")


def validate(matching_path, candidate_path, test_dir):
    """Checks every rule stated in the problem statement. Returns a list of issues (empty = pass)."""
    issues = []
    s1 = read_tsv(os.path.join(test_dir, "test_source1.tsv"))["entity_id"]
    right = set(read_tsv(os.path.join(test_dir, "test_source2.tsv"))["entity_id"]) | \
        set(read_tsv(os.path.join(test_dir, "test_source3.tsv"))["entity_id"])
    parsed = {}
    for path, col in ((matching_path, "matched_entity_ids"), (candidate_path, "candidate_entity_ids")):
        df = read_tsv(path)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{path}: header {list(df.columns)} != ['source1_entity_id', '{col}']")
            continue
        if df.source1_entity_id.duplicated().any():
            issues.append(f"{path}: duplicate source1_entity_id rows")
        missing = set(s1) - set(df.source1_entity_id)
        extra = set(df.source1_entity_id) - set(s1)
        if missing:
            issues.append(f"{path}: {len(missing)} test S1 entities missing")
        if extra:
            issues.append(f"{path}: {len(extra)} unknown source1 ids")
        m = {}
        for e, v in zip(df.source1_entity_id, df[col]):
            ids = [t for t in v.split(",") if t] if v else []
            if len(ids) != len(set(ids)):
                issues.append(f"{path}: duplicate ids in list for {e}")
            bad = [t for t in ids if not (t.startswith("S2-") or t.startswith("S3-")) or t not in right]
            if bad:
                issues.append(f"{path}: {e} has invalid ids {bad[:3]}")
            if '"' in v or " " in v:
                issues.append(f"{path}: {e} list contains quotes/spaces")
            m[e] = set(ids)
        parsed[col] = m
    if len(parsed) == 2:
        not_sub = sum(1 for e, ids in parsed["matched_entity_ids"].items()
                      if not ids <= parsed["candidate_entity_ids"].get(e, set()))
        if not_sub:
            issues.append(f"WARNING: {not_sub} S1 rows have matches that are not in candidate_pairs")
    return issues
