#!/usr/bin/env python3
"""
NYSED MDB → CSV Extractor
Run:  python3 nysed_extract.py
"""

import subprocess, sys, os, time
from pathlib import Path

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
MDB_PATH   = 'path/to/your/NYSED_file.mdb'  # UPDATE THIS PATH before running
OUT_FOLDER = 'data/extracted'  # UPDATE THIS PATH before running
TABLES     = []   # empty = extract everything
# ─────────────────────────────────────────────────────────────────────────────

def list_tables(mdb):
    # Use -1 flag so each table is on its own line — avoids space-splitting bug
    r = subprocess.run(['mdb-tables', '-1', mdb], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR: {r.stderr}")
        print("Try:  brew install mdbtools")
        sys.exit(1)
    return [t.strip() for t in r.stdout.strip().split('\n') if t.strip()]

def export_table(mdb, table, out_dir):
    safe = table.replace(' ', '_').replace('/', '_').replace('\\', '_')
    out  = Path(out_dir) / f"{safe}.csv"
    r    = subprocess.run(['mdb-export', mdb, table], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    out.write_text(r.stdout, encoding='utf-8')
    n_rows = r.stdout.count('\n') - 1
    kb     = out.stat().st_size / 1024
    return out, n_rows, kb

def main():
    print(f"\nFile : {MDB_PATH}")
    if not os.path.exists(MDB_PATH):
        print("ERROR: File not found. Check MDB_PATH at the top of this script.")
        sys.exit(1)

    size_mb = os.path.getsize(MDB_PATH) / 1_000_000
    print(f"Size : {size_mb:.0f} MB")

    print("\nReading table list...", end=' ', flush=True)
    all_tables = list_tables(MDB_PATH)
    print(f"{len(all_tables)} tables found\n")
    for i, t in enumerate(all_tables, 1):
        print(f"  {i:3}. {t}")

    to_do   = TABLES if TABLES else all_tables
    out_dir = Path(OUT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {len(to_do)} table(s) → {out_dir}\n")

    ok, fail = [], []
    for i, table in enumerate(to_do, 1):
        print(f"  [{i}/{len(to_do)}] {table}...", end=' ', flush=True)
        t0     = time.time()
        result = export_table(MDB_PATH, table, out_dir)
        if result:
            path, n_rows, kb = result
            print(f"{n_rows:,} rows  |  {kb:,.0f} KB  |  {time.time()-t0:.1f}s")
            ok.append((table, path, n_rows, kb))
        else:
            print("FAILED")
            fail.append(table)

    print(f"\n{'─'*60}")
    print(f"Done: {len(ok)} extracted, {len(fail)} failed")
    print(f"\nCSV files saved to:  {out_dir}")
    for table, path, n_rows, kb in ok:
        print(f"  {path.name:<60} {n_rows:>8,} rows")
    if fail:
        print(f"\nFailed: {fail}")
    print(f"\nUpload 'Annual_Regents_Exams.csv' back into Claude.")

if __name__ == '__main__':
    main()
