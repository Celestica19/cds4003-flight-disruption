"""Audit and clean the entire original CSV using bounded-size chunks."""
import argparse
from collections import Counter
import json
import shutil
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, config, save_json, sha256_file
from src.schema import COLUMNS
from src.cleaning import DuplicateIndex, clean_chunk, load_airport_map, row_hashes, split_labels

def prepare(raw_dir, output_dir, cfg, fixture_mode=False):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if fixture_mode and output_dir.resolve() == ROOT.resolve():
        raise ValueError("Test fixtures must not write into the real project outputs")
    processed = output_dir / "data/processed"
    reports = output_dir / "reports"
    sample_dir = output_dir / "data/sample"
    for p in [processed, reports, sample_dir]:
        p.mkdir(parents=True, exist_ok=True)
    input_file = raw_dir / "flights.csv"
    header = pd.read_csv(input_file, nrows=0).columns.tolist()
    if not set(COLUMNS).issubset(header):
        raise ValueError(f"Expected source schema; missing: {set(COLUMNS) - set(header)}")
    if len(header) < cfg["minimum_raw_columns"]:
        raise ValueError("Source does not meet the column requirement")
    # Only replace files owned by this stage, never the original data.
    for name in ["train.csv.gz", "validation.csv.gz", "test.csv.gz", "embargo.csv.gz", "quarantine.csv.gz", "duplicates.sqlite"]:
        (processed / name).unlink(missing_ok=True)
    index = DuplicateIndex(processed / "duplicates.sqlite")
    mapping_file = raw_dir / "airport_id_to_iata.csv"
    mapping = load_airport_map(mapping_file)
    counts = Counter()
    post_missing = Counter()
    split_counts = Counter()
    raw_sample = None
    written = set()
    start = time.monotonic()
    try:
        for batch, raw in enumerate(pd.read_csv(input_file, dtype=str, keep_default_na=False, chunksize=cfg["chunk_size"]), 1):
            counts["raw_rows"] += len(raw)
            digests = row_hashes(raw)
            keys = raw[["YEAR", "MONTH", "DAY", "AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE"]]
            keys = [json.dumps(list(row), separators=(",", ":")) for row in keys.itertuples(index=False, name=None)]
            keep = index.keep(digests, keys)
            counts["exact_duplicates_removed"] += int((~keep).sum())
            # Raw missingness includes every input row, including duplicates.
            all_clean, all_rejected, raw_counts = clean_chunk(raw, cfg["expected_year"], mapping)
            counts.update({k: v for k, v in raw_counts.items() if k.startswith("raw_missing__")})
            raw["record_id"] = digests
            unique = raw.loc[keep].copy()
            raw_sample = unique if raw_sample is None else pd.concat([raw_sample, unique], ignore_index=True)
            raw_sample = raw_sample.sort_values("record_id").head(1000)
            if keep.all():
                clean, rejected, changes = all_clean, all_rejected, raw_counts
            else:
                clean, rejected, changes = clean_chunk(unique, cfg["expected_year"], mapping)
            counts.update({k: v for k, v in changes.items() if not k.startswith("raw_missing__")})
            clean["record_id"] = unique.loc[clean.index, "record_id"]
            rejected["record_id"] = unique.loc[rejected.index, "record_id"]
            counts["eligible_rows"] += len(clean)
            counts["quarantined_rows"] += len(rejected)
            post_missing.update(clean[COLUMNS].isna().sum().to_dict())
            labels = split_labels(clean.flight_date, cfg)
            for split in ["train", "validation", "test", "embargo"]:
                piece = clean.loc[labels.eq(split)]
                split_counts[split] += len(piece)
                if len(piece):
                    path = processed / f"{split}.csv.gz"
                    piece.to_csv(path, mode="at", header=split not in written, index=False, compression={"method": "gzip", "compresslevel": 1})
                    written.add(split)
            if len(rejected):
                rejected.to_csv(processed / "quarantine.csv.gz", mode="at", header="quarantine" not in written, index=False, compression={"method": "gzip", "compresslevel": 1})
                written.add("quarantine")
            print(f"Batch {batch}: {counts['raw_rows']:,} raw; {counts['eligible_rows']:,} eligible", flush=True)
        counts["business_keys_with_distinct_records_retained_for_review"] = index.conflicting_business_keys()
    finally:
        index.close()
    assert counts["raw_rows"] == counts["exact_duplicates_removed"] + counts["quarantined_rows"] + counts["eligible_rows"]
    assert sum(split_counts.values()) == counts["eligible_rows"]
    for name in ["train", "validation", "test"]:
        if not split_counts[name]:
            raise ValueError(f"Empty {name} partition; check source date coverage")
    quality = pd.DataFrame([{
        "column": c, "raw_rows": counts["raw_rows"], "raw_missing": counts[f"raw_missing__{c}"],
        "raw_missing_pct": 100 * counts[f"raw_missing__{c}"] / counts["raw_rows"],
        "eligible_rows": counts["eligible_rows"], "clean_missing": post_missing[c],
        "clean_missing_pct": 100 * post_missing[c] / counts["eligible_rows"]
    } for c in COLUMNS])
    quality.to_csv(reports / "missingness_before_after.csv", index=False)
    raw_sample[COLUMNS].to_csv(sample_dir / "flights_sample.csv", index=False)
    sufficient = counts["raw_rows"] > cfg["minimum_raw_rows_exclusive"] and counts["eligible_rows"] > cfg["minimum_raw_rows_exclusive"]
    audit = {
        "status": "TEST_FIXTURE_ONLY" if fixture_mode else ("REAL_DATA_PREPARED" if sufficient else "REQUIREMENT_FAILED"),
        "raw_columns": len(header), "counts": dict(counts), "split_counts": dict(split_counts),
        "elapsed_seconds": round(time.monotonic() - start, 2), "source_sha256": sha256_file(input_file),
        "airport_crosswalk_sha256": sha256_file(mapping_file) if mapping_file.exists() else None,
        "configuration": cfg, "row_reconciliation_passed": True,
        "notes": [
            "Raw missingness uses all raw rows; clean missingness uses unique eligible rows. Denominators differ.",
            "Structural missing outcomes are preserved; unknown labels are quarantined, never imputed.",
            "Predictor imputation and encoding are fitted inside each model training fold.",
            "Unresolved BTS numeric airport IDs are retained in a separate namespace, not guessed.",
            "Exact duplicate rows are removed; non-identical business-key repeats are flagged, not automatically deleted.",
            "Two calendar days at each development/holdout boundary are embargoed, not discarded from the dataset."
        ]
    }
    save_json(audit, reports / "quality_audit.json")
    if not sufficient and not fixture_mode:
        raise ValueError("Dataset must have more than 300,000 raw AND eligible rows. See quality_audit.json.")
    lookups = {"airlines": {}, "airports": {}}
    for filename, kind, name_column in [("airlines.csv", "airlines", "AIRLINE"), ("airports.csv", "airports", "AIRPORT")]:
        lookup_path = raw_dir / filename
        if lookup_path.exists():
            table = pd.read_csv(lookup_path, dtype=str)
            if {"IATA_CODE", name_column}.issubset(table.columns):
                lookups[kind] = dict(zip(table.IATA_CODE, table[name_column].fillna("")))
    save_json(lookups, output_dir / "models/lookups.json")
    manifest_path = reports / "download_manifest.json"
    if manifest_path.exists() and not fixture_mode:
        manifest = json.loads(manifest_path.read_text())
        manifest["row_count_status"] = "Verified by full-file preparation"
        manifest["verified_rows"] = counts["raw_rows"]
        manifest["verified_columns"] = len(header)
        save_json(manifest, manifest_path)
    return audit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    audit = prepare(ROOT / "data/raw", ROOT, config(args.config))
    print(json.dumps({"status": audit["status"], "split_counts": audit["split_counts"]}, indent=2))

if __name__ == "__main__":
    main()
