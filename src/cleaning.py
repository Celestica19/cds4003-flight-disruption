"""Deterministic cleaning; no fitted statistics and no target imputation."""
from collections import Counter
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import COLUMNS, TEXT_COLUMNS, NUMERIC_COLUMNS, CLOCK_COLUMNS, NONNEGATIVE_COLUMNS, POSITIVE_COLUMNS

MISSING_TOKENS = {"", "NA", "N/A", "NAN", "NULL", "NONE", "?"}

def row_hashes(frame):
    """SHA256 of all raw fields, encoded without delimiter ambiguity."""
    return [hashlib.sha256(json.dumps(list(row), ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
            for row in frame[COLUMNS].fillna("").astype(str).itertuples(index=False, name=None)]

class DuplicateIndex:
    """Disk-backed exact-row duplicate tracking across CSV chunk boundaries."""
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS records (digest TEXT PRIMARY KEY, business_key TEXT) WITHOUT ROWID")

    def keep(self, digests, business_keys):
        flags = []
        with self.connection:
            for digest, key in zip(digests, business_keys):
                cursor = self.connection.execute("INSERT OR IGNORE INTO records VALUES (?, ?)", (digest, key))
                flags.append(cursor.rowcount == 1)
        return np.array(flags, dtype=bool)

    def conflicting_business_keys(self):
        return int(self.connection.execute("SELECT COUNT(*) FROM (SELECT business_key FROM records GROUP BY business_key HAVING COUNT(*) > 1)").fetchone()[0])

    def close(self):
        self.connection.close()

def valid_hhmm(values):
    x = pd.to_numeric(values, errors="coerce")
    return x.notna() & ((x == 2400) | ((x >= 0) & (x <= 2359) & (x % 100 < 60) & (x % 1 == 0)))

def hhmm_minutes(values):
    x = pd.to_numeric(values, errors="coerce").where(valid_hhmm(values))
    return ((x // 100) * 60 + x % 100).where(x != 2400, 0)

def load_airport_map(path):
    if path is None or not Path(path).exists():
        return {}
    mapping = pd.read_csv(path, dtype=str)
    if not {"airport_id", "iata_code"}.issubset(mapping):
        raise ValueError("Airport mapping must contain airport_id and iata_code")
    mapping = mapping[["airport_id", "iata_code"]].drop_duplicates()
    if mapping.airport_id.duplicated().any():
        raise ValueError("Ambiguous airport IDs in mapping; verify a 2015-valid authoritative crosswalk")
    if not mapping.airport_id.str.fullmatch(r"\d{5}").all() or not mapping.iata_code.str.fullmatch(r"[A-Z]{3}").all():
        raise ValueError("Airport mapping contains malformed IDs or IATA codes")
    return dict(zip(mapping.airport_id, mapping.iata_code))

def clean_chunk(raw, expected_year=2015, airport_map=None):
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    d = raw[COLUMNS].copy()
    counts = Counter()
    for c in COLUMNS:
        values = d[c].astype("string")
        stripped = values.str.strip()
        counts[f"whitespace_normalized__{c}"] = int((values != stripped).fillna(False).sum())
        absent = stripped.str.upper().isin(MISSING_TOKENS) | stripped.isna()
        d[c] = stripped.mask(absent)
        counts[f"raw_missing__{c}"] = int(absent.sum())
    for c in TEXT_COLUMNS:
        values = d[c].str.upper()
        counts[f"case_normalized__{c}"] = int((d[c] != values).fillna(False).sum())
        d[c] = values
    for c in NUMERIC_COLUMNS:
        values = pd.to_numeric(d[c], errors="coerce").astype(float)
        invalid = (d[c].notna() & values.isna()) | np.isinf(values)
        counts[f"invalid_numeric__{c}"] = int(invalid.sum())
        d[c] = values.mask(invalid)
    for c in CLOCK_COLUMNS:
        invalid = d[c].notna() & ~valid_hhmm(d[c])
        counts[f"invalid_clock__{c}"] = int(invalid.sum())
        counts[f"midnight_2400__{c}"] = int(d[c].eq(2400).sum())
        d.loc[invalid, c] = np.nan
    for c in NONNEGATIVE_COLUMNS + POSITIVE_COLUMNS:
        invalid = d[c].notna() & (d[c].lt(0) if c in NONNEGATIVE_COLUMNS else d[c].le(0))
        counts[f"invalid_domain__{c}"] = int(invalid.sum())
        d.loc[invalid, c] = np.nan
    # Negative departure/arrival delays mean early operation and are valid.
    for c in ["ARRIVAL_DELAY", "DEPARTURE_DELAY"]:
        counts[f"negative_delay_retained__{c}"] = int(d[c].lt(0).sum())
        counts[f"extreme_delay_over_1440_retained__{c}"] = int(d[c].gt(1440).sum())
    for c in ["CANCELLED", "DIVERTED"]:
        invalid = d[c].notna() & ~d[c].isin([0, 1])
        counts[f"invalid_binary__{c}"] = int(invalid.sum())
        d.loc[invalid, c] = np.nan
    for c in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        numeric = d[c].str.fullmatch(r"\d{5}").fillna(False)
        mapped = d[c].map(airport_map or {})
        counts[f"numeric_airport_ids__{c}"] = int(numeric.sum())
        counts[f"numeric_airport_ids_mapped__{c}"] = int((numeric & mapped.notna()).sum())
        # Do not guess a three-letter code or discard the entire month.
        unresolved = numeric & mapped.isna()
        d.loc[numeric & mapped.notna(), c] = mapped[numeric & mapped.notna()]
        d.loc[unresolved, c] = "BTS_" + d.loc[unresolved, c]
        invalid = d[c].notna() & ~d[c].str.fullmatch(r"[A-Z]{3}|BTS_\d{5}").fillna(False)
        counts[f"invalid_airport_format__{c}"] = int(invalid.sum())
        counts[f"unresolved_airport_ids__{c}"] = int(unresolved.sum())
        d.loc[invalid, c] = pd.NA
    invalid_airline = d.AIRLINE.notna() & ~d.AIRLINE.str.fullmatch(r"[A-Z0-9]{2}").fillna(False)
    counts["invalid_airline_format"] = int(invalid_airline.sum())
    d.loc[invalid_airline, "AIRLINE"] = pd.NA
    invalid_reason = d.CANCELLATION_REASON.notna() & ~d.CANCELLATION_REASON.isin(list("ABCD"))
    counts["invalid_cancellation_reason"] = int(invalid_reason.sum())
    d.loc[invalid_reason, "CANCELLATION_REASON"] = pd.NA
    # A cancellation reason is inapplicable for a non-cancelled flight.
    inconsistent_reason = d.CANCELLED.eq(0) & d.CANCELLATION_REASON.notna()
    counts["reason_on_noncancelled_flight_cleared"] = int(inconsistent_reason.sum())
    d.loc[inconsistent_reason, "CANCELLATION_REASON"] = pd.NA
    date_parts = d[["YEAR", "MONTH", "DAY"]].rename(columns=str.lower)
    integer_date = (date_parts % 1).eq(0).all(axis=1)
    d["flight_date"] = pd.to_datetime(date_parts.where(integer_date), errors="coerce")
    expected_dow = d.flight_date.dt.dayofweek + 1
    mismatch = d.DAY_OF_WEEK.notna() & expected_dow.notna() & d.DAY_OF_WEEK.ne(expected_dow)
    counts["day_of_week_corrected"] = int(mismatch.sum())
    d["DAY_OF_WEEK"] = expected_dow
    known_flags = d.CANCELLED.notna() & d.DIVERTED.notna()
    inconsistent_flags = d.CANCELLED.eq(1) & d.DIVERTED.eq(1)
    event = d.CANCELLED.eq(1) | d.DIVERTED.eq(1)
    known_target = known_flags & ~inconsistent_flags & (event | d.ARRIVAL_DELAY.notna())
    d["disrupted"] = np.where(known_target, (event | d.ARRIVAL_DELAY.ge(15)).astype(int), np.nan)
    # Mutually exclusive quarantine reasons permit exact row reconciliation.
    reason = pd.Series("", index=d.index, dtype="string")
    reason.loc[d.flight_date.isna()] = "invalid_or_missing_date"
    reason.loc[reason.eq("") & d.flight_date.dt.year.ne(expected_year)] = "unexpected_year"
    reason.loc[reason.eq("") & inconsistent_flags] = "contradictory_outcome_flags"
    reason.loc[reason.eq("") & ~known_target] = "unknown_target"
    for name, count in reason[reason.ne("")].value_counts().items():
        counts[f"quarantine__{name}"] = int(count)
    rejected = d.loc[reason.ne("")].copy()
    rejected["quarantine_reason"] = reason[reason.ne("")]
    good = d.loc[reason.eq("")].copy()
    good["disrupted"] = good.disrupted.astype("int8")
    good["flight_date"] = good.flight_date.dt.strftime("%Y-%m-%d")
    return good, rejected, counts

def split_labels(dates, cfg):
    dates = pd.to_datetime(dates)
    val = pd.Timestamp(cfg["validation_start"])
    test = pd.Timestamp(cfg["test_start"])
    gap = pd.Timedelta(days=cfg["embargo_days"])
    labels = pd.Series("embargo", index=dates.index)
    labels.loc[dates < val - gap] = "train"
    labels.loc[(dates >= val) & (dates < test - gap)] = "validation"
    labels.loc[dates >= test] = "test"
    return labels
