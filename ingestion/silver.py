import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd


BRONZE_DIR = Path(os.getenv('FINSENTINEL_BRONZE', 'data/bronze'))
DEDUP_DIR = Path(os.getenv('FINSENTINEL_BRONZE_DEDUP', 'data/bronze_dedup'))
SILVER_DIR = Path(os.getenv('FINSENTINEL_SILVER', 'data/silver'))
REPORT_DIR = Path(os.getenv('FINSENTINEL_VALIDATION', 'data/validation'))

SILVER_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    recs = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # skip malformed
                continue
    return recs


def _normalize_records(recs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in recs:
        row = {
            'id': r.get('id'),
            'title': r.get('title') or '',
            'summary': r.get('summary') or None,
            'content': r.get('content') or r.get('description') or None,
            'author': r.get('author') or None,
            'source': (r.get('source') or {}).get('name') if isinstance(r.get('source'), dict) else r.get('source'),
            'url': r.get('url') or None,
            'published': r.get('published') or None,
            'retrieved_at': r.get('retrieved_at') or None,
        }

        # fill missing summary using content or title
        if not row['summary']:
            if row['content']:
                row['summary'] = (row['content'][:1000]).strip()
            else:
                row['summary'] = (row['title'][:500]).strip()

        rows.append(row)

    df = pd.DataFrame(rows)

    # normalize timestamps to ISO UTC where possible
    for col in ['published', 'retrieved_at']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    # ensure id is string
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)

    return df


def build_silver(date_str: str = None) -> Dict[str, Any]:
    date_str = date_str or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    dedup_file = DEDUP_DIR / f"{date_str}.ndjson"
    bronze_file = BRONZE_DIR / f"{date_str}.ndjson"
    out_file = SILVER_DIR / f"{date_str}.parquet"
    report_file = REPORT_DIR / f"{date_str}.silver.json"

    report = {
        'date': date_str,
        'source_used': None,
        'records_in': 0,
        'records_out': 0,
        'output_file': str(out_file),
    }

    src = dedup_file if dedup_file.exists() else bronze_file
    report['source_used'] = str(src)
    if not src.exists():
        return report

    recs = _read_ndjson(src)
    report['records_in'] = len(recs)

    df = _normalize_records(recs)
    report['records_out'] = len(df)

    # write parquet
    try:
        df.to_parquet(out_file, index=False)
    except Exception as e:
        # fallback to CSV if parquet engine not available
        csv_file = SILVER_DIR / f"{date_str}.csv"
        df.to_csv(csv_file, index=False)
        report['output_file'] = str(csv_file)
        report['parquet_error'] = str(e)
    else:
        report['output_file'] = str(out_file)

    with report_file.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--date', help='YYYY-MM-DD (UTC) to build silver for', default=None)
    args = p.parse_args()

    r = build_silver(args.date)
    print(json.dumps(r, indent=2, default=str))
