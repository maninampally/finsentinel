import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd


BRONZE_DIR = Path(os.getenv('FINSENTINEL_BRONZE', 'data/bronze'))
VALIDATION_DIR = Path(os.getenv('FINSENTINEL_VALIDATION', 'data/validation'))
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_FIELDS = ['id', 'title', 'summary', 'published', 'source', 'retrieved_at']


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # skip malformed JSON lines but record later
                records.append({'_parse_error': line})
    return records


def validate(date_str: str = None) -> Dict[str, Any]:
    date_str = date_str or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    file = BRONZE_DIR / f"{date_str}.ndjson"
    report = {
        'date': date_str,
        'file': str(file),
        'exists': file.exists(),
        'total_records': 0,
        'parse_errors': 0,
        'missing_counts': {},
        'duplicate_count': 0,
        'sample_issues': [],
    }

    if not file.exists():
        return report

    records = _read_ndjson(file)
    report['total_records'] = len(records)

    ids = []
    seen = set()
    missing_counts = {k: 0 for k in REQUIRED_FIELDS}

    for rec in records:
        if '_parse_error' in rec:
            report['parse_errors'] += 1
            if len(report['sample_issues']) < 5:
                report['sample_issues'].append({'type': 'parse_error', 'line': rec['_parse_error'][:500]})
            continue

        for f in REQUIRED_FIELDS:
            if not rec.get(f):
                missing_counts[f] += 1
                if len(report['sample_issues']) < 5:
                    report['sample_issues'].append({'type': 'missing_field', 'field': f, 'record': rec.get('id')})

        ids.append(rec.get('id'))

        # check timestamp parseability
        try:
            if rec.get('published'):
                pd.to_datetime(rec.get('published'))
        except Exception:
            if len(report['sample_issues']) < 5:
                report['sample_issues'].append({'type': 'bad_published_ts', 'record': rec.get('id')})

        try:
            if rec.get('retrieved_at'):
                pd.to_datetime(rec.get('retrieved_at'))
        except Exception:
            if len(report['sample_issues']) < 5:
                report['sample_issues'].append({'type': 'bad_retrieved_ts', 'record': rec.get('id')})

    report['missing_counts'] = missing_counts

    # duplicates
    dup_count = len(ids) - len(set([i for i in ids if i]))
    report['duplicate_count'] = dup_count

    out_file = VALIDATION_DIR / f"{date_str}.json"
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--date', help='YYYY-MM-DD (UTC) to validate', default=None)
    args = p.parse_args()

    r = validate(args.date)
    print(json.dumps(r, indent=2))
