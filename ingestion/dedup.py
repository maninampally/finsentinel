import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any


BRONZE_DIR = Path(os.getenv('FINSENTINEL_BRONZE', 'data/bronze'))
DEDUP_DIR = Path(os.getenv('FINSENTINEL_BRONZE_DEDUP', 'data/bronze_dedup'))
REPORT_DIR = Path(os.getenv('FINSENTINEL_VALIDATION', 'data/validation'))

DEDUP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def dedupe(date_str: str = None) -> Dict[str, Any]:
    date_str = date_str or datetime.now(timezone.utc).strftime('%Y-%m-%d')
    src = BRONZE_DIR / f"{date_str}.ndjson"
    dst = DEDUP_DIR / f"{date_str}.ndjson"
    report_file = REPORT_DIR / f"{date_str}.dedup.json"

    report = {
        'date': date_str,
        'source': str(src),
        'dest': str(dst),
        'source_exists': src.exists(),
        'total_in': 0,
        'total_out': 0,
        'duplicates_removed': 0,
    }

    if not src.exists():
        return report

    seen = set()
    written = 0
    total = 0

    with src.open('r', encoding='utf-8') as rf, dst.open('w', encoding='utf-8') as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                # preserve malformed lines in output (counts as kept)
                wf.write(line + '\n')
                written += 1
                continue

            rid = rec.get('id')
            if not rid:
                # if no id, keep the record
                wf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                written += 1
                continue

            if rid in seen:
                continue
            seen.add(rid)
            wf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            written += 1

    report['source_exists'] = src.exists()
    report['total_in'] = total
    report['total_out'] = written
    report['duplicates_removed'] = total - written

    with report_file.open('w', encoding='utf-8') as rf:
        json.dump(report, rf, indent=2)

    return report


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--date', help='YYYY-MM-DD (UTC) to dedupe', default=None)
    args = p.parse_args()

    r = dedupe(args.date)
    print(json.dumps(r, indent=2))
