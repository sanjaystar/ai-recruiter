"""Stream JSONL/JSONL.gz candidate records one at a time."""

import gzip
import json
import os

def stream_candidates(filepath: str, stats: dict = None):
    """Yield candidate dicts one at a time so the full pool never lives in memory."""
    if stats is None:
        stats = {"loaded": 0, "filtered": 0, "errors": {}}

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith('.gz'):
        open_func = lambda f: gzip.open(f, 'rt', encoding='utf-8')
    else:
        open_func = lambda f: open(f, 'r', encoding='utf-8')

    with open_func(filepath) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    stats["filtered"] += 1
                    continue
                stats["loaded"] += 1
                yield obj
            except json.JSONDecodeError as e:
                stats["filtered"] += 1
                if "decode_errors" not in stats["errors"]:
                    stats["errors"]["decode_errors"] = []
                # Cap logged errors to avoid spam
                if len(stats["errors"]["decode_errors"]) < 10:
                    stats["errors"]["decode_errors"].append(f"Line {line_idx}: {str(e)}")
