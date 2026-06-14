"""
loader.py
Efficiently streams JSONL candidate records to avoid excessive memory consumption.
"""

import gzip
import json
import os

def stream_candidates(filepath: str, stats: dict = None):
    """
    Generator that yields candidate dictionaries one by one.
    Transparently handles both raw .jsonl, .json arrays, and compressed .gz files.
    """
    if stats is None:
        stats = {"loaded": 0, "filtered": 0, "errors": {}}

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith('.gz'):
        open_func = lambda f: gzip.open(f, 'rt', encoding='utf-8')
    else:
        open_func = lambda f: open(f, 'r', encoding='utf-8')

    # Read first character to determine format
    with open_func(filepath) as f:
        first_char = ""
        while True:
            char = f.read(1)
            if not char:
                break
            if char.strip():
                first_char = char
                break
                
    if first_char == '[':
        # JSON Array format (loads entirely into memory, which is fine for < 16GB)
        print("Detected JSON Array format. Loading entire array...")
        with open_func(filepath) as f:
            try:
                data = json.load(f)
                for item in data:
                    if not isinstance(item, dict):
                        stats["filtered"] += 1
                        continue
                    stats["loaded"] += 1
                    yield item
            except json.JSONDecodeError as e:
                stats["filtered"] += 1
                stats["errors"]["json_array_error"] = str(e)
    else:
        # JSONL format (one object per line)
        print("Detected JSONL format. Streaming line-by-line...")
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
                    error_msg = f"Line {line_idx}: {str(e)}"
                    if "decode_errors" not in stats["errors"]:
                        stats["errors"]["decode_errors"] = []
                    # Limit error messages logged to avoid spam
                    if len(stats["errors"]["decode_errors"]) < 10:
                        stats["errors"]["decode_errors"].append(error_msg)

def batch_stream_candidates(filepath: str, batch_size: int = 5000, stats: dict = None):
    """
    Yields lists of candidates in chunks to support vectorized batch processing.
    """
    batch = []
    for candidate in stream_candidates(filepath, stats):
        batch.append(candidate)
        if len(batch) >= batch_size:
            yield batch
            batch = []
            
    if batch:
        yield batch
