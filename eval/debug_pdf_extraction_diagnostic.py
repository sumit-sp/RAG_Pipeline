"""TEMPORARY diagnostic — not part of the eval suite. Prints extraction/chunking
fingerprints for every PDF in data/raw/, to compare Windows (local) vs Linux (CI)
output and pin down why CI's retrieval-hit rate is lower than the local number
despite identical code and config. Delete this file and its CI step once the
mechanism is confirmed — see EVALUATION_HISTORY.md Step 15 follow-up.
"""

import hashlib
import platform
import sys
from pathlib import Path

import pymupdf

from app.core import config
from app.pipelines.plain.chunking import chunk_text
from app.pipelines.plain.contextual_headers import load_headers

print(f"platform: {platform.platform()}")
print(f"python: {sys.version}")
print(f"pymupdf: {pymupdf.__version__}")
print()

headers = load_headers(Path(config.CONTEXTUAL_HEADERS_PATH))
raw_dir = Path(config.DATA_RAW_DIR)

for path in sorted(raw_dir.rglob("*.pdf")):
    source_doc = str(path.relative_to(raw_dir).as_posix())
    with pymupdf.open(path) as doc:
        text = " ".join(page.get_text() for page in doc)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    chunks = list(chunk_text(text))
    matched = sum(1 for i in range(len(chunks)) if headers.get((source_doc, i)))
    print(
        f"{source_doc}: text_len={len(text)} sha256={digest} "
        f"chunks={len(chunks)} headers_matched={matched}/{len(chunks)}"
    )
