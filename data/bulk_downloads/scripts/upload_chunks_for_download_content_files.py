#!/usr/bin/env python3
"""Deprecated wrapper for sync script.

Use `data/bulk_downloads/scripts/sync_with_sk.py` instead.
"""

from __future__ import annotations

import warnings

from sync_with_sk import main


if __name__ == "__main__":
    warnings.warn(
        "upload_chunks_for_download_content_files.py is deprecated; use sync_with_sk.py",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
