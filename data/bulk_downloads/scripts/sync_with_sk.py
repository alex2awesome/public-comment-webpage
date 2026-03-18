#!/usr/bin/env python3
"""Bidirectional sync utility for data/bulk_downloads."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_REMOTE = "sk2"
DEFAULT_REMOTE_ROOT = "/lfs/skampere2/0/alexspan/regulations-demo"
REMOTE_BULK_REL = "data/bulk_downloads"
SERVER_MAP = {
    "sk2": {
        "host": "sk2",
        "path": "/lfs/skampere2/0/alexspan/regulations-demo",
    },
    "sk3": {
        "host": "sk3",
        "path": "/lfs/skampere3/0/alexspan/regulations-demo",
    },
}

DATA_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".json",
    ".jsonl",
    ".txt",
    ".gz",
    ".zip",
    ".tar",
}
CODE_EXTENSIONS = {
    ".py",
    ".sh",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".ipynb",
}


def find_repo_root(start: Path) -> Path:
    candidates = [start] + list(start.parents)
    # First pass: look for the sync_with_sk package (dir with __init__.py).
    for candidate in candidates:
        pkg = candidate / "sync_with_sk"
        if pkg.is_dir() and (pkg / "__init__.py").exists():
            return candidate
    # Fallback: nearest .git root.
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Unable to locate repository root (missing .git directory).")


def is_year_dir(name: str) -> bool:
    return bool(re.match(r"^[a-z0-9]+_\d{4}_\d{4}$", name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General upload/download sync for code and data files in data/bulk_downloads."
    )
    parser.add_argument("--upload", action="store_true", help="Sync local files to remote.")
    parser.add_argument("--download", action="store_true", help="Sync remote files to local.")
    parser.add_argument(
        "--hosts",
        choices=sorted(SERVER_MAP.keys()),
        default=None,
        help=(
            "Named server shortcut. "
            "sk2 => host=sk2, root=/lfs/skampere2/0/alexspan/regulations-demo; "
            "sk3 => host=sk3, root=/lfs/skampere3/0/alexspan/regulations-demo."
        ),
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help=f"SSH host alias (default: {DEFAULT_REMOTE}).",
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Destination/source root on remote (default: {DEFAULT_REMOTE_ROOT}).",
    )
    parser.add_argument("--data-only", action="store_true", help="Only consider data files.")
    parser.add_argument("--csv-only", action="store_true", help="Only consider CSV files.")
    parser.add_argument("--code-only", action="store_true", help="Only consider code files.")
    parser.add_argument(
        "--all-text-only",
        action="store_true",
        help="Only consider files ending with all_text.csv (data-only filter).",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Case-insensitive substring filter on relative paths. Repeatable.",
    )
    parser.add_argument(
        "--speed-download",
        action="store_true",
        help=(
            "Fast data mode: if any file differs in an agency folder, transfer the "
            "whole agency directory via tar stream."
        ),
    )
    parser.add_argument(
        "--speed-scope",
        choices=["agency", "agency-year"],
        default="agency",
        help="Granularity for --speed-download tar grouping (default: agency).",
    )
    parser.add_argument(
        "--delete-remote-only",
        action="store_true",
        help="Delete remote-only files after confirmation.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full scan (ignore last-sync timestamp).",
    )
    parser.add_argument("--yes", action="store_true", help="Auto-confirm destructive actions.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing.")
    parser.add_argument(
        "--max-hash-bytes",
        type=int,
        default=None,
        help="Hash files <= this size unless treated as data files.",
    )
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=None,
        help="Skip downloading files larger than this size. Set to 0 to disable.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to state cache JSON file. Defaults to <scope-root>/.sync_with_sk_state.json",
    )
    parser.add_argument(
        "--last-sync-file",
        default=str(SCRIPT_DIR / ".sync_with_sk_last_sync.json"),
        help="Path to last-sync timestamp file. Defaults to <script-dir>/.sync_with_sk_last_sync.json",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--retries", type=int, default=3, help="Retries for transfer operations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Project paths are always relative to this script — independent of cwd.
    # SCRIPT_DIR = .../regulations-demo/data/bulk_downloads/scripts/
    scope_root = SCRIPT_DIR.parent          # .../data/bulk_downloads/
    project_root = SCRIPT_DIR.parent.parent.parent  # .../regulations-demo/

    # Import path: find where the sync_with_sk package lives.
    pkg_root = find_repo_root(Path(__file__).resolve())
    sys.path.insert(0, str(pkg_root))

    try:
        from sync_with_sk.core import ProjectConfig, run_sync  # noqa: WPS433
    except ModuleNotFoundError:
        from dataclasses import dataclass
        import fnmatch
        import subprocess
        import tempfile

        @dataclass
        class ProjectConfig:
            name: str
            local_root: Path
            scope_root: Path
            default_hosts: list[str]
            host_roots: dict[str, str]
            data_dirs: set
            data_suffixes: set[str]
            data_allowlist: set[str]
            code_suffixes: set[str]
            include_unknown: bool
            ignore_paths: list[Path]
            max_hash_bytes: int | None
            max_download_bytes: int | None
            log_csv_path: Path
            state_file_path: Path
            speed_grouping: str
            speed_applies_to_upload: bool
            code_only_skip_dir_predicate: callable

        def _load_ignore_patterns(ignore_paths: list[Path]) -> list[str]:
            patterns: list[str] = []
            for path in ignore_paths:
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    patterns.append(line)
            return patterns

        def _match_filters(rel_path: str, filters: list[str]) -> bool:
            lowered = rel_path.lower()
            return all(f.lower() in lowered for f in filters)

        def _iter_local_files(config: ProjectConfig, args: argparse.Namespace) -> list[str]:
            ignore_patterns = _load_ignore_patterns(config.ignore_paths)
            rel_paths: list[str] = []
            for path in config.scope_root.rglob("*"):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(config.scope_root).as_posix()
                if any(fnmatch.fnmatch(rel_path, pat) for pat in ignore_patterns):
                    continue
                if args.all_text_only and not rel_path.endswith("all_text.csv"):
                    continue
                if args.csv_only and not rel_path.endswith(".csv"):
                    continue
                if args.data_only and path.suffix not in config.data_suffixes:
                    continue
                if args.code_only and path.suffix not in config.code_suffixes:
                    continue
                if args.filters and not _match_filters(rel_path, args.filters):
                    continue
                rel_paths.append(rel_path)
            return rel_paths

        def _build_selection_includes(config: ProjectConfig, args: argparse.Namespace) -> list[str]:
            if args.all_text_only:
                return ["*all_text.csv"]
            if args.csv_only:
                return ["*.csv"]
            if args.data_only:
                return [f"*{ext}" for ext in sorted(config.data_suffixes)]
            if args.code_only:
                return [f"*{ext}" for ext in sorted(config.code_suffixes)]
            return []

        def run_sync(config: ProjectConfig, args: argparse.Namespace) -> int:
            if not args.upload and not args.download:
                raise SystemExit("Must pass --upload or --download")
            collect_deletes = bool(getattr(args, "_collect_deletes", False))

            remote_root = args.remote_root.rstrip("/")
            remote_path = f"{args.remote}:{remote_root}/{REMOTE_BULK_REL}"
            local_path = str(config.scope_root) + "/"

            ignore_patterns = _load_ignore_patterns(config.ignore_paths)
            selection_includes = _build_selection_includes(config, args)

            use_files_from = bool(args.filters)
            files_from = None

            if use_files_from:
                rel_paths = _iter_local_files(config, args)
                if not rel_paths:
                    print("No matching files to sync.")
                    return 0
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                    tmp.write("\n".join(rel_paths))
                    files_from = tmp.name

            rsync_cmd = ["rsync", "-az"]
            if args.dry_run:
                rsync_cmd.append("--dry-run")
            if args.delete_remote_only:
                rsync_cmd.append("--delete")

            exclude_file = None
            if ignore_patterns:
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                    tmp.write("\n".join(ignore_patterns))
                    exclude_file = tmp.name
                rsync_cmd.extend(["--exclude-from", exclude_file])

            if use_files_from:
                rsync_cmd.extend(["--files-from", files_from, "--relative"])
            elif selection_includes:
                rsync_cmd.append("--include=*/")
                for pattern in selection_includes:
                    rsync_cmd.append(f"--include={pattern}")
                rsync_cmd.append("--exclude=*")

            if args.upload:
                rsync_cmd.extend([local_path, remote_path])
            else:
                rsync_cmd.extend([remote_path, local_path])

            if args.verbose or collect_deletes:
                rsync_cmd.extend(
                    [
                        "--itemize-changes",
                        "--progress",
                        "--stats",
                        "--human-readable",
                        "--info=flist2,name2,stats2,progress2",
                    ]
                )

            if args.verbose:
                print("Sync scope:", local_path, "->", remote_path if args.upload else "<-")
                if selection_includes:
                    print("Include patterns:", selection_includes)
                if args.filters:
                    print("Python filters:", args.filters)
                if ignore_patterns:
                    print("Ignore patterns:", ignore_patterns)
                print(" ".join(rsync_cmd))

            try:
                if collect_deletes:
                    result = subprocess.run(
                        rsync_cmd,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    output = (result.stdout or "") + "\n" + (result.stderr or "")
                    delete_lines = [
                        line for line in output.splitlines()
                        if line.startswith("*deleting ") or line.startswith("deleting ")
                    ]
                    if delete_lines:
                        print("Files slated for deletion (dry-run):")
                        for line in delete_lines:
                            print(line)
                    else:
                        print("No remote-only deletions detected (dry-run).")
                    return result.returncode

                result = subprocess.run(rsync_cmd, check=False)
                return result.returncode
            finally:
                if files_from:
                    Path(files_from).unlink(missing_ok=True)
                if exclude_file:
                    Path(exclude_file).unlink(missing_ok=True)

    if args.hosts:
        args.remote = SERVER_MAP[args.hosts]["host"]
        args.remote_root = SERVER_MAP[args.hosts]["path"]

    args.hosts = [args.remote]

    if not scope_root.exists():
        raise SystemExit(f"Local bulk path does not exist: {scope_root}")

    config = ProjectConfig(
        name="regulations-demo",
        local_root=project_root,
        scope_root=scope_root,
        default_hosts=[DEFAULT_REMOTE],
        host_roots={
            "sk2": "/lfs/skampere2/0/alexspan/regulations-demo",
            "sk3": "/lfs/skampere3/0/alexspan/regulations-demo",
        },
        data_dirs=set(),
        data_suffixes=set(DATA_EXTENSIONS),
        data_allowlist=set(DATA_EXTENSIONS),
        code_suffixes=set(CODE_EXTENSIONS),
        include_unknown=False,
        ignore_paths=[SCRIPT_DIR / ".sync-ignore", project_root / ".sync-ignore"],
        max_hash_bytes=64 * 1024 * 1024,
        max_download_bytes=None,
        log_csv_path=SCRIPT_DIR / "sync_with_sk.log.csv",
        state_file_path=scope_root / ".sync_with_sk_state.json",
        speed_grouping="agency-year",
        speed_applies_to_upload=True,
        code_only_skip_dir_predicate=is_year_dir,
    )

    if args.upload and not args.dry_run:
        preview_args = argparse.Namespace(**vars(args))
        preview_args.dry_run = True
        preview_args.yes = True
        preview_args.delete_remote_only = True
        preview_args._collect_deletes = True
        preview_args.verbose = False
        print("Previewing deletions (dry-run). Review the list above.")
        run_sync(config, preview_args)
        response = input("Proceed with deleting remote-only files? [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            args.delete_remote_only = True
            args.yes = True
        else:
            args.delete_remote_only = False

    return run_sync(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
