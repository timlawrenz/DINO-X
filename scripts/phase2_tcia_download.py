#!/usr/bin/env python3
"""Download TCIA/NBIA series via the TCIA REST API.

This is intentionally simple and NAS-friendly:
- lists series for a collection (optional)
- downloads series ZIPs via getImage?SeriesInstanceUID=...
- extracts into an output directory

Docs:
- https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface+REST+API+Guides

Notes:
- Bulk downloads are large and long-running; consider running inside tmux/screen.
- Public collections often work without an API key; restricted collections require auth.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

DEFAULT_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def _http_get_json(url: str, api_key: str | None) -> object:
    req = urllib.request.Request(url)
    if api_key:
        # TCIA docs show api_key as a header in many examples.
        req.add_header("api_key", api_key)
    with urllib.request.urlopen(req, timeout=60) as r:  # nosec - URL is user provided
        return json.loads(r.read().decode("utf-8"))


def iter_series_uids(
    base_url: str,
    collection: str,
    api_key: str | None,
    *,
    modality: str | None = None,
    min_image_count: int = 0,
    sort_by: str = "none",
) -> Iterable[str]:
    params: dict[str, str] = {"Collection": collection, "format": "json"}
    if modality:
        params["Modality"] = modality
    q = urllib.parse.urlencode(params)
    url = f"{base_url}/getSeries?{q}"
    data = _http_get_json(url, api_key)

    if not isinstance(data, list):
        raise SystemExit(f"Unexpected getSeries response shape: {type(data)}")

    items = [x for x in data if isinstance(x, dict) and "SeriesInstanceUID" in x]

    if min_image_count:
        def _imgc(it: dict) -> int:
            try:
                return int(it.get("ImageCount") or 0)
            except Exception:
                return 0

        items = [it for it in items if _imgc(it) >= min_image_count]

    if sort_by == "imagecount":
        def _imgc2(it: dict) -> int:
            try:
                return int(it.get("ImageCount") or 0)
            except Exception:
                return 0

        items.sort(key=_imgc2, reverse=True)

    for item in items:
        yield str(item["SeriesInstanceUID"])


def _download_url_to_path(
    url: str,
    out_path: Path,
    api_key: str | None,
    *,
    resume: bool = True,
) -> None:
    """Download url -> out_path, optionally resuming via HTTP Range.

    We download to a *.part file and rename atomically when complete.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")

    def _request(range_start: int | None) -> urllib.request.Request:
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("api_key", api_key)
        if range_start is not None:
            req.add_header("Range", f"bytes={range_start}-")
        return req

    # If a previous full zip exists, trust it and do nothing.
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    # Resume into the .part file.
    start = part.stat().st_size if (resume and part.exists()) else 0

    # If resume requested, try range first; if server ignores it, restart cleanly.
    for attempt in range(2):
        range_start = start if (resume and start > 0) else None
        req = _request(range_start)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:  # nosec - URL is user provided
                code = getattr(r, "status", None) or r.getcode()
                if range_start is not None and code != 206:
                    # Server didn't honor Range; restart from scratch.
                    try:
                        part.unlink()
                    except FileNotFoundError:
                        pass
                    start = 0
                    continue

                mode = "ab" if range_start is not None else "wb"
                with part.open(mode) as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            part.replace(out_path)
            return
        except urllib.error.HTTPError as e:
            # Some servers respond 416 if the local file is already complete.
            if e.code == 416 and out_path.exists():
                return
            raise

    raise SystemExit(f"Failed to download after retries: {url}")


def _zip_seems_ok(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(p) as z:
            return len(z.namelist()) > 0
    except zipfile.BadZipFile:
        return False


def download_series_zip(
    base_url: str,
    series_uid: str,
    out_zip: Path,
    api_key: str | None,
    *,
    resume: bool = True,
) -> None:
    q = urllib.parse.urlencode({"SeriesInstanceUID": series_uid})
    url = f"{base_url}/getImage?{q}"
    _download_url_to_path(url, out_zip, api_key, resume=resume)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--collection", default="LIDC-IDRI")
    p.add_argument(
        "--api-key",
        default=os.environ.get("TCIA_API_KEY"),
        help="Optional; can also be set via TCIA_API_KEY env var.",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-series")
    _add_common(p_list)
    p_list.add_argument("--out", type=Path, default=Path("series_uids.txt"))
    p_list.add_argument("--modality", default=None, help="Optional (e.g., CT)")
    p_list.add_argument("--min-image-count", type=int, default=0)
    p_list.add_argument("--sort-by", choices=["none", "imagecount"], default="none")

    p_dl = sub.add_parser("download-series")
    _add_common(p_dl)
    p_dl.add_argument(
        "--uids",
        type=Path,
        required=True,
        help="Text file with one SeriesInstanceUID per line.",
    )
    p_dl.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/raw"),
        help="Root dir to extract series under.",
    )
    p_dl.add_argument("--keep-zips", action="store_true")
    p_dl.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior (otherwise partial .zip.part downloads and incomplete extracts are resumed by default).",
    )
    p_dl.add_argument("--sleep", type=float, default=0.0)

    p_dc = sub.add_parser("download-collection")
    _add_common(p_dc)
    p_dc.add_argument("--out-root", type=Path, default=Path("data/raw"), help="Root dir to extract series under.")
    p_dc.add_argument("--out-uids", type=Path, default=None, help="Optional: write resolved UIDs here.")
    p_dc.add_argument("--modality", default=None, help="Optional (e.g., CT)")
    p_dc.add_argument("--min-image-count", type=int, default=0)
    p_dc.add_argument("--sort-by", choices=["none", "imagecount"], default="none")
    p_dc.add_argument("--keep-zips", action="store_true")
    p_dc.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior (otherwise partial .zip.part downloads and incomplete extracts are resumed by default).",
    )
    p_dc.add_argument("--sleep", type=float, default=0.0)

    args = ap.parse_args()

    if args.cmd == "list-series":
        uids = list(
            iter_series_uids(
                args.base_url,
                args.collection,
                args.api_key,
                modality=args.modality,
                min_image_count=args.min_image_count,
                sort_by=args.sort_by,
            )
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(uids) + "\n")
        print("ok=true")
        print(f"collection={args.collection}")
        print(f"count={len(uids)}")
        print(f"out={args.out}")
        return 0

    def _download_many(uids: list[str], out_root: Path) -> None:
        out_root.mkdir(parents=True, exist_ok=True)

        resume_enabled = not getattr(args, "no_resume", False)

        for idx, uid in enumerate(uids, start=1):
            series_dir = out_root / uid
            complete_marker = series_dir / ".complete"
            if complete_marker.exists():
                print(f"skip=true uid={uid} reason=complete")
                continue

            zip_path = out_root / "_zips" / f"{uid}.zip"
            zip_part = zip_path.with_suffix(zip_path.suffix + ".part")

            if series_dir.exists() and any(series_dir.iterdir()):
                if resume_enabled and (zip_path.exists() or zip_part.exists()):
                    print(f"resume=true uid={uid} action=rm_incomplete_dir")
                    shutil.rmtree(series_dir)
                else:
                    # Likely a complete prior extract (older runs before .complete marker existed).
                    complete_marker.write_text("ok\n")
                    print(f"skip=true uid={uid} reason=assume_complete")
                    continue

            print(f"download_start idx={idx}/{len(uids)} uid={uid}")
            download_series_zip(args.base_url, uid, zip_path, args.api_key, resume=resume_enabled)

            if not _zip_seems_ok(zip_path):
                raise SystemExit(f"Bad ZIP for uid={uid} at {zip_path}")

            tmp_dir = out_root / "_tmp_extract" / uid
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.parent.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmp_dir)
            except zipfile.BadZipFile as e:
                raise SystemExit(f"Bad ZIP for uid={uid} at {zip_path}: {e}")

            series_dir.parent.mkdir(parents=True, exist_ok=True)
            if series_dir.exists():
                shutil.rmtree(series_dir)
            tmp_dir.replace(series_dir)
            complete_marker.write_text("ok\n")

            if not getattr(args, "keep_zips", False):
                try:
                    zip_path.unlink()
                except FileNotFoundError:
                    pass

            print(f"download_ok uid={uid} out={series_dir}")
            if getattr(args, "sleep", 0.0):
                time.sleep(args.sleep)

    if args.cmd == "download-series":
        uids = [ln.strip() for ln in args.uids.read_text().splitlines() if ln.strip()]
        if not uids:
            raise SystemExit(f"No UIDs found in: {args.uids}")
        _download_many(uids, args.out_root)
        print("ok=true")
        return 0

    if args.cmd == "download-collection":
        uids = list(
            iter_series_uids(
                args.base_url,
                args.collection,
                args.api_key,
                modality=args.modality,
                min_image_count=args.min_image_count,
                sort_by=args.sort_by,
            )
        )
        if args.out_uids:
            args.out_uids.parent.mkdir(parents=True, exist_ok=True)
            args.out_uids.write_text("\n".join(uids) + "\n")
        _download_many(uids, args.out_root)
        print("ok=true")
        return 0

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
