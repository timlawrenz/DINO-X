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
import sys
import time
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


def download_series_zip(
    base_url: str,
    series_uid: str,
    out_zip: Path,
    api_key: str | None,
) -> None:
    q = urllib.parse.urlencode({"SeriesInstanceUID": series_uid})
    url = f"{base_url}/getImage?{q}"

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url)
    if api_key:
        req.add_header("api_key", api_key)

    with urllib.request.urlopen(req, timeout=300) as r:  # nosec - URL is user provided
        # Stream to disk (no full RAM buffering).
        with out_zip.open("wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


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
    p_dl.add_argument("--sleep", type=float, default=0.0)

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

    if args.cmd == "download-series":
        uids = [ln.strip() for ln in args.uids.read_text().splitlines() if ln.strip()]
        if not uids:
            raise SystemExit(f"No UIDs found in: {args.uids}")

        out_root: Path = args.out_root
        out_root.mkdir(parents=True, exist_ok=True)

        for idx, uid in enumerate(uids, start=1):
            series_dir = out_root / uid
            if series_dir.exists() and any(series_dir.iterdir()):
                print(f"skip=true uid={uid} reason=nonempty_dir")
                continue

            zip_path = out_root / "_zips" / f"{uid}.zip"
            print(f"download_start idx={idx}/{len(uids)} uid={uid}")
            download_series_zip(args.base_url, uid, zip_path, args.api_key)

            series_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(series_dir)
            except zipfile.BadZipFile as e:
                raise SystemExit(f"Bad ZIP for uid={uid} at {zip_path}: {e}")

            if not args.keep_zips:
                try:
                    zip_path.unlink()
                except FileNotFoundError:
                    pass

            print(f"download_ok uid={uid} out={series_dir}")
            if args.sleep:
                time.sleep(args.sleep)

        print("ok=true")
        return 0

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
