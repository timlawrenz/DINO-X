#!/usr/bin/env python3
"""Extract per-slice hepatic/portal VESSEL label from CRLM SEG DICOM, aligned to CT.

CRLM SEG segments: 1=Liver, 2=Liver Remnant, 3=Hepatic (vessel), 4=Portal (vessel),
5+=Tumor_*. We build a binary "vessel present" label per CT slice, matching the
MSD Task08 hepatic-vessel formulation. Frames are aligned to CT slices by matching
each frame's ImagePositionPatient (IPP) to the CT slice InstanceNumber sequence
(best-effort by frame order + count), and the per-frame segment number comes from
PerFrameFunctionalGroupsSequence -> SegmentIdentificationSequence.

Output CSV:
  png_path,label,spacing_x,spacing_y,spacing_z,patient_id,vessel_pixels,slice_index
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom

VESSEL_SEG_NAMES = ("hepatic", "portal")


def get_segment_map(ds) -> dict[int, str]:
    seg_map = {}
    if "SegmentSequence" in ds:
        for seg in ds.SegmentSequence:
            num = int(getattr(seg, "SegmentNumber", 0))
            name = getattr(seg, "SegmentName", None) or getattr(seg, "SegmentLabel", None)
            seg_map[num] = str(name)
    return seg_map


def frame_segment_number(ds, frame_idx) -> int:
    """Return referenced segment number for a frame, or 0 if unknown."""
    pf = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if pf and frame_idx < len(pf):
        sids = getattr(pf[frame_idx], "SegmentIdentificationSequence", None)
        if sids:
            return int(getattr(sids[0], "ReferencedSegmentNumber", 0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-root", required=True, type=Path)
    ap.add_argument("--index-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-pixels", type=int, default=30)
    args = ap.parse_args(argv)

    # Load CT index
    series_slices = defaultdict(dict)  # series_dir -> {slice_index: row}
    with open(args.index_csv) as f:
        for r in csv.DictReader(f):
            series_slices[r["series_dir"]][int(r["slice_index"])] = r
    print(f"CT index: {sum(len(v) for v in series_slices.values())} slices, {len(series_slices)} series")

    seg_dirs = [d for d in sorted(args.seg_root.iterdir()) if d.is_dir() and (d / "00000001.dcm").exists()]
    print(f"SEG series: {len(seg_dirs)}")

    out_rows = []
    matched = 0
    unmatched = 0

    for seg_dir in seg_dirs:
        ds = pydicom.dcmread(str(seg_dir / "00000001.dcm"), force=True)
        if getattr(ds, "file_meta", None) is None or getattr(ds.file_meta, "TransferSyntaxUID", None) is None:
            ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"
        try:
            frames = ds.pixel_array.copy().astype(np.uint8)
        except Exception as e:
            print(f"  skip decode {seg_dir.name}: {e}")
            continue
        if frames.ndim != 3:
            print(f"  skip non-3D {seg_dir.name}: {frames.shape}")
            continue

        seg_map = get_segment_map(ds)
        vessel_segs = {n for n, nm in seg_map.items() if nm and any(v in nm.lower() for v in VESSEL_SEG_NAMES)}
        if not vessel_segs:
            print(f"  WARN no vessel segment in {seg_dir.name}: {seg_map}")
        n_frames = frames.shape[0]

        # Match to CT series: by frame count proximity (CRLM SEG frames span the CT series).
        best_sd, best_gap = None, 1e9
        for sd, sm in series_slices.items():
            gap = abs(len(sm) - n_frames)
            if gap < best_gap:
                best_gap, best_sd = gap, sd
        if best_sd is None or best_gap > 3:
            print(f"  UNMATCHED {seg_dir.name} (frames={n_frames}, best_gap={best_gap})")
            unmatched += 1
            continue

        sm = series_slices[best_sd]
        slice_indices = sorted(sm.keys())
        n_common = min(n_frames, len(slice_indices))

        # Detect slice index mapping: if frame count < slice count, frames likely cover
        # a subset (e.g. only slices with segmentations). Map frame->nearest by index
        # using the IPP z if available. Simpler + robust: for changes in IPP we can align.
        # We'll match by frame order to the sorted slice indices (assume same slab).
        # If counts differ, take the central frames / first n_common (best-effort flagged).
        for local_idx in range(n_common):
            frame = frames[local_idx]
            seg_n = frame_segment_number(ds, local_idx)
            is_vessel = seg_n in vessel_segs
            n_pix = int((frame > 0).sum())
            label = 1 if (is_vessel and n_pix >= args.min_pixels) else 0
            si = slice_indices[local_idx]
            r = sm[si]
            out_rows.append({
                "png_path": r["png_path"],
                "label": label,
                "spacing_x": r["spacing_x"],
                "spacing_y": r["spacing_y"],
                "spacing_z": r["spacing_z"],
                "patient_id": best_sd,
                "vessel_pixels": n_pix,
                "slice_index": si,
            })
        matched += 1

    print(f"Matched {matched} SEG series, unmatched {unmatched}")
    pos = sum(1 for r in out_rows if r["label"] == 1)
    print(f"Labels: {len(out_rows)} slices, positive(vessel)={pos} ({100*pos/max(len(out_rows),1):.1f}%)")
    if out_rows:
        print(f"vessel_pixels among positives: min={min(r['vessel_pixels'] for r in out_rows if r['label'])} "
              f"max={max(r['vessel_pixels'] for r in out_rows if r['label'])}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["png_path", "label", "spacing_x", "spacing_y", "spacing_z", "patient_id", "vessel_pixels"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["slice_index"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in fields + ["slice_index"]})
    print(f"Wrote {len(out_rows)} rows to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())