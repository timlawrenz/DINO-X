"""Real-slice separation check for the one-line classifier.

Pulls a few labeled vessel-present and vessel-absent CRLM slices and confirms
the one-line predict_proba scores separate them. This is the honest cold check:
does the number move the right way on real data it has a label for?
"""
import csv
import numpy as np
from PIL import Image
from zoo.predict import load_classifier, predict_proba

clf = load_classifier("timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1", device="cuda")

rows = list(csv.DictReader(open("data/crlm/vessel_labels_local.csv")))
pos = [r for r in rows if r["label"] == "1" and int(r["vessel_pixels"]) > 500][:6]
neg = [r for r in rows if r["label"] == "0" and int(r["vessel_pixels"]) == 0][:6]

def score(r):
    arr = np.array(Image.open(r["png_path"]))  # 16-bit PNG
    return predict_proba(clf, arr,
                         pixel_spacing=(float(r["spacing_x"]), float(r["spacing_y"])),
                         slice_thickness=float(r["spacing_z"]),
                         input_format="hu16_png")

print("vessel-PRESENT slices (expect high):")
ps = [score(r) for r in pos]
for r, s in zip(pos, ps):
    print(f"  {s:.3f}  (vessel_pixels={r['vessel_pixels']})")
print("vessel-ABSENT slices (expect low):")
ns = [score(r) for r in neg]
for r, s in zip(neg, ns):
    print(f"  {s:.3f}  (vessel_pixels={r['vessel_pixels']})")

print(f"\nmean present = {np.mean(ps):.3f} | mean absent = {np.mean(ns):.3f}")
print(f"separation (present > absent): {np.mean(ps) > np.mean(ns)}")
print(f"min(present) > max(absent): {min(ps) > max(ns)}")
print("REAL_SLICE_OK")
