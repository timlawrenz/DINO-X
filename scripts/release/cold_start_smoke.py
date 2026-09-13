"""Cold-PhD smoke test for the dinox one-line classifier.

Run on a machine with the model's deps installed. Exercises exactly the path a
stranger would take: load_classifier(model_id) -> predict_proba(slice).
"""
import numpy as np
from zoo.predict import load_classifier, predict_proba

MODEL = "timlawrenz/dinox-ct-msd-hepatic-vessel-vit-base-v1"

clf = load_classifier(MODEL, device="cuda")
print("window baked in:", clf.window_level, clf.window_width,
      "| classes:", clf.num_classes, "| head in_features:", clf.head.in_features)

# synthetic HU slice -> float in [0,1]
rng = np.random.default_rng(0)
hu = rng.uniform(-200, 200, size=(512, 512)).astype(np.float32)
p = predict_proba(clf, hu, pixel_spacing=(0.77, 0.77), slice_thickness=1.5)
print("synthetic P(vessel) =", round(p, 4), "| in [0,1]:", 0.0 <= p <= 1.0)

# determinism
p2 = predict_proba(clf, hu, pixel_spacing=(0.77, 0.77), slice_thickness=1.5)
print("deterministic:", abs(p - p2) < 1e-6)

# hu16_png input path (matches training input_format) should agree with hu_float
png16 = ((hu / 0.1) + 32768.0).clip(0, 65535).astype(np.uint16)
p3 = predict_proba(clf, png16, pixel_spacing=(0.77, 0.77), slice_thickness=1.5,
                   input_format="hu16_png")
print("hu16_png path P =", round(p3, 4), "| matches hu_float:", abs(p - p3) < 1e-3)
print("SMOKE_OK")
