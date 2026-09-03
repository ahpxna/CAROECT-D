import cv2
import tifffile
import numpy as np
from pathlib import Path

path = Path("calibration/gray_card.tiff")

print("Path:", path.resolve())
print("Exists:", path.exists())

img = tifffile.imread(path)

print("dtype:", img.dtype)
print("shape:", img.shape)
print("min/max:", img.min(), img.max())

# Convert chỉ để HIỂN THỊ, không sửa TIFF gốc
if img.dtype == np.uint16:
    show = (img / 257).astype(np.uint8)
else:
    show = img.astype(np.uint8)

# TIFF có thể là RGB, OpenCV display mong BGR
if show.ndim == 3 and show.shape[2] >= 3:
    show = cv2.cvtColor(show[:, :, :3], cv2.COLOR_RGB2BGR)

x, y, w, h = cv2.selectROI(
    "Select gray patch",
    show,
    showCrosshair=True,
    fromCenter=False
)

print(f"gray_card_roi: [{x}, {y}, {w}, {h}]")

cv2.destroyAllWindows()