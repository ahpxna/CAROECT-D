import cv2

path = "calibration/gray_card.tiff"

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

# Chỉ để hiển thị nếu TIFF 16-bit
show = (img / 256).astype("uint8") if img.dtype == "uint16" else img

x, y, w, h = cv2.selectROI("Select gray patch", show, False, False)

print(f"gray_card_roi: [{x}, {y}, {w}, {h}]")

cv2.destroyAllWindows()