python - <<'PY'
import cv2
from pathlib import Path

tests = [(7,7), (6,7), (5,7), (4,7),
         (7,6), (7,5), (7,4)]

for p in sorted(Path("calibration/chessboard").glob("*.tiff")):
    img = cv2.imread(str(p))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = []
    for pat in tests:
        ok, _ = cv2.findChessboardCornersSB(
            gray, pat,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                  cv2.CALIB_CB_EXHAUSTIVE
        )
        if ok:
            found.append(pat)

    print(p.name, found)
PY
