#!/usr/bin/env python3
"""Colour conversion helpers for the SAM3 annotation branch.

The working TIFF signal is linear-light and uses the primaries declared by
camera.working_primaries. Annotation images are produced by converting those
linear primaries to linear sRGB first, then applying the IEC 61966-2-1 sRGB
transfer function. Keeping both operations here prevents the quick and full
preprocessing paths from drifting apart.
"""

from pathlib import Path

import numpy as np


# D65 matrices from the published RGB colourspace definitions. Rows multiply
# column RGB/XYZ vectors. BT.2020 and sRGB use the same D65 white, therefore no
# chromatic-adaptation transform is required.
_RGB_TO_XYZ = {
    "srgb": np.array([
        [0.412390799266, 0.357584339384, 0.180480788402],
        [0.212639005872, 0.715168678768, 0.072192315361],
        [0.019330818716, 0.119194779795, 0.950532152250],
    ], dtype=np.float64),
    "bt2020": np.array([
        [0.636958048301, 0.144616903586, 0.168880975164],
        [0.262700212011, 0.677998071519, 0.059301716470],
        [0.000000000000, 0.028072693049, 1.060985057711],
    ], dtype=np.float64),
}
_XYZ_TO_RGB = {name: np.linalg.inv(matrix) for name, matrix in _RGB_TO_XYZ.items()}


def convert_linear_primaries(rgb: np.ndarray, src: str, dst: str = "srgb") -> np.ndarray:
    """Convert a linear RGB array between supported D65 primary sets.

    Values are not clipped because out-of-gamut intermediate values carry
    useful information. The display encoder performs the final gamut clip.
    The input scale (0..1, 0..65535, or another linear scale) is preserved.
    """
    src = str(src).lower()
    dst = str(dst).lower()
    if src not in _RGB_TO_XYZ:
        raise ValueError(f"Unsupported source primaries {src!r}; expected {sorted(_RGB_TO_XYZ)}")
    if dst not in _RGB_TO_XYZ:
        raise ValueError(f"Unsupported destination primaries {dst!r}; expected {sorted(_RGB_TO_XYZ)}")
    arr = np.asarray(rgb)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB data with a final dimension of 3, got {arr.shape}")
    if src == dst:
        return arr.copy()
    transform = _XYZ_TO_RGB[dst] @ _RGB_TO_XYZ[src]
    converted = np.einsum("...c,dc->...d", arr.astype(np.float64), transform)
    return converted.astype(np.result_type(arr.dtype, np.float32), copy=False)


def encode_srgb_u8(rgb_lin: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer function to linear sRGB on a 0..65535 scale."""
    n = np.clip(np.asarray(rgb_lin, dtype=np.float64) / 65535.0, 0.0, 1.0)
    encoded = np.where(
        n <= 0.0031308,
        n * 12.92,
        1.055 * np.power(n, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def linear_working_to_srgb_u8(rgb_lin: np.ndarray, working_primaries: str) -> np.ndarray:
    """Convert linear working primaries to sRGB primaries and transfer."""
    return encode_srgb_u8(convert_linear_primaries(rgb_lin, working_primaries, "srgb"))


class Unexpected16BitInputError(RuntimeError):
    """Raised when a linear/high-bit-depth image is about to reach SAM3."""


def guard_reject_16bit(img: np.ndarray, source: str = "<array>") -> None:
    """Require an HxWx3 uint8 annotation image before it is passed to SAM3."""
    if img.dtype != np.uint8:
        raise Unexpected16BitInputError(
            f"{source}: expected uint8, got {img.dtype}. Convert the linear TIFF "
            "with preprocess.py --output-rgb or quick_tiff_to_sam3.py first."
        )
    if img.ndim != 3 or img.shape[2] != 3:
        raise Unexpected16BitInputError(f"{source}: expected HxWx3 RGB, got {img.shape}.")


def guard_reject_16bit_path(path) -> None:
    """Inspect a TIFF header and reject non-uint8 input before SAM3 decoding."""
    import tifffile

    p = Path(path)
    with tifffile.TiffFile(str(p)) as tif:
        dtype = tif.series[0].dtype
    if dtype != np.uint8:
        raise Unexpected16BitInputError(
            f"{p}: expected uint8, got {dtype}. Run the controlled linear-gamut "
            "conversion and sRGB transfer before using this folder with SAM3."
        )
