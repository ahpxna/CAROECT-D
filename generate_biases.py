#!/usr/bin/env python3
"""Generate recorder-compatible IMX636 legacy .bias sweep files.

The format is plain text, one value-percent-name entry per line, as accepted
by Metavision Biases::set_from_file(). Values are relative offsets around the
factory-trimmed IMX636 defaults.
"""

import argparse
import itertools
from pathlib import Path


IMX636_RANGES = {
    "bias_diff": (-25, 23),
    "bias_diff_on": (-85, 140),
    "bias_diff_off": (-35, 190),
    "bias_fo": (-35, 55),
    "bias_hpf": (0, 120),
    "bias_refr": (-20, 235),
}
DIMENSION_TO_BIAS = {
    "diff_on": "bias_diff_on",
    "diff_off": "bias_diff_off",
    "fo": "bias_fo",
    "hpf": "bias_hpf",
    "refr": "bias_refr",
}
DEFAULT_BIASES = {
    "bias_diff": 0,
    "bias_diff_off": 0,
    "bias_diff_on": 0,
    "bias_fo": 0,
    "bias_hpf": 0,
    "bias_refr": 0,
}


def validate_biases(values: dict[str, int]) -> None:
    """Reject unknown names and offsets outside documented IMX636 ranges."""
    for name, value in values.items():
        if name not in IMX636_RANGES:
            raise ValueError(f"Unknown IMX636 bias {name!r}")
        lo, hi = IMX636_RANGES[name]
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name}={value} is outside the IMX636 range [{lo}, {hi}]")


def format_bias_text(values: dict[str, int]) -> str:
    """Return deterministic legacy .bias text after range validation."""
    validate_biases(values)
    ordered = [name for name in DEFAULT_BIASES if name in values]
    return "".join(f"{int(values[name])} % {name}\n" for name in ordered)


def parse_bias_text(text: str) -> dict[str, int]:
    """Parse the legacy text format used by the recorder and SDK."""
    parsed = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("%")]
        if len(parts) != 2:
            raise ValueError(f"Line {line_number}: expected value % bias_name")
        value, name = parts
        parsed[name] = int(value)
    validate_biases(parsed)
    return parsed


def write_bias_file(path: Path, values: dict[str, int]) -> None:
    path.write_text(format_bias_text(values))


def _parse_dimension(value: str) -> tuple[str, list[int]]:
    try:
        name, raw_values = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use dimension=value,value, for example fo=-10,0,10") from exc
    if name not in DIMENSION_TO_BIAS:
        raise argparse.ArgumentTypeError(f"Unknown dimension {name!r}; choose {sorted(DIMENSION_TO_BIAS)}")
    try:
        values = [int(item) for item in raw_values.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Non-integer value in {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"No values supplied for {name!r}")
    return name, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="biases", help="Output directory")
    parser.add_argument(
        "--dimension",
        action="append",
        type=_parse_dimension,
        default=[],
        help="Named sweep dimension, e.g. diff_on=-20,0,20; may be repeated",
    )
    args = parser.parse_args()

    dimensions = dict(args.dimension) or {
        "diff_on": [-20, 20, 50, 80],
        "diff_off": [-20, 20, 50, 80],
        "fo": [0, 20],
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(dimensions)
    count = 0
    for combination in itertools.product(*(dimensions[name] for name in names)):
        values = DEFAULT_BIASES.copy()
        tokens = []
        for name, value in zip(names, combination):
            values[DIMENSION_TO_BIAS[name]] = value
            tokens.append(f"{name}_{value}")
        write_bias_file(output_dir / ("_".join(tokens) + ".bias"), values)
        count += 1
    print(f"Created {count} recorder-compatible IMX636 bias files in {output_dir}/")


if __name__ == "__main__":
    main()
