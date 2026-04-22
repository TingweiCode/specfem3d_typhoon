from __future__ import annotations

import argparse
from pathlib import Path


INPUT_STATIONS = Path("DATA/STATIONS")
OUTPUT_STATIONS = Path("DATA/STATIONS")
NUM_X = 100
NUM_Z = 100
X_MIN = -20_000.0
X_MAX = 20_000.0
Z_MIN = -25_000.0
Z_MAX = 0.0
Y_VALUE = 0.0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate a 2-D SPECFEM station grid in (y, x, 0, z) format "
			"using DATA/STATIONS as the template for station/network metadata."
		)
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=INPUT_STATIONS,
		help="Template station file used to infer the network code.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=OUTPUT_STATIONS,
		help="Output file for the generated 2-D station grid.",
	)
	parser.add_argument(
		"--network",
		default=None,
		help="Override the network code instead of using the first station entry.",
	)
	return parser.parse_args()


def first_nonempty_line(path: Path) -> str:
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			stripped = line.strip()
			if stripped:
				return stripped
	raise ValueError(f"No station entries found in {path}")


def infer_network(path: Path) -> str:
	parts = first_nonempty_line(path).split()
	if len(parts) < 2:
		raise ValueError(
			f"Expected at least two columns in template station file {path}"
		)
	return parts[1]


def linspace_inclusive(start: float, stop: float, count: int) -> list[float]:
	if count < 2:
		return [start]
	step = (stop - start) / (count - 1)
	return [start + index * step for index in range(count)]


def linspace_open(start: float, stop: float, count: int) -> list[float]:
	step = (stop - start) / (count + 1)
	return [start + (index + 1) * step for index in range(count)]


def station_name(ix: int, iz: int) -> str:
	return f"X{ix:03d}Z{iz:03d}"


def format_station_line(name: str, network: str, y: float, x: float, z: float) -> str:
	return f"{name:<8s} {network:<4s} {y:12.1f} {x:12.1f} {0.0:8.1f} {z:12.1f}\n"


def generate_station_lines(network: str) -> list[str]:
	x_values = linspace_inclusive(X_MIN, X_MAX, NUM_X)
	z_values = linspace_open(Z_MIN, Z_MAX, NUM_Z)
	lines: list[str] = []

	for iz, z_value in enumerate(z_values):
		for ix, x_value in enumerate(x_values):
			lines.append(
				format_station_line(
					station_name(ix, iz),
					network,
					Y_VALUE,
					x_value,
					z_value,
				)
			)

	return lines


def main() -> None:
	args = parse_args()
	network = args.network or infer_network(args.input)
	lines = generate_station_lines(network)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w", encoding="utf-8") as handle:
		handle.writelines(lines)

	print(
		f"Wrote {len(lines)} stations to {args.output} "
		f"with network '{network}'."
	)


if __name__ == "__main__":
	main()
