from __future__ import annotations

import argparse
import os
from pathlib import Path

from create_free_surface_fields import lonlatr2xyz, read_config_env
from holland_model import R_EARTH_M, destination_point

import numpy as np
import h5py 

INPUT_STATIONS = Path("DATA/STATIONS")
OUTPUT_STATIONS = Path("DATA/STATIONS")
NUM_ALONG_TRACK = 100
NUM_DEPTH = 100
SOUTH_KM = 100.0  # extent behind the storm start, opposite the wind azimuth
NORTH_KM = 600.0  # extent ahead of the storm start, along the wind azimuth
DEPTH_MIN_KM = 0.0
DEPTH_MAX_KM = 100.0


def parse_args() -> argparse.Namespace:
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    default_config = base_dir / "config.env"

    parser = argparse.ArgumentParser(
        description=(
            "Generate a 2-D SPECFEM station grid along the wind azimuth "
            "(from config.env), converted from lon/lat/depth to ECEF x/y/z."
        )
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=default_config,
        help="Path to config.env (WIND_LON/WIND_LAT/WIND_AZI).",
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
    parser.add_argument(
        "--south-km", type=float, default=SOUTH_KM, help="Extent behind the storm start (opposite azimuth), km."
    )
    parser.add_argument(
        "--north-km", type=float, default=NORTH_KM, help="Extent ahead of the storm start (along azimuth), km."
    )
    parser.add_argument("--depth-min-km", type=float, default=DEPTH_MIN_KM, help="Shallowest depth, km.")
    parser.add_argument("--depth-max-km", type=float, default=DEPTH_MAX_KM, help="Deepest depth, km.")
    parser.add_argument("--num-along-track", type=int, default=NUM_ALONG_TRACK, help="Number of along-track samples.")
    parser.add_argument("--num-depth", type=int, default=NUM_DEPTH, help="Number of depth samples.")
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


def generate_station_lines(
    network: str,
    lon0: float,
    lat0: float,
    azi: float,
    south_km: float,
    north_km: float,
    depth_min_km: float,
    depth_max_km: float,
    num_along_track: int,
    num_depth: int,
) -> list[str]:
    dist_values_km = linspace_inclusive(-south_km, north_km, num_along_track)
    depth_values_km = linspace_open(depth_min_km, depth_max_km, num_depth)
    lines: list[str] = []

    fio = h5py.File("DATA/rot.h5", "w")

    for iz, depth_km in enumerate(depth_values_km):
        r_m = R_EARTH_M - depth_km * 1000.0
        for ix, dist_km in enumerate(dist_values_km):
            lon, lat = destination_point(lon0, lat0, azi, dist_km * 1000.0)
            x, y, z = lonlatr2xyz(lon, lat, r_m)
            lines.append(
                format_station_line(
                    station_name(ix, iz),
                    network,
                    y,
                    x,
                    z,
                )
            )
            nu = get_rot_matrix(lon, lat)
            name = network + '.' + station_name(ix, iz)
            grp = fio.create_group(name)
            grp.create_dataset("nu", data=nu)

    fio.close()
    return lines

def get_rot_matrix(lon, lat):
    """Station orientation matrix nu at (lon, lat) in degrees.

    nu[orientation, :] is the unit vector, expressed in global ECEF
    (x, y, z) components, of the local orientation 0=North, 1=East,
    2=Vertical (up). lon/lat may be scalars or numpy arrays; the
    returned array has shape (3, 3) + lon.shape.
    """
    phi = np.radians(lon)
    theta = np.radians(90.0 - lat)  # colatitude

    sint, cost = np.sin(theta), np.cos(theta)
    sinp, cosp = np.sin(phi), np.cos(phi)

    nu = np.zeros((3, 3) + np.shape(lon), dtype=np.float64)

    for iorientation, (stazi, stdip) in enumerate(
        ((0.0, 0.0), (90.0, 0.0), (0.0, -90.0))  # North, East, Vertical
    ):
        thetan = np.radians(90.0 + stdip)
        phin = np.radians(stazi)

        n1 = np.cos(thetan)
        n2 = -np.sin(thetan) * np.cos(phin)
        n3 = np.sin(thetan) * np.sin(phin)

        nu[iorientation, 0] = n1 * sint * cosp + n2 * cost * cosp - n3 * sinp
        nu[iorientation, 1] = n1 * sint * sinp + n2 * cost * sinp + n3 * cosp
        nu[iorientation, 2] = n1 * cost - n2 * sint

    return nu


def main() -> None:
    args = parse_args()
    lon0, lat0, azi, _ = read_config_env(args.config_file)
    network = args.network or infer_network(args.input)
    lines = generate_station_lines(
        network,
        lon0,
        lat0,
        azi,
        args.south_km,
        args.north_km,
        args.depth_min_km,
        args.depth_max_km,
        args.num_along_track,
        args.num_depth,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)

    print(
        f"Wrote {len(lines)} stations to {args.output} "
        f"with network '{network}', along wind azimuth {azi} deg from ({lon0}, {lat0})."
    )


if __name__ == "__main__":
    main()
