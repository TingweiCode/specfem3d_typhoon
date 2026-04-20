from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEA_LEVEL_KM = -5  # -5 m expressed in km

# ---------------------------------------------------------------------------
# GMT bash script written once per run; parameters passed via env vars
# ---------------------------------------------------------------------------

_GMT_SCRIPT = r"""#!/usr/bin/env bash
set -e

module add gmt

gmt makecpt -Cpolar -I -T${VMIN}/${VMAX}/100+n -Z  > velocity.cpt

gmt surface "${DATAFILE}" -bi3d -Gvelocity.grd -I${DX}/${DZ} \
    -R${XMIN}/${XMAX}/${ZMIN}/${ZMAX}

gmt begin "${STEM}" png
  gmt basemap -R${XMIN}/${XMAX}/${ZMIN}/${ZMAX} -JX17c/8c \
    -BWSne+t"${TITLE}" -Bxaf+l"x (km)" -Byaf+l"z (km)"
  gmt grdimage velocity.grd -Cvelocity.cpt -E200
  gmt plot -W1p,black << EOF
${XMIN} ${SEA_LEVEL}
${XMAX} ${SEA_LEVEL}
EOF
  gmt colorbar -Cvelocity.cpt -Bxaf+l"${CLABEL}"
  printf "%s\t${SEA_LEVEL}\n%s\t${SEA_LEVEL}\n" "${XMIN}" "${XMAX}" | \
    gmt plot -W1p,cyan,dashed
gmt end
"""


# ---------------------------------------------------------------------------
# Data-reading helpers (identical logic to plot_2d_seismogram_gif.py)
# ---------------------------------------------------------------------------


def load_station_grid(
    stations_path: Path,
) -> tuple[dict[tuple[str, str], tuple[float, float]], np.ndarray, np.ndarray]:
    stations = np.loadtxt(stations_path, dtype=str, ndmin=2)
    station_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    x_values: list[float] = []
    z_values: list[float] = []

    for row in stations:
        station = row[0]
        network = row[1]
        x_value = float(row[3])
        z_value = float(row[5])
        station_lookup[(network, station)] = (x_value, z_value)
        x_values.append(x_value)
        z_values.append(z_value)

    x_coords = np.unique(np.round(np.asarray(x_values, dtype=np.float64), decimals=6))
    z_coords = np.unique(np.round(np.asarray(z_values, dtype=np.float64), decimals=6))
    return station_lookup, np.sort(x_coords), np.sort(z_coords)


def component_from_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) < 3:
        raise ValueError(f"Unexpected record name format: {name}")
    channel = parts[2]
    if not channel:
        raise ValueError(f"Missing channel code in record name: {name}")
    return channel[-1]


def packed_nt(file_size: int, nrecords: int) -> int:
    nt = (file_size // nrecords - 512 - 8) // 16
    expected_size = nrecords * (512 + 8 + nt * 16)
    if expected_size != file_size:
        raise ValueError(
            "Packed seismogram file size does not match the expected FWAT binary layout"
        )
    return int(nt)


def read_component_cube(
    input_path: Path,
    stations_path: Path,
    component: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    station_lookup, x_coords, z_coords = load_station_grid(stations_path)
    nstations = len(station_lookup)
    nrecords = nstations * 3
    nt = packed_nt(input_path.stat().st_size, nrecords)

    x_index = {value: index for index, value in enumerate(x_coords)}
    z_index = {value: index for index, value in enumerate(z_coords)}
    cube = np.full((len(z_coords), len(x_coords), nt), np.nan, dtype=np.float32)
    time_axis = None

    with input_path.open("rb") as handle:
        for _ in range(nrecords):
            _ = np.fromfile(handle, dtype=np.int32, count=1)[0]
            raw_name = np.fromfile(handle, dtype="S512", count=1)[0]
            _ = np.fromfile(handle, dtype=np.int32, count=1)[0]
            raw_block = np.fromfile(handle, dtype=np.float32, count=nt * 4).reshape(nt, 4)

            name = raw_name.decode("utf-8", errors="ignore").rstrip("\x00").rstrip()
            if component_from_name(name) != component:
                continue

            parts = name.split(".")
            network = parts[0]
            station = parts[1]
            if (network, station) not in station_lookup:
                continue

            x_value, z_value = station_lookup[(network, station)]
            x_key = round(x_value, 6)
            z_key = round(z_value, 6)
            cube[z_index[z_key], x_index[x_key], :] = raw_block[:, 2]
            if time_axis is None:
                time_axis = raw_block[:, 1].copy()

    if time_axis is None:
        raise ValueError(f"No component {component} traces were found in {input_path}")
    if np.isnan(cube).any():
        raise ValueError(
            "The selected station set does not form a complete rectangular "
            "grid for this component"
        )

    return x_coords, z_coords, time_axis, cube


def build_frame_indices(nt: int, sample_every: int) -> np.ndarray:
    indices = np.arange(0, nt, max(sample_every, 1), dtype=int)
    if indices[-1] != nt - 1:
        indices = np.append(indices, nt - 1)
    return indices


def interpolate_frame(
    frame: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    target_x: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    interp_x = np.empty((frame.shape[0], len(target_x)), dtype=np.float32)
    for iz in range(frame.shape[0]):
        interp_x[iz, :] = np.interp(target_x, x_coords, frame[iz, :])

    interp_z = np.empty((len(target_z), len(target_x)), dtype=np.float32)
    for ix in range(len(target_x)):
        interp_z[:, ix] = np.interp(target_z, z_coords, interp_x[:, ix])

    return interp_z


def interpolate_frames(
    frames: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    plot_nx: int,
    plot_nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plot_nx < 2 or plot_nz < 2:
        raise ValueError("--plot-nx and --plot-nz must both be at least 2")

    target_x = np.linspace(float(x_coords[0]), float(x_coords[-1]), plot_nx)
    target_z = np.linspace(float(z_coords[0]), float(z_coords[-1]), plot_nz)
    interpolated = np.empty((frames.shape[0], plot_nz, plot_nx), dtype=np.float32)

    for iframe in range(frames.shape[0]):
        interpolated[iframe, :, :] = interpolate_frame(
            frames[iframe, :, :],
            x_coords,
            z_coords,
            target_x,
            target_z,
        )

    return target_x, target_z, interpolated


def contour_levels(
    frames: np.ndarray,
    nlevels: int,
    color_min: float | None = None,
    color_max: float | None = None,
) -> np.ndarray:
    if color_min is not None or color_max is not None:
        if color_min is None or color_max is None:
            raise ValueError("--color-min and --color-max must be provided together")
        if color_min >= color_max:
            raise ValueError("--color-min must be smaller than --color-max")
        return np.linspace(color_min, color_max, nlevels)

    max_abs = float(np.nanmax(np.abs(frames)))
    if max_abs == 0.0:
        max_abs = 1.0
    return np.linspace(-max_abs, max_abs, nlevels)


def resolve_input_file(path: Path) -> Path:
    if path.is_file():
        return path

    parent = path.parent
    preferred = [
        parent / "all_seismograms_v_main.bin",
        parent / "all_seismograms_d_main.bin",
        parent / "all_seismograms_a_main.bin",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    matches = sorted(parent.glob("all_seismograms*_main.bin"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find a packed seismogram file near {path}")


def infer_output_path(input_path: Path, component: str, output: Path | None) -> Path:
    if output is not None:
        return output
    return input_path.parent / f"{input_path.stem}_{component.lower()}_2d_gmt.gif"


def quantity_label_from_filename(path: Path) -> str:
    stem = path.stem
    if "_v_" in stem or stem.endswith("_v_main"):
        return "Velocity"
    if "_d_" in stem or stem.endswith("_d_main"):
        return "Displacement"
    if "_a_" in stem or stem.endswith("_a_main"):
        return "Acceleration"
    return "Seismogram"


# ---------------------------------------------------------------------------
# GMT frame rendering
# ---------------------------------------------------------------------------


def write_gmt_script(work_dir: Path) -> Path:
    script_path = work_dir / "plot_frame.sh"
    script_path.write_text(_GMT_SCRIPT)
    script_path.chmod(0o755)
    return script_path


def plot_frame_gmt(
    x_km: np.ndarray,
    z_km: np.ndarray,
    field: np.ndarray,
    frame_idx: int,
    it: int,
    time_val: float,
    vmin: float,
    vmax: float,
    work_dir: Path,
    script_path: Path,
    quantity_label: str,
    component: str,
) -> Path:
    """Write one binary data file and call the GMT script to produce a PNG."""
    # Flatten the (nz, nx) grid to (N, 3) float64 binary: x  z  value
    xx, zz = np.meshgrid(x_km, z_km)
    xyz = np.column_stack(
        [xx.ravel(), zz.ravel(), field.ravel()]
    ).astype(np.float64)
    data_file = work_dir / "data.bin"
    data_file.write_bytes(xyz.tobytes())

    nx, nz = len(x_km), len(z_km)
    xmin, xmax = float(x_km[0]), float(x_km[-1])
    zmin, zmax = float(z_km[0]), float(z_km[-1])
    dx = (xmax - xmin) / (nx - 1)
    dz = (zmax - zmin) / (nz - 1)

    # Guard against degenerate color range
    if abs(vmax - vmin) < 1e-30:
        vmin, vmax = -1.0, 1.0

    frame_stem = f"frame_{frame_idx:06d}"
    title = f"{quantity_label} {component}  t = {time_val:.3f} s"

    env = {
        **os.environ,
        "VMIN": f"{vmin:.6g}",
        "VMAX": f"{vmax:.6g}",
        "DX": f"{dx:.10g}",
        "DZ": f"{dz:.10g}",
        "XMIN": f"{xmin:.10g}",
        "XMAX": f"{xmax:.10g}",
        "ZMIN": f"{zmin:.10g}",
        "ZMAX": f"{zmax:.10g}",
        "STEM": frame_stem,
        "TITLE": title,
        "CLABEL": f"{quantity_label} component {component}",
        "SEA_LEVEL": f"{SEA_LEVEL_KM:.6g}",
        "DATAFILE": str(data_file),
    }

    subprocess.run(
        ["bash", str(script_path)],
        cwd=str(work_dir),
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return work_dir / f"{frame_stem}.png"


# ---------------------------------------------------------------------------
# GIF assembly
# ---------------------------------------------------------------------------


def make_gif(png_files: list[Path], output_path: Path, fps: int) -> None:
    frames = [Image.open(str(p)).convert("RGBA") for p in png_files]
    delay_ms = max(20, int(1000 / fps))
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=delay_ms,
        loop=0,
        optimize=False,
    )
    for f in frames:
        f.close()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / "OUTPUT_FILES" / "all_seismograms_main.bin"
    default_stations = base_dir / "DATA" / "STATIONS_FILTERED"

    parser = argparse.ArgumentParser(
        description=(
            "Read a packed SPECFEM all_seismograms binary file and create "
            "a 2-D x-z GIF using GMT, sampled every N time steps."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Packed seismogram file.",
    )
    parser.add_argument(
        "--stations",
        type=Path,
        default=default_stations,
        help="Station file used to recover x-z coordinates.",
    )
    parser.add_argument(
        "--component",
        choices=["X", "Y", "Z"],
        default="Z",
        help="Cartesian component to plot.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Frame stride in time steps.",
    )
    parser.add_argument(
        "--plot-nx",
        type=int,
        default=256,
        help="Number of x samples in the interpolated plotting grid.",
    )
    parser.add_argument(
        "--plot-nz",
        type=int,
        default=256,
        help="Number of z samples in the interpolated plotting grid.",
    )
    parser.add_argument(
        "--color-min",
        type=float,
        default=None,
        help="Fixed lower bound for the colorbar.",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=None,
        help="Fixed upper bound for the colorbar.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=30,
        help="Stop the GIF at this simulation time (s). Defaults to the full record.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Copy intermediate PNG frames next to the output GIF.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if shutil.which("gmt") is None:
        raise SystemExit(
            "ERROR: 'gmt' not found in PATH. "
            "Load the GMT module (e.g. 'module add gmt') before running this script."
        )

    input_path = resolve_input_file(args.input)
    output_path = infer_output_path(input_path, args.component, args.output)

    print(f"Reading {input_path} ...")
    x_coords, z_coords, time_axis, cube = read_component_cube(
        input_path=input_path,
        stations_path=args.stations,
        component=args.component,
    )

    nt = cube.shape[-1]
    if args.end_time is not None:
        nt = int(np.searchsorted(time_axis, args.end_time, side="right"))
        nt = max(1, min(nt, cube.shape[-1]))

    frame_indices = build_frame_indices(nt, args.sample_every)
    sampled_frames = np.transpose(cube[:, :, frame_indices], (2, 0, 1))

    plot_x_coords, plot_z_coords, plot_frames = interpolate_frames(
        sampled_frames,
        x_coords,
        z_coords,
        plot_nx=args.plot_nx,
        plot_nz=args.plot_nz,
    )

    levels = contour_levels(
        plot_frames,
        nlevels=21,
        color_min=args.color_min,
        color_max=args.color_max,
    )

    quantity_label = quantity_label_from_filename(input_path)
    x_km = plot_x_coords / 1000.0
    z_km = plot_z_coords / 1000.0
    vmin = float(levels[0])
    vmax = float(levels[-1])
    vmin=-0.002
    vmax=0.002

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="gmt_frames_"))

    try:
        script_path = write_gmt_script(work_dir)
        png_files: list[Path] = []
        nframes = len(frame_indices)

        for iframe, it in enumerate(frame_indices):
            print(f"\r  GMT frame {iframe + 1}/{nframes}", end="", flush=True)
            png = plot_frame_gmt(
                x_km=x_km,
                z_km=z_km,
                field=plot_frames[iframe, :, :],
                frame_idx=iframe,
                it=int(it),
                time_val=float(time_axis[it]),
                vmin=vmin,
                vmax=vmax,
                work_dir=work_dir,
                script_path=script_path,
                quantity_label=quantity_label,
                component=args.component,
            )
            png_files.append(png)

        print()
        print(f"Assembling GIF ({nframes} frames) ...")
        make_gif(png_files, output_path, args.fps)
        print(f"Saved {output_path}")

        if args.keep_frames:
            frames_dir = output_path.parent / (output_path.stem + "_frames")
            frames_dir.mkdir(exist_ok=True)
            for p in png_files:
                shutil.copy(p, frames_dir / p.name)
            print(f"PNG frames saved in {frames_dir}")

    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
