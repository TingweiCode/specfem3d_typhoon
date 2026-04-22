from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import PillowWriter

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / "OUTPUT_FILES" / "all_seismograms_main.bin"
    default_stations = base_dir / "DATA" / "STATIONS_FILTERED"

    parser = argparse.ArgumentParser(
        description=(
            "Read a packed SPECFEM all_seismograms binary file and create "
            "a 2-D x-z contour GIF sampled every N time steps."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Packed seismogram file. If missing, the script tries known *_main.bin variants.",
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
        help="Cartesian component to plot from BXX/BXY/BXZ records.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Frame stride in time steps.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=21,
        help="Number of contour levels.",
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
        default=266,
        help="Number of z samples in the interpolated plotting grid.",
    )
    parser.add_argument(
        "--color-min",
        type=float,
        default=None,
        help="Fixed lower bound for the shared colorbar.",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=None,
        help="Fixed upper bound for the shared colorbar.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="GIF output DPI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to OUTPUT_FILES/<input_stem>_<component>_2d.gif.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=30,
        help="Stop the GIF at this simulation time (s). Defaults to the full record.",
    )
    return parser.parse_args()


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
    return input_path.parent / f"{input_path.stem}_{component.lower()}_2d.gif"


def load_station_grid(stations_path: Path) -> tuple[dict[tuple[str, str], tuple[float, float]], np.ndarray, np.ndarray]:
    stations = np.loadtxt(stations_path, dtype=str, ndmin=2)
    station_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    x_values = []
    z_values = []

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

    print(z_coords)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            "The selected station set does not form a complete rectangular grid for this component"
        )

    return x_coords, z_coords, time_axis, cube


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


def save_gif(
    plot_x_coords: np.ndarray,
    plot_z_coords: np.ndarray,
    time_axis: np.ndarray,
    frames: np.ndarray,
    frame_indices: np.ndarray,
    levels: np.ndarray,
    output_path: Path,
    fps: int,
    dpi: int,
    quantity_label: str,
    component: str,
) -> None:
    x_km = plot_x_coords / 1000.0
    z_km = plot_z_coords / 1000.0
    x_grid, z_grid = np.meshgrid(x_km, z_km)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("seismic")
    norm = colors.Normalize(vmin=float(levels[0])*0.1, vmax=float(levels[-1])*0.1)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label(f"{quantity_label} component {component}")

    image = ax.imshow(
        frames[0, :, :],
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=(float(x_km[0]), float(x_km[-1]), float(z_km[0]), float(z_km[-1])),
        aspect="auto",
        interpolation="bilinear",
    )

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_path), dpi=dpi):
        for iframe, it in enumerate(frame_indices):
            field = frames[iframe, :, :]

            image.set_data(field)

            ax.set_title(
                f"{quantity_label} component {component} at step {it} (t = {time_axis[it]:.3f} s)"
            )
            ax.set_xlabel("x (km)")
            ax.set_ylabel("z (km)")
            ax.set_xlim(float(x_km[0]), float(x_km[-1]))
            ax.set_ylim(float(z_km[0]), float(z_km[-1]))
            xmid = (x_km[0] + x_km[-1]) / 2
            #ax.axhline(y=-5, color="cyan", linewidth=1.0, linestyle="--", label="sea level")
            ax.plot([xmid,xmid,x_km[-1]], [0, -5,-5], color="cyan", linewidth=1.0, linestyle="--", label="sea level")
            #ax.plot([(x_km[0] + x_km[-1])/2, (x_km[0] + x_km[-1])/2], [-5, 0], color="cyan", linewidth=1.0, linestyle="-", label="sea level")
            writer.grab_frame()

    plt.close(fig)


def quantity_label_from_filename(path: Path) -> str:
    stem = path.stem
    if "_v_" in stem or stem.endswith("_v_main"):
        return "Velocity"
    if "_d_" in stem or stem.endswith("_d_main"):
        return "Displacement"
    if "_a_" in stem or stem.endswith("_a_main"):
        return "Acceleration"
    return "Seismogram"


def main() -> None:
    args = parse_args()
    input_path = resolve_input_file(args.input)
    output_path = infer_output_path(input_path, args.component, args.output)

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
        args.levels,
        color_min=args.color_min,
        color_max=args.color_max,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(
        plot_x_coords=plot_x_coords,
        plot_z_coords=plot_z_coords,
        time_axis=time_axis,
        frames=plot_frames,
        frame_indices=frame_indices,
        levels=levels,
        output_path=output_path,
        fps=args.fps,
        dpi=args.dpi,
        quantity_label=quantity_label_from_filename(input_path),
        component=args.component,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()