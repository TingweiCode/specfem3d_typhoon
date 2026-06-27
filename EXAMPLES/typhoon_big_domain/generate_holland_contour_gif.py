import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from create_free_surface_fields import read_config_env
from holland_model import R_EARTH_M, micro_seis_ddchi


def read_par_file(par_path):
    values = {}
    with open(par_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = [item.strip() for item in line.split("=", 1)]
            values[key] = value

    nstep = int(values["NSTEP"])
    dt = float(values["DT"])
    print(nstep * dt, "seconds total simulation time")
    ntstep_between_frames = int(values.get("NTSTEP_BETWEEN_FRAMES", 1))
    # ntstep_between_frames = 100
    return nstep, dt, ntstep_between_frames


def km_to_lonlat(x_km, y_km, lon_center, lat_center):
    """Convert local east/north offsets (km) from (lon_center, lat_center) to
    (lon, lat) degrees using a flat-Earth tangent-plane approximation."""
    deg_per_km_lat = 180.0 / (np.pi * R_EARTH_M / 1000.0)
    lat = lat_center + y_km * deg_per_km_lat
    deg_per_km_lon = deg_per_km_lat / np.cos(np.radians(lat_center))
    lon = lon_center + x_km * deg_per_km_lon
    return lon, lat


def build_grid(lon0, lat0, width_km, height_km, nx, ny, x_center_km, y_center_km):
    x_km = np.linspace(
        x_center_km - 0.5 * width_km,
        x_center_km + 0.5 * width_km,
        nx,
        dtype=np.float64,
    )
    y_km = np.linspace(
        y_center_km - 0.5 * height_km,
        y_center_km + 0.5 * height_km,
        ny,
        dtype=np.float64,
    )
    x_grid_km, y_grid_km = np.meshgrid(x_km, y_km)
    lon_grid, lat_grid = km_to_lonlat(x_grid_km, y_grid_km, lon0, lat0)
    lon_1d, _ = km_to_lonlat(x_km, 0.0, lon0, lat0)
    _, lat_1d = km_to_lonlat(0.0, y_km, lon0, lat0)
    return lon_grid, lat_grid, lon_1d, lat_1d


def advance_ddchi_frames(nstep, dt, lon, lat, lon0, lat0, azi, v_ms, sample_every):
    sampled_steps = []
    sampled_times = []
    sampled_ddchi = []

    for it in range(nstep):
        t_s = it * dt
        ddchi = micro_seis_ddchi(lon, lat, t_s, lon0, lat0, azi, v_ms=v_ms).astype(np.float64)

        if it % sample_every == 0 or it == nstep - 1:
            sampled_steps.append(it)
            sampled_times.append(t_s)
            sampled_ddchi.append(ddchi.copy())

    return {
        "step": np.asarray(sampled_steps, dtype=np.int32),
        "time_s": np.asarray(sampled_times, dtype=np.float64),
        "ddchi": sampled_ddchi,
    }


def contour_levels(frames, nlevels):
    max_abs = max(float(np.max(np.abs(frame))) for frame in frames)
    if max_abs == 0.0:
        max_abs = 1.0
    return np.linspace(-max_abs, max_abs, nlevels, dtype=np.float64)


def remove_contour_set(contour_set):
    if contour_set is None:
        return

    remove = getattr(contour_set, "remove", None)
    if callable(remove):
        remove()
        return

    for collection in getattr(contour_set, "collections", []):
        collection.remove()


def save_gif(lon, lat, lat_center, frames, times_s, field_name, output_path, fps, levels):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cmap = plt.get_cmap("RdBu_r")
    contourf = None
    contour = None
    colorbar = None

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, output_path, dpi=140):
        for frame, t_s in zip(frames, times_s):
            remove_contour_set(contourf)
            remove_contour_set(contour)

            contourf = ax.contourf(
                lon_grid,
                lat_grid,
                frame,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            contour = ax.contour(
                lon_grid,
                lat_grid,
                frame,
                levels=levels[::2],
                colors="k",
                linewidths=0.35,
                alpha=0.35,
            )

            if colorbar is None:
                colorbar = fig.colorbar(contourf, ax=ax, pad=0.02)
                colorbar.set_label(field_name)

            ax.set_title(f"Holland model {field_name} at t = {t_s:.1f} s")
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
            ax.set_aspect(1.0 / np.cos(np.radians(lat_center)))
            ax.set_xlim(lon[0], lon[-1])
            ax.set_ylim(lat[0], lat[-1])

            writer.grab_frame()

    plt.close(fig)


def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_par = os.path.join(base_dir, "DATA", "Par_file")
    default_config = os.path.join(base_dir, "config.env")
    default_out = os.path.join(base_dir, "OUTPUT_FILES", "holland_ddchi_contours_300km.gif")

    parser = argparse.ArgumentParser(
        description="Generate a ddchi contour GIF for the Holland-model free-surface forcing."
    )
    parser.add_argument(
        "--config-file", default=default_config, help="Path to config.env (WIND_LON/WIND_LAT/WIND_AZI/WIND_SPEED_IN_MPS)"
    )
    parser.add_argument("--par-file", default=default_par, help="Path to DATA/Par_file")
    parser.add_argument("--width-km", type=float, default=60.0, help="Domain width in km")
    parser.add_argument("--height-km", type=float, default=60.0, help="Domain height in km")
    parser.add_argument(
        "--x-center-km", type=float, default=0.0, help="Domain center x offset from storm start, in km"
    )
    parser.add_argument(
        "--y-center-km", type=float, default=0.0, help="Domain center y offset from storm start, in km"
    )
    parser.add_argument("--nx", type=int, default=201, help="Number of x samples")
    parser.add_argument("--ny", type=int, default=201, help="Number of y samples")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="Time-step stride between animation frames; defaults to NTSTEP_BETWEEN_FRAMES",
    )
    parser.add_argument("--levels", type=int, default=21, help="Number of contour levels")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    parser.add_argument("--output", default=default_out, help="Output GIF path")
    return parser.parse_args()


def main():
    args = parse_args()
    lon0, lat0, azi, v_ms = read_config_env(args.config_file)
    nstep, dt, ntstep_between_frames = read_par_file(args.par_file)
    sample_every = args.sample_every or ntstep_between_frames
    sample_every = max(sample_every, 1)

    lon_grid, lat_grid, lon_1d, lat_1d = build_grid(
        lon0=lon0,
        lat0=lat0,
        width_km=args.width_km,
        height_km=args.height_km,
        nx=args.nx,
        ny=args.ny,
        x_center_km=args.x_center_km,
        y_center_km=args.y_center_km,
    )
    samples = advance_ddchi_frames(
        nstep=nstep,
        dt=dt,
        lon=lon_grid,
        lat=lat_grid,
        lon0=lon0,
        lat0=lat0,
        azi=azi,
        v_ms=v_ms,
        sample_every=sample_every,
    )
    frames = samples["ddchi"]
    levels = contour_levels(frames, args.levels)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_gif(
        lon=lon_1d,
        lat=lat_1d,
        lat_center=lat0,
        frames=frames,
        times_s=samples["time_s"],
        field_name="ddchi",
        output_path=args.output,
        fps=args.fps,
        levels=levels,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()