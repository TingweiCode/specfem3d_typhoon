import os
import re
import struct

import numpy as np
from scipy import integrate

from holland_model import micro_seis_ddchi

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - optional dependency
    MPI = None


def read_par_file(par_path):
    nstep = None
    dt = None
    with open(par_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("NSTEP"):
                nstep = int(line.split("=")[1].strip())
            elif line.startswith("DT"):
                dt = float(line.split("=")[1].strip())
    if nstep is None or dt is None:
        raise ValueError("Failed to read NSTEP or DT from Par_file")
    
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Read from Par_file: NSTEP={nstep}, DT={dt}")
    return nstep, dt


def read_config_env(config_path):
    values = {}
    pattern = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    with open(config_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if match:
                key, value = match.groups()
                value = value.split("#", 1)[0]
                values[key] = value.strip().strip('"').strip("'")

    required = ("WIND_LON", "WIND_LAT", "WIND_AZI")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing {missing} in {config_path}")

    wind_lon = float(values["WIND_LON"])
    wind_lat = float(values["WIND_LAT"])
    wind_azi = float(values["WIND_AZI"])
    wind_speed = float(values.get("WIND_SPEED_IN_MPS", 10.0))

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        print(
            f"Read from config.env: WIND_LON={wind_lon}, WIND_LAT={wind_lat}, "
            f"WIND_AZI={wind_azi}, WIND_SPEED_IN_MPS={wind_speed}"
        )
    return wind_lon, wind_lat, wind_azi, wind_speed


def find_proc_files(databases_dir):
    pattern = re.compile(r"proc\d+_free_surface_nodes\.txt$")
    files = []
    for name in os.listdir(databases_dir):
        if pattern.match(name):
            files.append(os.path.join(databases_dir, name))
    return sorted(files)


def xyz2lonlatr(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (lon, lat, r).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z / r))
    return lon, lat, r

def lonlatr2xyz(lon, lat, r):
    """
    Convert spherical coordinates (lon, lat, r) to Cartesian coordinates (x, y, z).
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return x, y, z

def read_nodes_lonlat(node_path):
    if os.path.getsize(node_path) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    data = np.loadtxt(node_path, usecols=(2, 3, 4), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    lon, lat, _ = xyz2lonlatr(x, y, z)
    return lon, lat


def write_fortran_record(handle, array):
    payload = array.tobytes(order="C")
    size = len(payload)
    handle.write(struct.pack("<I", size))
    handle.write(payload)
    handle.write(struct.pack("<I", size))


def process_proc_file(node_path, output_path, nstep, dt, lon0, lat0, azi, v_ms):
    lon, lat = read_nodes_lonlat(node_path)
    nnode = lon.size

    chi_prev = np.zeros(nnode, dtype=np.float64)
    dchi_prev = np.zeros(nnode, dtype=np.float64)
    ddchi_prev = None

    with open(output_path, "wb") as handle:
        t_prev = None
        for it in range(nstep):
            t_s = it * dt
            ddchi = micro_seis_ddchi(lon, lat, t_s, lon0, lat0, azi, v_ms=v_ms).astype(np.float64)

            if it == 0:
                dchi = dchi_prev
                chi = chi_prev
            else:
                inc_dchi = integrate.trapezoid(
                    np.stack([ddchi_prev, ddchi], axis=0),
                    x=[t_prev, t_s],
                    axis=0,
                )
                dchi = dchi_prev + inc_dchi
                inc_chi = integrate.trapezoid(
                    np.stack([dchi_prev, dchi], axis=0),
                    x=[t_prev, t_s],
                    axis=0,
                )
                chi = chi_prev + inc_chi

            write_fortran_record(handle, chi.astype(np.float32))
            write_fortran_record(handle, dchi.astype(np.float32))
            write_fortran_record(handle, ddchi.astype(np.float32))

            chi_prev = chi
            dchi_prev = dchi
            ddchi_prev = ddchi
            t_prev = t_s


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    par_path = os.path.join(base_dir, "DATA", "Par_file")
    config_path = os.path.join(base_dir, "config.env")
    databases_dir = os.path.join(base_dir, "DATABASES_MPI")

    lon0, lat0, azi, v_ms = read_config_env(config_path)
    nstep, dt = read_par_file(par_path)
    proc_files = find_proc_files(databases_dir)
    if not proc_files:
        raise FileNotFoundError("No proc*_free_surface_nodes.txt files found")

    if MPI is None:
        rank = 0
        size = 1
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    for idx, node_path in enumerate(proc_files):
        if idx % size != rank:
            continue
        name = os.path.basename(node_path)
        out_name = name.replace("_free_surface_nodes.txt", "_free_surface_fields.bin")
        out_path = os.path.join(databases_dir, out_name)
        process_proc_file(node_path, out_path, nstep, dt, lon0, lat0, azi, v_ms)


if __name__ == "__main__":
    main()
