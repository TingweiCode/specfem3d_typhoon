import numpy as np
from scipy.special import expit

R_EARTH_M = 6371e3  # Earth radius in meters


def great_circle_distance_m(lon1, lat1, lon2, lat2):
    """Haversine great-circle distance (meters) between points in decimal degrees."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R_EARTH_M * c


def destination_point(lon0, lat0, bearing_deg, distance_m):
    """Destination point (lon, lat) reached from (lon0, lat0) travelling distance_m
    along initial bearing bearing_deg (degrees clockwise from north) on a sphere."""
    delta = distance_m / R_EARTH_M
    theta = np.radians(bearing_deg)
    phi1 = np.radians(lat0)
    lambda1 = np.radians(lon0)

    phi2 = np.arcsin(
        np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta)
    )
    lambda2 = lambda1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi1),
        np.cos(delta) - np.sin(phi1) * np.sin(phi2),
    )

    return np.degrees(lambda2), np.degrees(phi2)


def holland_radius_km(lon, lat, t_s, lon0, lat0, azi_deg, v_ms=10.0):
    """Compute radius r (km) from the moving storm center to (lon, lat).

    The storm center starts at (lon0, lat0) and moves along great-circle
    bearing azi_deg (degrees clockwise from north) at speed v_ms (m/s).
    """
    center_lon, center_lat = destination_point(lon0, lat0, azi_deg, v_ms * t_s)
    r_m = great_circle_distance_m(center_lon, center_lat, lon, lat)
    return r_m / 1000.0


def micro_seis_hpa(
    lon,
    lat,
    t_s,
    lon0,
    lat0,
    azi_deg,
    k=0.3,
    pmin_hpa=900,
    pn_hpa=1000,
    rmax_km=5.0,
    b=1.5,
    v_ms=10.0,
    r_min_km=1e-6,
):
    """Compute pressure P(t,lon,lat) (hPa) using the Holland model on a sphere."""
    r_km = holland_radius_km(lon, lat, t_s, lon0, lat0, azi_deg, v_ms=v_ms)

    # time variation
    t0 = 3600 * 5  # 5 hours
    pc = pmin_hpa + (pn_hpa - pmin_hpa) * expit(k * (t_s - t0))

    # spatial variation
    expo = pc + np.abs(pc - pn_hpa) * np.exp(-(rmax_km / np.maximum(r_km, r_min_km)) ** b)
    return expo - pn_hpa


def micro_seis_pa(lon, lat, t_s, lon0, lat0, azi_deg, *args, **kwargs):
    """Compute micro-seismic pressure in Pa from lon/lat input (degrees)."""
    return micro_seis_hpa(lon, lat, t_s, lon0, lat0, azi_deg, *args, **kwargs) * 100.0


def micro_seis_ddchi(lon, lat, t_s, lon0, lat0, azi_deg, *args, **kwargs):
    """Compute ddchi in SI units (Pa) from lon/lat input (degrees)."""
    return -micro_seis_pa(lon, lat, t_s, lon0, lat0, azi_deg, *args, **kwargs)
