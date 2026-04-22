import numpy as np
from scipy.special import expit


def holland_radius_km(x_km, y_km, t_s, v_ms=10.0):
    """Compute radius r (km) from moving center along +x."""
    v_km_s = v_ms / 1000.0
    return np.hypot(x_km - v_km_s * t_s, y_km)


def micro_seis_hpa(
    x_km,
    y_km,
    t_s,
    k = 0.3,
    pmin_hpa = 900,
    pn_hpa = 1000,
    rmax_km=5.0,
    b=1.5,
    v_ms=-10.0,
    r_min_km=1e-6,
):
    """Compute pressure P(t,x,y) (hPa) using the Holland model."""
    r_km = holland_radius_km(x_km, y_km, t_s, v_ms=v_ms)

    # time variation
    t0 = 3600 * 5 # 5 hours
    pc = pmin_hpa + (pn_hpa - pmin_hpa) * expit(k * (t_s - t0))

    # spatial variation
    #expo = pc + np.abs(pc - pn_hpa) * np.exp(-abs(rmax_km  - r_km) / (rmax_km * 0.5))
    expo = pc + np.abs(pc - pn_hpa) * np.exp(-(rmax_km / np.maximum(r_km, r_min_km)) ** b)
    return expo - pn_hpa


def micro_seis_pa(x_m, y_m, t_s,*args, **kwargs):
    """Compute micro-seismic pressure in Pa from meters input."""
    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    return micro_seis_hpa(x_km, y_km, t_s, *args, **kwargs) * 100.0


def micro_seis_ddchi(x_m, y_m, t_s, *args, **kwargs):
    """Compute ddchi in SI units (Pa) from meters input."""
    return -micro_seis_pa(x_m, y_m, t_s, *args, **kwargs)
