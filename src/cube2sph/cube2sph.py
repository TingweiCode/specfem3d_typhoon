import numpy as np

def cube2sph_trans(xi,eta,zeta,cen_lon,cen_lat,rot_azi,R=6371000):
    """
    Convert cube coordinates to spherical coordinates.

    Parameters:
    xi, eta, zeta : float | array_like
        Cube coordinates.
    cen_lon, cen_lat : float
        Center longitude and latitude of the cube face in degrees.
    rot_azi : float
        Rotation azimuth of the cube face in degrees.
    R : float
        Radius of the sphere (default is Earth's radius in meters).

    Returns:
    x,y,z: float | array_like
        x/y/z in greenwich-based cartesian coordinates
    """
    # get rotation matrix
    alpha = np.deg2rad(cen_lon)
    beta = np.pi / 2.0 - np.deg2rad(cen_lat)
    gamma = np.deg2rad(rot_azi)
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    cosb = np.cos(beta)
    sinb = np.sin(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    M11 = cosg * cosb * cosa - sing * sina
    M12 = - sing* cosb * cosa - cosg * sina
    M13 = sinb * cosa
    M21 = cosg * cosb * sina + sing * cosa
    M22 = - sing * cosb * sina + cosg * cosa
    M23 = sinb * sina
    M31 = - cosg * sinb
    M32 = sing * sinb
    M33 = cosb

    # Convert cube coordinates to spherical coordinates
    yp =  eta / R
    xp = xi / R

    z = (R + zeta) / np.sqrt(1 + np.tan(xp)**2 + np.tan(yp)**2)
    x = -z * np.tan(yp)
    y = z * np.tan(xp)

    # Apply rotation matrix
    x_rot = M11 * x + M12 * y + M13 * z
    y_rot = M21 * x + M22 * y + M23 * z
    z_rot = M31 * x + M32 * y + M33 * z


    return x_rot, y_rot, z_rot

def xyz2lonlatr(x, y, z):
    """
    Convert cartesian coordinates to longitude, latitude, and radius.

    Parameters:
    x, y, z : float | array_like
        Cartesian coordinates.

    Returns:
    lon, lat, r : float | array_like
        Longitude and latitude in degrees, and radius.
    """
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    r = np.sqrt(x**2 + y**2 + z**2)
    return lon, lat, r
