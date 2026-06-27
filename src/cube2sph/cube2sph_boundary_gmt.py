import numpy as np 
from cube2sph import cube2sph_trans, xyz2lonlatr
import sys 

def main():
    # Read command line arguments
    # input is xmin xmax ymin ymax cen_lon cen_lat rot_azi
    if len(sys.argv) != 8:
        print("Usage: python cube2sph_boundary_gmt.py xmin xmax ymin ymax cen_lon cen_lat rot_azi")
        sys.exit(1) 

    xmin, xmax, ymin, ymax, cen_lon, cen_lat, rot_azi = map(float, sys.argv[1:])

    # generate grid points along the boundary of the cube face
    num_points = 128  # number of points along each edge
    x = np.linspace(xmin, xmax, num_points)
    y = np.linspace(ymin, ymax, num_points)
    z = np.zeros(num_points)

    # Create boundary points for 4 edges of the cube face
    boundary_points = []
    # Bottom edge
    for xi in x:
        boundary_points.append((xi, ymin, 0))
    # Right edge
    for yi in y:
        boundary_points.append((xmax, yi, 0))
    # Top edge
    for xi in reversed(x):
        boundary_points.append((xi, ymax, 0))
    # Left edge
    for yi in reversed(y):
        boundary_points.append((xmin, yi, 0))

    # Convert boundary points to spherical coordinates
    boundary_points = np.array(boundary_points)
    xi = boundary_points[:, 0]
    eta = boundary_points[:, 1]
    zeta = boundary_points[:, 2]
    x1,y1,z1 = cube2sph_trans(xi, eta, zeta, cen_lon, cen_lat, rot_azi)

    # Convert to lon, lat, r
    lon, lat, r = xyz2lonlatr(x1, y1, z1)

    # Print the results in GMT format
    fio = open("boundary_gmt.txt", "w")
    for i in range(len(lon)):
        fio.write(f"{lon[i]} {lat[i]}\n")
    fio.close()

if __name__ == "__main__":
    main()