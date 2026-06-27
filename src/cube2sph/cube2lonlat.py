import numpy as np 
from cube2sph import cube2sph_trans, xyz2lonlatr
import sys 

def main():
    # Read command line arguments
    # input is filename cen_lon cen_lat rot_azi outfile
    if len(sys.argv) != 6:
        print("Usage: python cube2lonlat.py filename cen_lon cen_lat rot_azi outfile")
        sys.exit(1)  

    filename = sys.argv[1]
    cen_lon, cen_lat, rot_azi = map(float, sys.argv[2:5])
    outfile = sys.argv[5]

    # load coordinates
    cords = np.loadtxt(filename)

    xi = cords[:, 0]
    eta = cords[:, 1]
    zeta = np.zeros_like(xi)

    x,y,z = cube2sph_trans(xi, eta, zeta, cen_lon, cen_lat, rot_azi)
    lon, lat, r = xyz2lonlatr(x, y, z)

    # write lon/lat to outfile
    fio = open(outfile, "w")
    for i in range(len(lon)):
        fio.write(f"{lon[i]} {lat[i]} {r[i]}\n")
    fio.close()

if __name__ == "__main__":
    main()