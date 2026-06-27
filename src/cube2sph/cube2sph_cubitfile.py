import numpy as np 
from cube2sph import cube2sph_trans, xyz2lonlatr
import sys 

def main():
    # Read command line arguments
    # input is filename cen_lon cen_lat rot_azi
    if len(sys.argv) != 5:
        print("Usage: python cube2lonlat.py filename cen_lon cen_lat rot_azi")
        sys.exit(1)  

    filename = sys.argv[1]
    cen_lon, cen_lat, rot_azi = map(float, sys.argv[2:5])

    # load coordinates
    cords = np.loadtxt(filename,skiprows=1)[:,:]

    xi = cords[:, 1]
    eta = cords[:, 2]
    zeta = cords[:, 3]
    number = cords[:, 0]

    x,y,z = cube2sph_trans(xi, eta, zeta, cen_lon, cen_lat, rot_azi)

    # write out 
    fio = open(filename, "w")
    fio.write(f"{len(x)}\n")
    for i in range(len(x)):
        fio.write(f"{int(number[i])} %lf %lf %lf\n" % (x[i], y[i], z[i]))
    fio.close()

if __name__ == "__main__":
    main()