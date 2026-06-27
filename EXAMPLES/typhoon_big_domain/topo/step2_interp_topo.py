import numpy as np 
import os 
import subprocess
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# set parameters  begin

# module load 
module_cmd = "source ../config.env"

# cube2sph region
xmin=-780000
xmax=780000
ymin=-620000
ymax=650000

# cube2sph transformation params, from step0_check_your_region.sh
cube2sph_param="-91 29.5 0"

# out file
outfile = "interface_top.dat"

###### STOP HERE !!!!!! #######################

# functions 

def find_loc(x0,x):
    return np.argmin(np.abs(x0 - x))

def find_taperloc(x0,x1,x):
    ix1 = np.argmin(abs(x0 - x))
    if x0 > x[ix1]:
        ix1 = ix1 + 1
    ix2 = ix1 + 4
    ix4 = np.argmin(abs(x1 - x))
    if x1 < x[ix4]:
        ix4 = ix4 - 1
    ix3 = ix4 - 4

    return ix1,ix2,ix3,ix4

def costaper(n):
    return  0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, n)))

def add_taper(n,ix1,ix2,ix3,ix4):
    taper = np.zeros((n))
    taper[ix1:ix2+1] = costaper(ix2 - ix1 + 1)
    taper[ix3:ix4 + 1] = costaper(ix4-ix3 + 1)[::-1]
    taper[ix2+1:ix3] = 1.

    return taper 

def main():

    # get point cloud in top of cube 
    nx=101
    ny=101
    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)

    # write file
    fio = open("topo.in","w")
    for iy in range(ny):
        for ix in range(nx):
            fio.write("%f %f %f\n" % (x[ix],y[iy],0.))
    fio.close()

    # cube2latlon
    print("cube2sph transformation ... ")
    cmd = module_cmd + " && " + f"python $CUBE2SPH_PATH/cube2lonlat.py topo.in {cube2sph_param} topo.out  "
    subprocess.run(cmd, shell=True, executable="/bin/bash")

    # interpolate
    print("interpolating topography by using external_topo.txt ...")

    # load coordinates
    cords = np.loadtxt("topo.out")[:,:2] # lon lat
    cords[cords[:,0] > 180,0] -= 360. # convert to -180~180

    # load external_topo
    topo = np.loadtxt("external_topo.txt")
    topo_intp = griddata(topo[:,:2],topo[:,2],cords)
    idx = np.isnan(topo_intp)
    if np.sum(idx) > 0:
        topo_intp[idx] = griddata(topo[:,:2],topo[:,2],cords[idx,:],method="nearest")


    # taper topography
    fio = open(outfile,"w")
    for iy in range(ny):
        for ix in range(nx):
            idx = iy * nx + ix
            z = topo_intp[idx]
            fio.write("%g\n" %(z))
    fio.close()

    print("paste the info below to interfaces.dat, and move the interface_top.dat to the meshfem3D_files")
    print(f"\n\n.true. %d %d %f %f %f %f\n {outfile}" %(nx,ny,x[0],y[0],x[1]-x[0],y[1]-y[0]))


    # clean files
    os.remove("topo.in")
    os.remove("topo.out")


if __name__ == "__main__":
    main()

