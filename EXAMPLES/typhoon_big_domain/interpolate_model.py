import numpy as np 
import os 
import glob
from scipy.io import FortranFile
from numba import jit
from scipy.interpolate import interp1d

def xyz2lonlatr(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (lon, lat, r).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z / r))
    return lon, lat, r

@jit(nopython=True)
def dof2elemwise(ibool,pstore):
    """
    Convert DOF-wise data to element-wise data.
    ibool: connectivity array (element to DOF mapping)
    pstore: DOF-wise data
    pelem: output element-wise data
    """
    pelem = np.zeros(len(ibool))
    for iel in range(len(ibool)):
        idx = ibool[iel]
        pelem[iel] = pstore[idx]
    return pelem


def main():

    # load ak135 model
    ak135 = np.loadtxt("veloc_model/ak135.txt") # -depth(km) vp(m/s) vs(m/s) rho(kg/m3)
    ak135[:,0] *= -1000 # to positive depth in km

    # file vp/vs/rho in DATABASES_MPI 
    nproc = len(glob.glob("DATABASES_MPI/*vp.bin"))
    for iproc in range(nproc):
        # read x,y,z 
        fname = f"DATABASES_MPI/proc{iproc:06d}_x.bin"
        fio = FortranFile(fname, 'r')
        xstore = fio.read_reals('f4')
        fio.close()
        fname = f"DATABASES_MPI/proc{iproc:06d}_y.bin"
        fio = FortranFile(fname, 'r')
        ystore = fio.read_reals('f4')
        fio.close()
        fname = f"DATABASES_MPI/proc{iproc:06d}_z.bin"
        fio = FortranFile(fname, 'r')
        zstore = fio.read_reals('f4')
        fio.close()

        # connectivity
        fname = f"DATABASES_MPI/proc{iproc:06d}_ibool.bin"
        fio = FortranFile(fname, 'r')
        ibool = fio.read_ints('i4')
        fio.close()
        ibool = ibool - 1 # convert to 0-based indexing

        # note x,y,z are in meters, convert to lon,lat,r
        lon_store,lat_store,r_store = xyz2lonlatr(xstore, ystore, zstore)
        depth_store = 6371e3 - r_store # depth in meters
        depth = dof2elemwise(ibool, depth_store)

        # get element_wise depth
        depth = np.zeros(len(ibool))
        for iel in range(len(ibool)):
            idx = ibool[iel]
            depth[iel] = depth_store[idx]

        for i,param in enumerate(["vp","vs","rho"]):
            fname = f"DATABASES_MPI/proc{iproc:06d}_{param}.bin"

            # create interpolation function
            f_interp = interp1d(ak135[:,0], ak135[:,i+1], bounds_error=False) # depth in meters
            data_interp = f_interp(depth)
            idx = np.isnan(data_interp)
            f_interp1 = interp1d(ak135[:,0], ak135[:,i+1], kind='nearest',fill_value="extrapolate") # depth in meters
            data_interp[idx] = f_interp1(depth[idx])

            # print min/max of data_interp
            print("min/max of data_interp for proc", iproc, "param", param, np.min(data_interp), np.max(data_interp), "depth min/max", np.min(depth), np.max(depth))

            # write back to file
            fio = FortranFile(fname, 'w')
            fio.write_record(data_interp.astype('f4'))
            fio.close()

if __name__ == "__main__":
    main()