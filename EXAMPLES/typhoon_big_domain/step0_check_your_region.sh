#!/bin/bash
set -e 

source config.env

# set your study region here
lonmin=-98
lonmax=-84
latmin=24
latmax=35

# cube2sph region, this should be edited in meshfem3D_files/Mesh_Par_file
xmin=-780000
xmax=780000
ymin=-620000
ymax=650000
rot=0.


## do nothing .... #######################

# central 
clon=`echo $lonmin $lonmax | awk '{print 0.5*($1+$2)}'`
clat=`echo $latmin $latmax | awk '{print 0.5*($1+$2)}'`

## generate boundary gmt
# module load intel hdf5 netcdf

info=(`echo "$xmin $xmax $ymin $ymax" `)
python $CUBE2SPH_PATH/cube2sph_boundary_gmt.py ${info[*]} $clon $clat $rot
echo "cube2sph_params =  $clon $clat $rot"
echo "CUBE2SPH_PARAM=($clon $clat $rot)" > cube2sph.env


# create study region file
: > studyregion.txt
echo $lonmin $latmin >> studyregion.txt
echo $lonmin $latmax >> studyregion.txt
echo $lonmax $latmax >> studyregion.txt
echo $lonmax $latmin >> studyregion.txt
echo $lonmin $latmin >> studyregion.txt

# larger region
x0=`echo "$lonmin-3"|bc -l`
x1=`echo "$lonmax+3"|bc -l`
y0=`echo "$latmin-3"|bc -l`
y1=`echo "$latmax+3"|bc -l`
bounds1=-R$x0/$x1/$y0/$y1
proj=-JM12c

gmt begin region jpg 
gmt basemap $bounds1 $proj -Bxaf -Byaf 
#gmt plot boundary_2d.gmt.pml -W0.5p,blue -l"Simu Region + PML" $bounds1 
gmt plot boundary_gmt.txt -W0.5p,black -l"Simu Region" $bounds1 
gmt plot studyregion.txt -W0.5p,red -l"Study Region" $bounds1 
gmt coast -A1000 -W0.5p $bounds1 
#awk '{print $1,$2}' station.lst  |gmt plot -St0.4c -Gblack $bounds1
#awk '{print $4,$3}' ../cascadia-cube2sph/src_rec/STATIONS_P28_globe |gmt plot -St0.5c -Gblack
gmt end 

\rm studyregion.txt boundary_gmt.txt