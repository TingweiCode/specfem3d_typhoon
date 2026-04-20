#!/bin/bash

module add gmt 

# data  is save in binaries (x,z,velocity) in double precision 

xmin=
xmax=
zmin=
zmax=
datafile=
bounds="-R$xmin/$xmax/$zmin/$zmax"
proj="-JX10c/5c"

vmin=
vmax=
gmt makecpt -Cpolar -I -T$vmin/$vmax/100+n > velocity.cpt

# dx dz = (xmax-xmin)/256 (zmax-zmin)/256

gmt surface $datafile -bi3d -Gvelocity.grd -I$dx/$dz $bounds

gmt begin $title
gmt basemap $bounds $proj -BWSne 
gmt grdimage velocity.grd -Cvelocity.cpt -E200
gmt colorbar -Cvelocity.cpt -Bxaf:"velocity (m/s)"
gmt end 

