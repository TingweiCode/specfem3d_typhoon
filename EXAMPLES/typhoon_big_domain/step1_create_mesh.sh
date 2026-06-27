#!/bin/bash 
set -e 


source  config.env
source cube2sph.env
NPROC=4


bash clean_files.sh
# cleans output files
rm -rf DATABASES_MPI*
rm -rf MESH*
rm -rf OUTPUT_FILES_*
mkdir -p OUTPUT_FILES
rm -rf OUTPUT_FILES/*
mkdir -p MESH
cp DATA/Par_file.init DATA/Par_file

# create mesh 
mkdir -p DATABASES_MPI
$SEM_PATH/bin/xmeshfem3D

# copy mesh to mesh-default
mv OUTPUT_FILES/ OUTPUT_FILES_initmesh
cp -R MESH/ MESH-default
rm -rf DATABASES_MPI
cp MESH-default/* .

# convert 8 to 27 points
python $CUBE2SPH_PATH/hex8tohex27.py
bash change_names.sh

# remove mesh-default and mesh files
\rm -rf absorbing_* mesh_file* materials_file* nodes_coords_file* free_or_absorbing* nummaterial_velocity_file
\rm -rf MESH 

# change coordinates in MESH-default/nodes_coords_file
python $CUBE2SPH_PATH/cube2sph_cubitfile.py MESH-default/nodes_coords_file ${CUBE2SPH_PARAM[*]}


# create databases
# change "NPROC  =  " in Par_file to "NPROC  =  $NPROC"
change_par NPROC "$NPROC" "DATA/Par_file"
change_par NGNOD  27 "DATA/Par_file"

# decompose mesh
mkdir -p DATABASES_MPI OUTPUT_FILES
$SEM_PATH/bin/xdecompose_mesh $NPROC MESH-default/ DATABASES_MPI/

