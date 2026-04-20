---
name: specfem-model
description: "Use when setting up or documenting a 3-D SPECFEM model from DATA/meshfem3D_files/Mesh_Par_file, including mesh preparation, parameter checks, run sequencing, and output validation."
---

# Goal

Set up, review, or document a 3-D SPECFEM model using the parameters defined in `DATA/meshfem3D_files/Mesh_Par_file` and the supporting files in `DATA/`.

# Use When

- Creating a new SPECFEM model configuration.
- Porting an existing case to a smaller or larger domain.
- Verifying whether `Mesh_Par_file` and related inputs are internally consistent.
- Explaining the end-to-end workflow for mesh generation, database creation, solver execution, and post-processing.

# Inputs

- `DATA/meshfem3D_files/Mesh_Par_file`
- Material and boundary-condition files under `DATA/`
- Source and station files, if applicable
- Runtime configuration such as `config.env`
- Optional helper scripts such as `step1_create_mesh.sh` through `step2_slurm_database.sh`

# Preconditions

- The model directory contains a valid `DATA/` tree.
- The mesh parameter file exists and is readable.
- The workflow owner knows the target resolution, domain size, time stepping, and output cadence.
- Required executables or cluster scripts are available before attempting a full run.

# Workflow Template

1. Inspect `Mesh_Par_file` and summarize the domain, element spacing , processor layout, and boundary settings.
2. Confirm that material definitions, topography, and any external forcing inputs match the mesh dimensions and coordinate system.
3. Check consistency between the mesh setup and runtime configuration, including time step, number of steps, and requested outputs.
4. Prepare or review the execution sequence:
	- mesh generation
	- database generation
	- solver run
	- post-processing or movie generation
5. Identify model-specific risks such as unstable time stepping, mismatched processor counts, missing files, or output volume that is too large.
6. Produce a concise action summary with:
	- what is already valid
	- what must be edited
	- what commands or scripts should run next

# Validation Checklist

- `Mesh_Par_file` path and filename are correct.
- Mesh dimensions and element counts are physically reasonable.
- MPI decomposition matches the intended run environment.
- Input files referenced by the workflow actually exist.
- Output directories are present or will be created by the workflow.
- Post-processing scripts expect the same file names and iteration cadence produced by the solver.

# Expected Output

Return a short, operator-focused summary with these sections:

## Model Summary

- Domain and resolution
- Key runtime settings
- Important dependencies

## Issues Found

- Missing files
- Inconsistent parameters
- Run-order or environment problems

## Recommended Next Steps

1. Minimal edits required before execution
2. Commands or scripts to run in order
3. Optional checks after the run completes

# Notes

- Prefer small, explicit parameter changes over broad rewrites.
- Preserve the existing run-script order unless there is a clear workflow error.
- If the model is a reduced copy of a larger case, verify that every copied script still points to the local directory structure.