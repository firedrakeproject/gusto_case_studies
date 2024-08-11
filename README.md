# Gusto Case Studies

Welcome to the Gusto Case Studies repository!

This stores a collection of Python scripts which implement standard (and more exotic) test cases for geophysical fluids, particularly those used in the development of the dynamical cores used in numerical weather prediction models.

The test cases are run using the [Gusto](https://www.firedrakeproject.org/gusto/) code library, which uses compatible finite element methods to discretise the fluid's equations of motion.
Gusto is built upon the [Firedrake](https://www.firedrakeproject.org/) software, which provides the finite element infrastructure used by Gusto.

Case studies are organised by governing equation. Each case study is accompanied by a plotting script, and reference figures.

Our continuous integration is intended to ensure that this repository runs at the Gusto head-of-trunk.

## Visualisation

Gusto can produce output in two formats:
- VTU files, which can be viewed with the [Paraview](https://www.paraview.org/) software
- netCDF files, which has data that can be plotted using standard python packages such as matplotlib. We suggest using the [tomplot](https://github.com/tommbendall/tomplot) python library, which contains several routines to simplify the plotting of Gusto output.
