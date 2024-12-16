"""
A solid body rotation case from 

This is a 3D test on the sphere, with an initial state that is in unsteady
balance, with a perturbation added to the wind.

This setup uses a cubed-sphere with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    ExtrudedMesh, SpatialCoordinate, cos, sin, pi, sqrt, exp, Constant,
    Function, acos, errornorm, norm, le, ge, conditional, inner, dx,
    NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction
)
from gusto import (
    Domain, GeneralCubedSphereMesh, CompressibleParameters, CompressibleSolver,
    CompressibleEulerEquations, OutputParameters, IO, EmbeddedDGOptions, SSPRK3,
    DGUpwind, logger, SemiImplicitQuasiNewton, lonlatr_from_xyz,
    xyz_vector_from_lonlatr, compressible_hydrostatic_balance
)

solid_body_sphere_defaults = {
    'ncell_per_edge': 16,
    'nlayers': 15,
    'dt': 900.0,               # 15 minutes
    'tmax': 15.*24.*60.*60.,   # 15 days
    'dumpfreq': 48,            # Corresponds to every 12 hours with default opts
    'dirname': 'solid_body_sphere'
}