"""
This case study demonstrates how to initialise a Gusto shallow water
simulation from ERA5 geopotential and velocity data.

The example here uses the icosahedral sphere mesh and degree 1 spaces.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, pi, Function, Constant,
    FunctionSpace, VectorFunctionSpace, assemble, interpolate, dx
)
from gusto import (
    OutputParameters, ShallowWaterParameters, ShallowWaterEquations,
    lonlatr_from_xyz,
    ZonalComponent, MeridionalComponent, xyz_vector_from_lonlatr,
    GeneralIcosahedralSphereMesh, SIQNModel
)
import numpy as np
from scipy.interpolate import RegularGridInterpolator

era5_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 24,            # every 6hrs
    'dirname': 'era5'
}


def era5(
        ncells_per_edge=era5_defaults['ncells_per_edge'],
        dt=era5_defaults['dt'],
        tmax=era5_defaults['tmax'],
        dumpfreq=era5_defaults['dumpfreq'],
        dirname=era5_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.                  # planetary radius (m)
    mean_depth = 5960.                 # reference depth (m)

    # ------------------------------------------------------------------------ #
    # Set up model
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)

    # Parameters and equation
    parameters = ShallowWaterParameters(mesh, H=mean_depth)
    eqns = ShallowWaterEquations

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True
    )
    diagnostic_fields = [MeridionalComponent("u"), ZonalComponent("u")]

    # Model
    model = SIQNModel(mesh, dt, parameters, eqns, family='BDM')
    model.setup(output, diagnostic_fields=diagnostic_fields)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    stepper = model.stepper
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")

    # load in the lat and lon values of ERA5 data and convert to radians
    lats = np.load("utilities/ERA5_lats_array.npy")
    lons = np.load("utilities/ERA5_lons_array.npy")
    lats *= 2*pi/360
    lons *= 2*pi/360
    data_locs = [lats, lons]
    dlon = lons[1]-lons[0]

    def interpERA5(X, field):
        """
        Given coordinates X (i.e. the DoF locations on the
        unstructured grid), and a field defined at data_locs (i.e. the
        ERA5 data), return an ordered list of field values
        interpolated to the coordinates X.

        Note that X is assumed to be (x, y, z) coordinates but the
        data is defined on (lat, lon) coordinates.
        """

        # set up a python interpolator object
        interp = RegularGridInterpolator(data_locs, field, method="cubic")
        # convert the X coordinates to (lon, lat) coordinates (r
        # assumed to be constant)
        lon, lat, _ = lonlatr_from_xyz(X.dat.data_ro[:, 0],
                                       X.dat.data_ro[:, 1],
                                       X.dat.data_ro[:, 2])
        lon += pi
        fvals = []
        # loop over (lat, lon) in the same order as X, interpolate the
        # field value and append to the list
        for (ilat, ilon) in zip(lat, lon):
            # deal with periodicity
            if ilon > 2*pi-0.5*dlon:
                ilon = 0.
            fvals.append(interp((ilat, ilon)))

        return fvals

    # use DG2 space for velocity components
    Vu = FunctionSpace(mesh, "DG", 2)
    u = Function(Vu)
    v = Function(Vu)
    # extract coordinates of DoFs for DG2 space
    mesh_u = Vu.mesh()
    Wu = VectorFunctionSpace(mesh_u, Vu.ufl_element())
    Xu = assemble(interpolate(mesh_u.coordinates, Wu))
    # read in u and v components and interpolate to DG2 DoF locations
    udat = np.load("utilities/ERA5_u_wind.npy")
    u.dat.data[:] = interpERA5(Xu, udat)
    vdat = np.load("utilities/ERA5_v_wind.npy")
    v.dat.data[:] = interpERA5(Xu, vdat)
    # interpolate to the HDiv space
    xyz = SpatialCoordinate(mesh)
    u0.interpolate(xyz_vector_from_lonlatr(u, v, Constant(0), xyz))

    # read in geopotential data and interpolate to initial depth
    geopot = np.load("utilities/ERA5_geopot_array.npy")
    VD = D0.function_space()
    mesh_D = VD.mesh()
    WD = VectorFunctionSpace(mesh_D, VD.ufl_element())
    XD = assemble(interpolate(mesh_D.coordinates, WD))
    D0.dat.data[:] = interpERA5(XD, geopot)

    # convert geopotential to depth
    D0 /= parameters.g

    # Adjust mean value of initial D
    C = Function(D0.function_space()).assign(Constant(1.0))
    area = assemble(C*dx)
    Dmean = assemble(D0*dx)/area
    D0 -= Dmean
    D0 += Constant(mean_depth)

    Dbar = Function(D0.function_space()).assign(mean_depth)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)

# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ncells_per_edge',
        help="The number of cells per edge of icosahedron",
        type=int,
        default=era5_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=era5_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=era5_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=era5_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=era5_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    era5(**vars(args))
