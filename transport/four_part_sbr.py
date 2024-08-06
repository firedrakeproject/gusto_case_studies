"""
The four-part solid body rotation test of Bendall & Wimmer, 2023:
``Improving the accuracy of discretisations of the vector transport equation on
the lowest-order quadrilateral Raviart-Thomas finite elements'', JCP.

The test transports a vector-valued field around the surface of the sphere, with
the flow composed of a solid body rotation in four parts. The impact of metric
terms is reversed through the different parts of the rotation, so that the final
state is equal to the initial state. This makes the test appropriate for
performing convergence tests of the vector transport equation on the sphere.

The setup here uses an icosahedral sphere with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    exp, cos, sin, SpatialCoordinate, pi, Constant
)
from gusto import (
    Domain, AdvectionEquation, OutputParameters, IO, ZonalComponent,
    MeridionalComponent, lonlatr_from_xyz, SSPRK3, DGUpwind, great_arc_angle,
    PrescribedTransport, GeneralIcosahedralSphereMesh, xyz_vector_from_lonlatr
)

four_part_sbr_defaults = {
    'ncells_per_edge': 8,  # num points per icosahedron edge (ref level 3)
    'dt': 0.25,
    'tmax': 400.0,
    'dumpfreq': 200,       # output 8 times, twice per half-rotation
    'dirname': 'four_part_sbr'
}


def four_part_sbr(
        ncells_per_edge=four_part_sbr_defaults['ncells_per_edge'],
        dt=four_part_sbr_defaults['dt'],
        tmax=four_part_sbr_defaults['tmax'],
        dumpfreq=four_part_sbr_defaults['dumpfreq'],
        dirname=four_part_sbr_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    tau = 200.         # time period relating to solid body rotations, in s
    radius = 6371220.  # radius of sphere, in m
    lamda_c = 0.       # central longtiude of transported vector blob, in rad
    theta_c = -pi/6.   # central latitude of transported vector blob, in rad
    F0 = 3.            # max magnitude of transported vector blob, in m/s
    r0 = 0.25          # width parameter of transported vector blob, dimensionless

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1
    hdiv_family = 'BDM'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    xyz = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, hdiv_family, degree)

    # Equation
    V = domain.spaces("HDiv")
    eqn = AdvectionEquation(domain, V, "F")

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
    )

    diagnostic_fields = [ZonalComponent('F'), MeridionalComponent('F')]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Details of transport
    transport_scheme = SSPRK3(domain)
    transport_method = DGUpwind(eqn, "F")

    # Transporting wind ------------------------------------------------------ #
    lamda, theta, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    def u_t(t):
        u_max = 2*pi*radius/tau
        tc = float(2*t/tau)

        if (tc % 2 < 1):
            u_zonal = u_max*cos(theta)
            u_merid = Constant(0.0)*u_max
        else:
            u_zonal = -u_max*cos(lamda)*sin(theta)
            u_merid = u_max*sin(lamda)*(cos(theta)**2 - sin(theta)**2)

        return xyz_vector_from_lonlatr(u_zonal, u_merid, Constant(0.0), xyz)

    # Time stepper
    stepper = PrescribedTransport(
        eqn, transport_scheme, io, transport_method,
        prescribed_transporting_velocity=u_t
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Initialise the vector field to be transported
    F_init_zonal = Constant(0.0)*lamda
    dist = great_arc_angle(lamda, theta, lamda_c, theta_c)
    F_init_merid = F0*exp(-(dist/r0)**2)

    F_init_expr = xyz_vector_from_lonlatr(
        F_init_zonal, F_init_merid, Constant(0.0), xyz
    )

    # Set fields
    u0 = stepper.fields("u")
    F0 = stepper.fields("F")
    u0.project(u_t(0))
    F0.interpolate(F_init_expr)

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
        help="The number of cells per edge of the icosahedron",
        type=int,
        default=four_part_sbr_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=four_part_sbr_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=four_part_sbr_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=four_part_sbr_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=four_part_sbr_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    four_part_sbr(**vars(args))
