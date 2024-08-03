"""
The <test_name> test of <authors>, <year>:
``<paper_title>'', <journal>.

<Description of test>

The setup here uses <details of our configuration>.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    SpatialCoordinate, Constant, pi, cos, exp
)
from gusto import (
    Domain, IO, OutputParameters,
    GeneralCubedSphereMesh, ZonalComponent, MeridionalComponent,
    lonlatr_from_xyz, xyz_vector_from_lonlatr, great_arc_angle,
    AdvectionEquation, PrescribedTransport,
    SSPRK3, DGUpwind
)

# Dictionary containing default values of arguments to case study
""" TO REMOVE
Should include resolution, dt, tmax, dumpfreq and dirname
May also include other important configuration options
"""
test_name_defaults = {
    'ncells_1d': 8,
    'dt': 1.0,
    'tmax': 10.0,
    'dumpfreq': 10,
    'dirname': 'test_name'
}


def test_name(
        ncells_1d=test_name_defaults['ncells_1d'],
        dt=test_name_defaults['dt'],
        tmax=test_name_defaults['tmax'],
        dumpfreq=test_name_defaults['dumpfreq'],
        dirname=test_name_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    """ TO REMOVE
    These should contain inline comments to make the parameters clear
    """
    tau = 200.         # time period relating to reversible flow, in s
    radius = 6371220.  # radius of sphere, in m
    lamda_c = 0.       # central longtiude of transported blob, in rad
    theta_c = -pi/6.   # central latitude of transported blob, in rad
    F0 = 3.            # max magnitude of transported blob, in kg/kg
    r0 = 0.25          # width parameter of transported blob, dimensionless

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1
    hdiv_family = 'BDM'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralCubedSphereMesh(radius, ncells_1d, degree=2)
    xyz = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, hdiv_family, degree)

    # Equation
    V = domain.spaces("DG")
    eqn = AdvectionEquation(domain, V, "F")

    # I/O and diagnostics
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
    )

    diagnostic_fields = [ZonalComponent('u'), MeridionalComponent('u')]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Details of transport
    transport_scheme = SSPRK3(domain)
    transport_method = DGUpwind(eqn, "F")

    # Transporting wind ------------------------------------------------------ #
    """ TO REMOVE
    Any prescribed wind for transport tests should be added here
    """
    def u_t(t):
        _, theta, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
        umax = 2*pi*radius / tau
        u_zonal = umax*cos(theta)

        return xyz_vector_from_lonlatr(u_zonal, Constant(0.0), Constant(0.0), xyz)

    # Physics parametrisation ------------------------------------------------ #
    """ TO REMOVE
    Any physics parametrisations need defining here
    """

    # Time stepper
    stepper = PrescribedTransport(
        eqn, transport_scheme, io, transport_method,
        prescribed_transporting_velocity=u_t
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Initialise the field to be transported
    lamda, theta, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    dist = great_arc_angle(lamda, theta, lamda_c, theta_c)
    F_init_expr = F0*exp(-(dist/r0)**2)

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
        '--ncells_1d',
        help="The number of cells in one dimension",
        type=int,
        default=test_name_defaults['ncells_1d']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=test_name_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=test_name_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=test_name_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=test_name_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    test_name(**vars(args))
