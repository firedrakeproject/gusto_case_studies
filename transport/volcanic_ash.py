"""
An example experiment for volcanic ash dispersion.

This is not a published test, but is intended as a demo of transport coupled to
a time-dependent source term (representing the volcano). The wind is chosen to
create an aesthetically pleasing solution.

The setup here uses a plane with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    RectangleMesh, exp, SpatialCoordinate, pi, Constant, sin, cos, sqrt, grad
)
from gusto import (
    Domain, AdvectionEquation, OutputParameters, CourantNumber, XComponent,
    YComponent, DGUpwind, SourceSink, PrescribedTransport, SSPRK3, IO, logger,
    DG1Limiter
)

volcanic_ash_defaults = {
    'ncells_1d': 200,        # defines number of points in x and y directions
    'dt': 120.0,             # 2 minutes
    'tmax': 5.*24.*60.*60.,  # 5 days
    'dumpfreq': 720,         # corresponds to once per day
    'dirname': 'volcanic_ash'
}


def volcanic_ash(
        ncells_1d=volcanic_ash_defaults['ncells_1d'],
        dt=volcanic_ash_defaults['dt'],
        tmax=volcanic_ash_defaults['tmax'],
        dumpfreq=volcanic_ash_defaults['dumpfreq'],
        dirname=volcanic_ash_defaults['dirname']
):

    # ---------------------------------------------------------------------------- #
    # Test case parameters
    # ---------------------------------------------------------------------------- #

    Lx = 1e6                 # Domain length in x direction, in m
    Ly = 1e6                 # Domain length in y direction, in m
    tau = 2.0*24*60*60       # Half life of source, in s
    centre_x = 3 * Lx / 8.0  # x coordinate for volcano, in m
    centre_y = 2 * Ly / 3.0  # y coordinate for volcano, in m
    width = Lx / 50.0        # width of volcano, in m
    umax = 12.0              # Representative wind value, in m/s
    twind = 5*24*60*60       # Time scale for wind components, in s
    omega22 = 3.0 / twind    # Frequency for sin(2*pi*x/Lx)*sin(2*pi*y/Ly), in 1/s
    omega21 = 0.9 / twind    # Frequency for sin(2*pi*x/Lx)*sin(pi*y/Ly), in 1/s
    omega12 = 0.6 / twind    # Frequency for sin(pi*x/Lx)*sin(2*pi*y/Ly), in 1/s
    omega44 = 0.1 / twind    # Frequency for sin(4*pi*x/Lx)*sin(4*pi*y/Ly), in 1/s

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = RectangleMesh(ncells_1d, ncells_1d, Lx, Ly, quadrilateral=True)
    domain = Domain(mesh, dt, "RTCF", degree)
    x, y = SpatialCoordinate(mesh)

    # Equation
    V = domain.spaces('DG')
    eqn = AdvectionEquation(domain, V, "ash")

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False,
        dumplist=['ash']
    )
    diagnostic_fields = [CourantNumber(), XComponent('u'), YComponent('u')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_scheme = SSPRK3(domain, limiter=DG1Limiter(V))
    transport_method = [DGUpwind(eqn, "ash")]

    # Physics scheme --------------------------------------------------------- #
    # Source is a Lorentzian centred on a point
    dist_x = x - centre_x
    dist_y = y - centre_y
    dist = sqrt(dist_x**2 + dist_y**2)
    # Lorentzian function
    basic_expression = -width / (dist**2 + width**2)

    def time_varying_expression(t):
        return 2*basic_expression*exp(-t/tau)

    physics_parametrisations = [
        SourceSink(eqn, 'ash', time_varying_expression, time_varying=True)
    ]

    # Time stepper
    time_varying_velocity = True
    stepper = PrescribedTransport(
        eqn, transport_scheme, io, time_varying_velocity, transport_method,
        physics_parametrisations=physics_parametrisations
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    ash0 = stepper.fields("ash")
    # Start with some ash over the volcano
    ash0.interpolate(Constant(0.0)*-basic_expression)

    # Transporting wind ------------------------------------------------------ #
    def transporting_wind(t):
        # Divergence-free wind. A series of sines/cosines with different time factors
        psi_expr = (0.25*Lx/pi)*umax*(
            sin(pi*x/Lx)*sin(pi*y/Ly)
            + 0.15*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*(1.0 + cos(2*pi*omega22*t))
            + 0.25*sin(2*pi*x/Lx)*sin(pi*y/Ly)*sin(2*pi*omega21*(t-0.7*twind))
            + 0.17*sin(pi*x/Lx)*sin(2*pi*y/Ly)*cos(2*pi*omega12*(t+0.2*twind))
            + 0.12*sin(4*pi*x/Lx)*sin(4*pi*y/Ly)*(1.0 + sin(2*pi*omega44*(t-0.83*twind)))
        )

        return domain.perp(grad(psi_expr))

    stepper.setup_prescribed_expr(transporting_wind)

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    num_steps = int(tmax / dt)
    logger.info(f'Beginning run to do {num_steps} steps')
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
        help="The number of cells in the x and y directions",
        type=int,
        default=volcanic_ash_defaults['ncells_1d']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=volcanic_ash_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=volcanic_ash_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=volcanic_ash_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=volcanic_ash_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    volcanic_ash(**vars(args))
