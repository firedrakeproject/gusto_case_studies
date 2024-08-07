"""
The divergence-free transport test of Nair & Lauritzen, 2010:
``A class of deformational flow test cases for linear transport problems
on the sphere'', JCP.

The test involves deformational transport on the surface of a sphere. This
script describes implements Case 4 from the paper (the more deformational case
with background flow), and contains options for different scalar initial
conditions:
- 'gaussian': for a smooth scalar field
- 'cosine_bells': for a quasi-smooth scalar field
- 'slotted_cylinder': for a non-smooth scalar field

The setup here uses an icosahedral sphere with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    exp, cos, sin, conditional, SpatialCoordinate, pi, min_value, grad
)
from gusto import (
    Domain, AdvectionEquation, OutputParameters, IO, lonlatr_from_xyz, SSPRK3,
    DGUpwind, PrescribedTransport, GeneralIcosahedralSphereMesh,
    great_arc_angle
)

nair_lauritzen_non_divergent_defaults = {
    'initial_conditions': 'slotted_cylinder',  # one of 'slotted_cylinder',
                                               # 'cosine_bells' or 'gaussian'
    'ncells_per_edge': 8,     # num points per icosahedron edge (ref level 3)
    'dt': 900.0,              # 15 minutes
    'tmax': 12.*24.*60.*60.,  # 12 days
    'dumpfreq': 288,          # once every 3 days with default values
    'dirname': 'nair_lauritzen_non_divergent'
}


def nair_lauritzen_non_divergent(
        initial_conditions=nair_lauritzen_non_divergent_defaults['initial_conditions'],
        ncells_per_edge=nair_lauritzen_non_divergent_defaults['ncells_per_edge'],
        dt=nair_lauritzen_non_divergent_defaults['dt'],
        tmax=nair_lauritzen_non_divergent_defaults['tmax'],
        dumpfreq=nair_lauritzen_non_divergent_defaults['dumpfreq'],
        dirname=nair_lauritzen_non_divergent_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    tau = 12.*24.*60.*60.    # time period for reversible flow, in s
    radius = 6371220.        # radius of sphere, in m
    theta_c1 = 0.0           # latitude of first blob, in rad
    theta_c2 = 0.0           # latitude of second blob, in rad
    lamda_c1 = -pi/4         # longitude of first blob, in rad
    lamda_c2 = pi/4          # longitude of second blob, in rad

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    xyz = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', degree)

    # Equation
    V = domain.spaces("DG")
    eqn = AdvectionEquation(domain, V, "D")

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
    )

    io = IO(domain, output)

    # Details of transport
    transport_scheme = SSPRK3(domain)
    transport_method = DGUpwind(eqn, "D")

    # Transporting wind ------------------------------------------------------ #
    lamda, theta, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    # Set up the non-divergent, time-varying, velocity field
    def u_t(t):
        k = 10*radius/tau
        lamda_prime = lamda - 2*pi*t/tau

        # Divergence-free wind, obtained from stream function
        psi_expr = (
            k*(sin(lamda_prime)*cos(theta))**2*cos(pi*t/tau)
            - 2.*pi*sin(theta)/tau
        )

        return domain.perp(grad(psi_expr))

    # Time stepper
    stepper = PrescribedTransport(
        eqn, transport_scheme, io, transport_method,
        prescribed_transporting_velocity=u_t
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    if initial_conditions == 'cosine_bells':

        d1 = min_value(1.0, 0.5*great_arc_angle(lamda, theta, lamda_c1, theta_c1))
        d2 = min_value(1.0, 0.5*great_arc_angle(lamda, theta, lamda_c2, theta_c2))
        Dexpr = 0.5*(1 + cos(pi*d1)) + 0.5*(1 + cos(pi*d2))

    elif initial_conditions == 'gaussian':

        X = cos(theta)*cos(lamda)
        Y = cos(theta)*sin(lamda)
        Z = sin(theta)

        X1 = cos(theta_c1)*cos(lamda_c1)
        Y1 = cos(theta_c1)*sin(lamda_c1)
        Z1 = sin(theta_c1)

        X2 = cos(theta_c2)*cos(lamda_c2)
        Y2 = cos(theta_c2)*sin(lamda_c2)
        Z2 = sin(theta_c2)

        # Define the two Gaussian bumps
        g1 = exp(-5*((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2))
        g2 = exp(-5*((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2))

        Dexpr = g1 + g2

    elif initial_conditions == 'slotted_cylinder':

        Dexpr = conditional(
            great_arc_angle(lamda, theta, lamda_c1, theta_c1) < 0.5,
            conditional(
                abs(lamda - lamda_c1) < 1./12.,
                conditional(theta - theta_c1 < -5./24., 1.0, 0.1),
                1.0
            ),
            conditional(
                great_arc_angle(lamda, theta, lamda_c2, theta_c2) < 0.5,
                conditional(
                    abs(lamda - lamda_c2) < 1./12.,
                    conditional(theta - theta_c2 > 5./24., 1.0, 0.1),
                    1.0
                ),
                0.1
            )
        )

    else:
        raise ValueError('Specified initial condition is not valid')

    # Set fields
    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    D0.interpolate(Dexpr)
    u0.project(u_t(0))

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
        '--initial_conditions',
        help="The initial scalar conditions, 'gaussian', 'cosine_bells' or 'slotted_cylinder",
        type=str,
        default=nair_lauritzen_non_divergent_defaults['initial_conditions']
    )
    parser.add_argument(
        '--ncells_per_edge',
        help="The number of cells per edge of the icosahedron",
        type=int,
        default=nair_lauritzen_non_divergent_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=nair_lauritzen_non_divergent_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=nair_lauritzen_non_divergent_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=nair_lauritzen_non_divergent_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=nair_lauritzen_non_divergent_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    nair_lauritzen_non_divergent(**vars(args))
