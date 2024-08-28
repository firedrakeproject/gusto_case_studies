"""
The Terminator Toy test of Lauritzen et al., 2015:
``The terminator "toy" chemistry test: a simple tool to assess errors in
transport schemes'', GMD.

This test transports two interacting chemicals on the surface of a sphere,
representing the combination the combination and dissociation of two species.

The setup here uses an icosahedral sphere with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    exp, cos, sin, SpatialCoordinate, pi, max_value, sqrt, Constant
)
from gusto import (
    Domain, ActiveTracer, OutputParameters, IO, lonlatr_from_xyz, SSPRK3,
    DGUpwind, CoupledTransportEquation, BackwardEuler, DG1Limiter, Sum,
    TerminatorToy, TransportEquationType, TracerVariableType, TracerDensity,
    MixedFSLimiter, SplitPrescribedTransport, GeneralIcosahedralSphereMesh,
    great_arc_angle, xyz_vector_from_lonlatr, xyz_from_lonlatr
)

terminator_toy_defaults = {
    'ncells_per_edge': 16,     # num points per icosahedron edge (ref level 4)
    'dt': 900.0,              # 15 minutes
    'tmax': 12.*24.*60.*60.,  # 12 days
    'dumpfreq': 288,          # once every 3 days with default values
    'dirname': 'terminator_toy'
}


def terminator_toy(
        ncells_per_edge=terminator_toy_defaults['ncells_per_edge'],
        dt=terminator_toy_defaults['dt'],
        tmax=terminator_toy_defaults['tmax'],
        dumpfreq=terminator_toy_defaults['dumpfreq'],
        dirname=terminator_toy_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    tau = 12.*24.*60.*60.  # time period of reversible wind, in s
    radius = 6371220.      # radius of the sphere, in m
    theta_cr = pi/9.       # central latitude of first reaction rate, in rad
    lamda_cr = -pi/3.      # central longitude of first reaction rate, in rad
    k1_max = 1.            # amplitude of first reaction rate parameter, in 1/s
    k2 = 1.                # second reaction rate parameter, in 1/s
    theta_c1 = 0.          # central latitude of first chemical blob, in rad
    theta_c2 = 0.          # central latitude of second chemical blob, in rad
    lamda_c1 = -pi/4.      # central longitude of first chemical blob, in rad
    lamda_c2 = pi/4.       # central longitude of second chemical blob, in rad

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)

    domain = Domain(mesh, dt, 'BDM', degree)

    # Define the dry density and the two species as tracers
    rho_d = ActiveTracer(
        name='rho_d', space='DG',
        variable_type=TracerVariableType.density,
        transport_eqn=TransportEquationType.conservative
    )

    X = ActiveTracer(
        name='X', space='DG',
        variable_type=TracerVariableType.mixing_ratio,
        transport_eqn=TransportEquationType.advective
    )

    X2 = ActiveTracer(
        name='X2', space='DG',
        variable_type=TracerVariableType.mixing_ratio,
        transport_eqn=TransportEquationType.advective
    )

    tracers = [rho_d, X, X2]

    # Equation
    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
    )

    # Define intermediate sums to be able to use the TracerDensity diagnostic
    X_plus_X2 = Sum('X', 'X2')
    X_plus_X2_plus_X2 = Sum('X_plus_X2', 'X2')
    tracer_diagnostic = TracerDensity('X_plus_X2_plus_X2', 'rho_d')
    diagnostic_fields = [X_plus_X2, X_plus_X2_plus_X2, tracer_diagnostic]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Details of transport
    # Define limiters for the interacting species
    limiter_space = domain.spaces('DG')
    sublimiters = {
        'rho_d': DG1Limiter(limiter_space),
        'X': DG1Limiter(limiter_space),
        'X2': DG1Limiter(limiter_space)
    }
    MixedLimiter = MixedFSLimiter(eqn, sublimiters)

    transport_scheme = SSPRK3(domain, limiter=MixedLimiter)
    transport_method = [
        DGUpwind(eqn, 'rho_d'), DGUpwind(eqn, 'X'), DGUpwind(eqn, 'X2')
    ]

    # Physics scheme --------------------------------------------------------- #
    # Define the k1 reaction rate
    xyz = SpatialCoordinate(mesh)
    lamda, theta, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    k1 = max_value(
        0, k1_max*cos(great_arc_angle(lamda, theta, lamda_cr, theta_cr))
    )

    terminator_stepper = BackwardEuler(domain)

    physics_schemes = [
        (TerminatorToy(eqn, k1=k1, k2=k2, species1_name='X', species2_name='X2'),
         terminator_stepper)
    ]

    # Timestepper that solves the physics separately to the dynamics
    # with a defined prescribed transporting velocity
    time_varying_velocity = True
    stepper = SplitPrescribedTransport(
        eqn, transport_scheme, io, time_varying_velocity,
        spatial_methods=transport_method, physics_schemes=physics_schemes
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Transporting wind ------------------------------------------------------ #
    # Set up a non-divergent, time-varying, velocity field
    def u_t(t):
        k = 10*radius/tau

        u_zonal = (
            k*(sin(lamda - 2*pi*t/tau)**2)*sin(2*theta)*cos(pi*t/tau)
            + ((2*pi*radius)/tau)*cos(theta)
        )
        u_merid = k*sin(2*(lamda - 2*pi*t/tau))*cos(theta)*cos(pi*t/tau)

        return xyz_vector_from_lonlatr(u_zonal, u_merid, Constant(0.0), xyz)

    stepper.setup_prescribed_expr(u_t)

    X, Y, Z = xyz
    X1, Y1, Z1 = xyz_from_lonlatr(lamda_c1, theta_c1, radius)
    X2, Y2, Z2 = xyz_from_lonlatr(lamda_c2, theta_c2, radius)

    b0 = 5

    # The initial condition for the density is two Gaussians
    g1 = exp(-(b0/(radius**2))*((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2))
    g2 = exp(-(b0/(radius**2))*((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2))
    rho_expr = g1 + g2

    X_T_0 = 4e-6
    r = k1/(4*k2)
    D_val = sqrt(r**2 + 2*X_T_0*r)

    # Initial condition for each species
    X_0 = D_val - r
    X2_0 = 0.5*(X_T_0 - D_val + r)

    # Initial conditions
    stepper.fields("rho_d").interpolate(rho_expr)
    stepper.fields("X").interpolate(X_0)
    stepper.fields("X2").interpolate(X2_0)

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
        default=terminator_toy_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=terminator_toy_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=terminator_toy_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=terminator_toy_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=terminator_toy_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    terminator_toy(**vars(args))
