"""
The Williamson 5 test case, with the thermal shallow water equations.
The initial conditions are taken from Hartney et al, 2024: ``A compatible finite
element discretisation for moist shallow water equations''.
"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, as_vector, pi, sqrt, min_value, cos, sin, Function
)
from gusto import (
    Domain, IO, OutputParameters, DGUpwind, SubcyclingOptions,
    ShallowWaterParameters, ThermalShallowWaterEquations, Sum,
    lonlatr_from_xyz, GeneralIcosahedralSphereMesh, RelativeVorticity,
    ZonalComponent, MeridionalComponent, RungeKuttaFormulation, SSPRK3,
    SemiImplicitQuasiNewton, ThermalSWSolver
)

thermal_williamson_5_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 1200.0,              # 10 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 72,            # once per day with default options
    'dirname': 'thermal_williamson_5'
}


def thermal_williamson_5(
        ncells_per_edge=thermal_williamson_5_defaults['ncells_per_edge'],
        dt=thermal_williamson_5_defaults['dt'],
        tmax=thermal_williamson_5_defaults['tmax'],
        dumpfreq=thermal_williamson_5_defaults['dumpfreq'],
        dirname=thermal_williamson_5_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.           # planetary radius (m)
    mean_depth = 5960           # reference depth (m)
    g = 9.80616                 # acceleration due to gravity (m/s^2)
    u_max = 20.                 # max amplitude of the zonal wind (m/s)
    epsilon = 1/300             # linear air expansion coeff (1/K)
    theta_SP = -40*epsilon      # value of theta at south pole (no units)
    theta_EQ = 30*epsilon       # value of theta at equator (no units)
    theta_NP = -20*epsilon      # value of theta at north pole (no units)
    mu1 = 0.05                  # scaling for theta with longitude (no units)
    mountain_height = 2000.     # height of mountain (m)
    R0 = pi/9.                  # radius of mountain (rad)
    lamda_c = -pi/2.            # longitudinal centre of mountain (rad)
    phi_c = pi/6.               # latitudinal centre of mountain (rad)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    alpha = 0.5
    element_order = 1
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, "BDM", element_order)
    x, y, z = SpatialCoordinate(mesh)
    lamda, phi, _ = lonlatr_from_xyz(x, y, z)

    # Coriolis
    parameters = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = parameters.Omega
    fexpr = 2*Omega*z/radius

    # Mountain
    rsq = min_value(R0**2, (lamda - lamda_c)**2 + (phi - phi_c)**2)
    r = sqrt(rsq)
    tpexpr = mountain_height * (1 - r/R0)

    # Equation
    eqns = ThermalShallowWaterEquations(
        domain, parameters, fexpr=fexpr, topog_expr=tpexpr, thermal=True,
        u_transport_option=u_eqn_type
    )

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True,
    )
    diagnostic_fields = [
        Sum('D', 'topography'), RelativeVorticity(),
        ZonalComponent('u'), MeridionalComponent('u')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport
    subcycling_opts = SubcyclingOptions(subcycle_by_courant=0.25)
    transported_fields = [
        SSPRK3(domain, "u", subcycling_options=subcycling_opts),
        SSPRK3(
            domain, "D", subcycling_options=subcycling_opts,
            rk_formulation=RungeKuttaFormulation.linear
        ),
        SSPRK3(domain, "b", subcycling_options=subcycling_opts),
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "D", advective_then_flux=True),
        DGUpwind(eqns, "b"),
    ]

    # Linear solver
    linear_solver = ThermalSWSolver(eqns, alpha=alpha)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver, alpha=alpha,
        num_outer=2, num_inner=2
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")

    uexpr = as_vector([-u_max*y/radius, u_max*x/radius, 0.0])

    Dexpr = (
        mean_depth - tpexpr
        - (radius * Omega * u_max + 0.5*u_max**2)*(z/radius)**2/g
    )

    # Expression for initial buoyancy - note the bracket around 1-mu
    theta_expr = (
        2/(pi**2) * (
            phi*(phi - pi/2)*theta_SP
            - 2*(phi + pi/2) * (phi - pi/2)*(1 - mu1)*theta_EQ
            + phi*(phi + pi/2)*theta_NP
        )
        + mu1*theta_EQ*cos(phi)*sin(lamda)
    )
    bexpr = g * (1 - theta_expr)

    # Initialise
    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)

    # Set reference profiles
    Dbar = Function(D0.function_space()).assign(mean_depth)
    bbar = Function(b0.function_space()).interpolate(bexpr)
    stepper.set_reference_profiles([('D', Dbar), ('b', bbar)])

    # ----------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------- #

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
        default=thermal_williamson_5_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=thermal_williamson_5_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=thermal_williamson_5_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=thermal_williamson_5_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=thermal_williamson_5_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    thermal_williamson_5(**vars(args))
