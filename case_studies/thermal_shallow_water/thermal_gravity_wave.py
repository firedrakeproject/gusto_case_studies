"""
A gravity wave on the sphere, solved with the thermal shallow water equations.
The initial conditions are formed by adding a perturbation to the thermal
solid-body rotation test case of  Zerroukat & Allen, 2015:
``A moist Boussinesq shallow water equations set for testing atmospheric
models'', JCP.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, pi, sqrt, min_value, cos, Function, sin
)
from gusto import (
    Domain, IO, OutputParameters, DGUpwind, ShallowWaterParameters,
    ThermalShallowWaterEquations, lonlatr_from_xyz, SubcyclingOptions,
    RungeKuttaFormulation, SSPRK3, MeridionalComponent,
    SemiImplicitQuasiNewton, xyz_vector_from_lonlatr, ZonalComponent,
    GeneralIcosahedralSphereMesh, RelativeVorticity
)

thermal_gw_defaults = {
    'ncells_per_edge': 16,         # number of cells per icosahedron edge
    'dt': 900.0,                   # 15 minutes
    'tmax': 5.*24.*60.*60.,        # 5 days
    'dumpfreq': 96,                # dump once per day
    'dirname': 'thermal_gw'
}


def thermal_gw(
        ncells_per_edge=thermal_gw_defaults['ncells_per_edge'],
        dt=thermal_gw_defaults['dt'],
        tmax=thermal_gw_defaults['tmax'],
        dumpfreq=thermal_gw_defaults['dumpfreq'],
        dirname=thermal_gw_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.           # planetary radius (m)
    R0 = pi/9.                  # radius of perturbation (rad)
    lamda_c = -pi/2.            # longitudinal centre of perturbation (rad)
    phi_c = pi/6.               # latitudinal centre of perturbation (rad)
    phi_0 = 3.0e4               # scale factor for poleward buoyancy gradient
    epsilon = 1/300             # linear air expansion coeff (1/K)
    u_max = 20.                 # max amplitude of the zonal wind (m/s)
    g = 9.80616                 # acceleration due to gravity (m/s^2)
    mean_depth = phi_0/g        # reference depth (m)

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    degree = 1
    domain = Domain(mesh, dt, "BDM", degree)
    xyz = SpatialCoordinate(mesh)

    # Equation parameters
    parameters = ShallowWaterParameters(mesh, H=mean_depth, g=g)

    # Equation
    eqns = ThermalShallowWaterEquations(domain, parameters)

    # IO
    diagnostic_fields = [
        MeridionalComponent('u'), ZonalComponent('u'), RelativeVorticity()
    ]
    dumplist = ['b', 'D']

    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False,
        dumplist=dumplist
    )
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

    # ------------------------------------------------------------------------ #
    # Timestepper
    # ------------------------------------------------------------------------ #

    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods, predictor='D',
        tau_values={'D': 1.0, 'b': 1.0}, reference_update_freq=10800.
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")

    lamda, phi, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    # Velocity -- a solid body rotation
    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, xyz)

    # Buoyancy -- dependent on latitude
    g = parameters.g
    w = parameters.Omega*radius*u_max + (u_max**2)/2
    sigma = w/10
    theta_0 = epsilon*phi_0**2
    numerator = (
        theta_0 + sigma*((cos(phi))**2) * (
            (w + sigma)*(cos(phi))**2 + 2*(phi_0 - w - sigma)
        )
    )
    denominator = (
        phi_0**2 + (w + sigma)**2
        * (sin(phi))**4 - 2*phi_0*(w + sigma)*(sin(phi))**2
    )
    theta = numerator / denominator
    bexpr = parameters.g * (1 - theta)

    # Depth -- in balance before the addition of a perturbation
    Dexpr = mean_depth - (1/g)*(w + sigma)*((sin(phi))**2)

    # Perturbation
    lsq = (lamda - lamda_c)**2
    thsq = (phi - phi_c)**2
    rsq = min_value(R0**2, lsq+thsq)
    r = sqrt(rsq)
    pert = 2000 * (1 - r/R0)
    Dexpr += pert

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)

    # Set reference profiles to initial state
    Dbar = Function(D0.function_space()).interpolate(Dexpr)
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
        default=thermal_gw_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=thermal_gw_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=thermal_gw_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=thermal_gw_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=thermal_gw_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    thermal_gw(**vars(args))
