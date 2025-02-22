"""
The moist rising bubble test from Bryan & Fritsch, 2002:
``A Benchmark Simulation for Moist Nonhydrostatic Numerical Models'', GMD.

The test simulates a rising thermal in a cloudy atmosphere, which is fueled by
latent heating from condensation.

This setup uses a vertical slice with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, conditional, cos, pi,
    sqrt, NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
    dx, TrialFunction, Function, as_vector, LinearVariationalProblem,
    LinearVariationalSolver, Constant
)

from gusto import (
    Domain, CompressibleEulerEquations, IO, CompressibleParameters, DGUpwind,
    SSPRK3, TrapeziumRule, SemiImplicitQuasiNewton, EmbeddedDGOptions,
    WaterVapour, CloudWater, OutputParameters, Theta_e, SaturationAdjustment,
    ForwardEuler, saturated_hydrostatic_balance, thermodynamics, Recoverer,
    CompressibleSolver
)

moist_bryan_fritsch_defaults = {
    'ncolumns': 50,
    'nlayers': 50,
    'dt': 2.0,
    'tmax': 1000.0,
    'dumpfreq': 125,
    'dirname': 'moist_bryan_fritsch'
}


def moist_bryan_fritsch(
        ncolumns=moist_bryan_fritsch_defaults['ncolumns'],
        nlayers=moist_bryan_fritsch_defaults['nlayers'],
        dt=moist_bryan_fritsch_defaults['dt'],
        tmax=moist_bryan_fritsch_defaults['tmax'],
        dumpfreq=moist_bryan_fritsch_defaults['dumpfreq'],
        dirname=moist_bryan_fritsch_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    domain_width = 10000.     # domain width, in m
    domain_height = 10000.    # domain height, in m
    zc = 2000.                # vertical centre of bubble, in m
    rc = 2000.                # radius of bubble, in m
    Tdash = 2.0               # strength of temperature perturbation, in K
    Tsurf = 320.0             # background theta_e value, in K
    total_water = 0.02        # total moisture mixing ratio, in kg/kg

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #
    element_order = 1
    u_eqn_type = 'vector_invariant_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=domain_height/nlayers
    )
    domain = Domain(mesh, dt, 'CG', element_order)

    # Equation
    params = CompressibleParameters(mesh)
    tracers = [WaterVapour(), CloudWater()]
    eqns = CompressibleEulerEquations(
        domain, params, active_tracers=tracers, u_transport_option=u_eqn_type)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True
    )
    diagnostic_fields = [Theta_e(eqns)]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [
        SSPRK3(domain, "rho"),
        SSPRK3(domain, "theta", options=EmbeddedDGOptions()),
        SSPRK3(domain, "water_vapour", options=EmbeddedDGOptions()),
        SSPRK3(domain, "cloud_water", options=EmbeddedDGOptions()),
        TrapeziumRule(domain, "u")
    ]

    transport_methods = [
        DGUpwind(eqns, field) for field in
        ["u", "rho", "theta", "water_vapour", "cloud_water"]
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Physics schemes (condensation/evaporation)
    physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver, physics_schemes=physics_schemes
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    water_v0 = stepper.fields("water_vapour")
    water_c0 = stepper.fields("cloud_water")

    # spaces
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")
    x, z = SpatialCoordinate(mesh)
    quadrature_degree = (4, 4)
    dxp = dx(degree=(quadrature_degree))

    # Define constant theta_e and water_t
    theta_e = Function(Vt).assign(Tsurf)
    water_t = Function(Vt).assign(total_water)

    # Calculate hydrostatic fields
    saturated_hydrostatic_balance(eqns, stepper.fields, theta_e, water_t)

    # make mean fields
    theta_b = Function(Vt).assign(theta0)
    rho_b = Function(Vr).assign(rho0)
    water_vb = Function(Vt).assign(water_v0)
    water_cb = Function(Vt).assign(water_t - water_vb)

    # define perturbation
    xc = domain_width / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    theta_pert = Function(Vt).interpolate(
        conditional(
            r > rc,
            0.0,
            Tdash * (cos(pi * r / (2.0 * rc))) ** 2
        )
    )

    # define initial theta
    theta0.interpolate(theta_b * (theta_pert / 300.0 + 1.0))

    # find perturbed rho
    gamma = TestFunction(Vr)
    rho_trial = TrialFunction(Vr)
    a = gamma * rho_trial * dxp
    L = gamma * (rho_b * theta_b / theta0) * dxp
    rho_problem = LinearVariationalProblem(a, L, rho0)
    rho_solver = LinearVariationalSolver(rho_problem)
    rho_solver.solve()

    # find perturbed water_v
    w_v = Function(Vt)
    phi = TestFunction(Vt)
    rho_averaged = Function(Vt)
    rho_recoverer = Recoverer(rho0, rho_averaged)
    rho_recoverer.project()

    exner = thermodynamics.exner_pressure(eqns.parameters, rho_averaged, theta0)
    p = thermodynamics.p(eqns.parameters, exner)
    T = thermodynamics.T(eqns.parameters, theta0, exner, r_v=w_v)
    w_sat = thermodynamics.r_sat(eqns.parameters, T, p)

    w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
    w_problem = NonlinearVariationalProblem(w_functional, w_v)
    w_solver = NonlinearVariationalSolver(w_problem)
    w_solver.solve()

    water_v0.assign(w_v)
    water_c0.assign(water_t - water_v0)

    # wind initially zero
    u0.project(as_vector(
        [Constant(0.0, domain=mesh), Constant(0.0, domain=mesh)]
    ))

    stepper.set_reference_profiles(
        [
            ('rho', rho_b),
            ('theta', theta_b),
            ('water_vapour', water_vb),
            ('cloud_water', water_cb)
        ]
    )

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
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=moist_bryan_fritsch_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=moist_bryan_fritsch_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_bryan_fritsch_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_bryan_fritsch_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_bryan_fritsch_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_bryan_fritsch_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    moist_bryan_fritsch(**vars(args))
