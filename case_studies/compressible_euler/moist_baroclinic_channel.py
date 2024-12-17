"""
The moist baroclinic wave in a channel from the appendix of Ullrich, Reed &
Jablonowski, 2015:
``Analytical initial conditions and an analysis of baroclinic instability waves
in f - and β-plane 3D channel models'', QJRMS.

This test emulates a moist baroclinic wave in a channel, with two moisture
variables with associated latent heating as they change phase.

The setup here is for the order 1 finite elements, in a 3D slice which is
periodic in the x direction but with rigid walls in the y direction.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    PeriodicRectangleMesh, ExtrudedMesh, SpatialCoordinate, conditional, cos,
    sin, pi, sqrt, ln, exp, Constant, Function, as_vector, errornorm, norm
)

from gusto import (
    Domain, CompressibleParameters, CompressibleSolver, WaterVapour, CloudWater,
    CompressibleEulerEquations, OutputParameters, IO, logger, SSPRK3,
    DGUpwind, SemiImplicitQuasiNewton, compressible_hydrostatic_balance,
    Perturbation, SaturationAdjustment, ForwardEuler, thermodynamics
)

moist_baroclinic_channel_defaults = {
    'nx': 160,                 # number of columns in x-direction
    'ny': 24,                  # number of columns in y-direction
    'nlayers': 20,             # number of layers in mesh
    'dt': 300.0,               # 5 minutes
    'tmax': 15.*24.*60.*60.,   # 15 days
    'dumpfreq': 144,           # Corresponds to every 12 hours with default opts
    'dirname': 'moist_baroclinic_channel'
}


def moist_baroclinic_channel(
        nx=moist_baroclinic_channel_defaults['nx'],
        ny=moist_baroclinic_channel_defaults['ny'],
        nlayers=moist_baroclinic_channel_defaults['nlayers'],
        dt=moist_baroclinic_channel_defaults['dt'],
        tmax=moist_baroclinic_channel_defaults['tmax'],
        dumpfreq=moist_baroclinic_channel_defaults['dumpfreq'],
        dirname=moist_baroclinic_channel_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    Lx = 4.0e7                   # length of domain in x direction, in m
    Ly = 6.0e6                   # width of domain in y direction, in m
    H = 3.0e4                    # height of domain, in m
    omega = Constant(7.292e-5)   # planetary rotation rate, in 1/s
    phi0 = Constant(pi/4)        # latitude of centre of channel, in radians
    a = Constant(6.371229e6)     # radius of earth, in m
    b = Constant(2)              # vertical width parameter, dimensionless
    T0 = Constant(288.)          # reference temperature, in K
    u0 = Constant(35.)           # reference zonal wind speed, in m/s
    Gamma = Constant(0.005)      # lapse rate, in K/m
    beta0 = Constant(0.0)        # beta-plane parameter, in 1/s
    eta_w = Constant(0.3)        # moisture height parameter, dimensionless
    deltay_w = Constant(3.2e6)   # moisture meridional parameter, in m
    q0 = Constant(0.016)         # specific humidity parameter, in kg/kg
    xc = 2.0e6                   # x coordinate for centre of perturbation, in m
    yc = 2.5e6                   # y coordinate for centre of perturbation, in m
    Lp = 6.0e5                   # width parameter for perturbation, in m
    up = Constant(1.0)           # strength of wind perturbation, in m/s

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #
    # NB: this test seems to be unstable with 2x2 iterations
    num_outer = 4
    num_inner = 1
    element_order = 1
    u_eqn_type = 'vector_invariant_form'
    max_iterations = 40          # max num of iterations for finding eta coords
    tolerance = 1e-10            # tolerance of error in finding eta coords

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicRectangleMesh(nx, ny, Lx, Ly, "x", quadrilateral=True)
    mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "RTCF", element_order)
    x, y, z = SpatialCoordinate(mesh)

    # Equation
    params = CompressibleParameters(Omega=omega*sin(phi0))
    tracers = [WaterVapour(), CloudWater()]
    eqns = CompressibleEulerEquations(
        domain, params, active_tracers=tracers,
        no_normal_flow_bc_ids=[1, 2], u_transport_option=u_eqn_type
    )

    # I/O
    dirname = 'moist_baroclinic_channel'
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False,
        dumplist=['cloud_water']
    )
    diagnostic_fields = [Perturbation('theta')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [
        SSPRK3(domain, "u"),
        SSPRK3(domain, "rho"),
        SSPRK3(domain, "theta"),
        SSPRK3(domain, "water_vapour"),
        SSPRK3(domain, "cloud_water")
    ]

    transport_methods = [
        DGUpwind(eqns, field) for field in
        ["u", "rho", "theta", "water_vapour", "cloud_water"]
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Physics schemes
    physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, spatial_methods=transport_methods,
        linear_solver=linear_solver, physics_schemes=physics_schemes,
        num_outer=num_outer, num_inner=num_inner
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Physical parameters
    beta0 = 2 * omega * cos(phi0) / a
    Rd = params.R_d
    Rv = params.R_v
    f0 = 2 * omega * sin(phi0)
    y0 = Constant(Ly / 2)
    g = params.g
    p0 = params.p_0

    # Initial conditions
    u = stepper.fields("u")
    rho = stepper.fields("rho")
    theta = stepper.fields("theta")
    water_v = stepper.fields("water_vapour")
    water_c = stepper.fields("cloud_water")

    # spaces
    Vu = u.function_space()
    Vt = theta.function_space()
    Vr = rho.function_space()

    # set up background state expressions
    eta = Function(Vt).interpolate(Constant(1e-7))
    Phi = Function(Vt).interpolate(g * z)
    q = Function(Vt)
    T = Function(Vt)
    Phi_prime = u0 / 2 * (
        (f0 - beta0 * y0) * (y - (Ly / 2) - (Ly / (2 * pi)) * sin(2*pi*y/Ly))
        + beta0 / 2*(
            y**2 - (Ly * y / pi) * sin(2*pi*y/Ly)
            - (Ly**2 / (2 * pi**2)) * cos(2*pi*y/Ly) - (Ly**2 / 3)
            - (Ly**2 / (2 * pi**2))
        )
    )
    Phi_expr = (
        T0 * g / Gamma * (1 - eta ** (Rd * Gamma / g))
        + Phi_prime * ln(eta) * exp(-(ln(eta) / b) ** 2)
    )

    Tv_expr = (
        T0 * eta ** (Rd * Gamma / g) + Phi_prime / Rd * exp(-(ln(eta) / b)**2)
        * ((2 / b**2) * (ln(eta)) ** 2 - 1)
    )
    u_expr = as_vector(
        [-u0 * (sin(pi*y/Ly))**2 * ln(eta) * eta ** (-ln(eta) / b ** 2),
         0.0, 0.0]
    )
    q_bar = q0 / 2 * conditional(
        eta > eta_w, (1 + cos(pi * (1 - eta) / (1 - eta_w))), 0.0
    )
    q_expr = q_bar * exp(-(y / deltay_w) ** 4)
    r_expr = q / (1 - q)
    T_expr = Tv_expr / (1 + q * (Rv / Rd - 1))

    # do Newton method to obtain eta
    eta_new = Function(Vt)
    F = -Phi + Phi_expr
    dF = -Rd * Tv_expr / eta
    for _ in range(max_iterations):
        eta_new.interpolate(eta - F/dF)
        if errornorm(eta_new, eta) / norm(eta) < tolerance:
            eta.assign(eta_new)
            break
        eta.assign(eta_new)

    # make mean u and theta
    u.project(u_expr)
    q.interpolate(q_expr)
    water_v.interpolate(r_expr)
    water_c.interpolate(Constant(0.0))
    T.interpolate(T_expr)
    theta.interpolate(
        thermodynamics.theta(params, T_expr, p0 * eta) * (1 + water_v * Rv / Rd)
    )
    Phi_test = Function(Vt).interpolate(Phi_expr)
    logger.info(
        f"Error-norm for setting up p: {errornorm(Phi_test, Phi) / norm(Phi)}"
    )

    # Calculate hydrostatic fields
    compressible_hydrostatic_balance(
        eqns, theta, rho, mr_t=water_v, solve_for_rho=True
    )

    # make mean fields
    rho_b = Function(Vr).assign(rho)
    u_b = stepper.fields("ubar", space=Vu, dump=False).project(u)
    theta_b = Function(Vt).assign(theta)
    water_vb = Function(Vt).assign(water_v)

    # define perturbation
    r = sqrt((x - xc) ** 2 + (y - yc) ** 2)
    u_pert = Function(Vu).project(as_vector([up * exp(-(r / Lp)**2), 0.0, 0.0]))

    # define initial u
    u.assign(u_b + u_pert)

    # initialise fields
    stepper.set_reference_profiles(
        [('rho', rho_b), ('theta', theta_b), ('water_vapour', water_vb)]
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
        '--nx',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=moist_baroclinic_channel_defaults['nx']
    )
    parser.add_argument(
        '--ny',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=moist_baroclinic_channel_defaults['ny']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=moist_baroclinic_channel_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_baroclinic_channel_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_baroclinic_channel_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_baroclinic_channel_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_baroclinic_channel_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    moist_baroclinic_channel(**vars(args))
