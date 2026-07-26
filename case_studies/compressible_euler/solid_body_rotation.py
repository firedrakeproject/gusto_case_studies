"""
A solid body rotation case from Staniforth & White (2007).

This is a 3D test on the sphere, with an initial state that is in unsteady
balance, with a perturbation added to the wind.

This setup uses a cubed-sphere with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    ExtrudedMesh, SpatialCoordinate, cos, sqrt, exp, Constant,
    Function, errornorm, norm, inner, dx,
    NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
)
from gusto import (
    GeneralCubedSphereMesh, CompressibleParameters, CompressibleEulerEquations,
    OutputParameters, logger, SIQNModel, lonlatr_from_xyz, Temperature,
    xyz_vector_from_lonlatr, compressible_hydrostatic_balance, Pressure,
    ZonalComponent, MeridionalComponent, RadialComponent
)

solid_body_sphere_defaults = {
    'ncell_per_edge': 16,
    'nlayers': 15,
    'dt': 900.0,               # 15 minutes
    'tmax': 30.*24.*60.*60.,   # 30 days
    'dumpfreq': 96,            # Corresponds to every 24 hours with default opts
    'dirname': 'solid_body_sphere'
}


def solid_body_sphere(
        ncell_per_edge=solid_body_sphere_defaults['ncell_per_edge'],
        nlayers=solid_body_sphere_defaults['nlayers'],
        dt=solid_body_sphere_defaults['dt'],
        tmax=solid_body_sphere_defaults['tmax'],
        dumpfreq=solid_body_sphere_defaults['dumpfreq'],
        dirname=solid_body_sphere_defaults['dirname']
):
    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    a = 6.371229e6    # radius of planet, in m
    htop = 3.0e4      # height of top of atmosphere above surface, in m
    omega = 7.292e-5  # rotation frequency of planet, in 1/s
    T0 = 280.         # background temperature in K
    u0 = 40.          # initial Zonal wind speed, in M/s

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #
    u_eqn_type = 'vector_advection_form'  # Form of the momentum equation to use

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #
    # Domain
    layer_height = []
    running_height = 0
    # Use the DCMIP vertical grid stretching
    for m in range(1, nlayers+1):
        mu = 15
        height = htop * (
            ((mu * (m / nlayers)**2 + 1)**0.5 - 1)
            / ((mu + 1)**0.5 - 1)
        )
        depth = height - running_height
        running_height = height
        layer_height.append(depth)

    base_mesh = GeneralCubedSphereMesh(
        radius=a, num_cells_per_edge_of_panel=ncell_per_edge, degree=2
    )
    mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=layer_height,
        extrusion_type='radial'
    )

    # Equations
    params = CompressibleParameters(mesh, Omega=omega)
    eqn = CompressibleEulerEquations

    # Model
    model = SIQNModel(
        mesh, dt, params, eqn, u_transport_option=u_eqn_type, family="RTCF"
    )

    # Outputting and IO
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
    )
    diagnostic_fields = [
        MeridionalComponent('u'), ZonalComponent('u'), RadialComponent('u'),
        Temperature(model.equation), Pressure(model.equation)
    ]

    # Wrap up model setup
    model.setup(output, diagnostic_fields=diagnostic_fields)

    # ------------------------------------------------------------------------ #
    # Initial Conditions
    # ------------------------------------------------------------------------ #
    x, y, z = SpatialCoordinate(mesh)
    _, lat, r = lonlatr_from_xyz(x, y, z)

    r = sqrt(x**2 + y**2 + z**2)

    # set up parameters
    Rd = params.R_d
    g = params.g
    p0 = Constant(100000)
    T0 = 280.  # in K

    stepper = model.stepper
    vel0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vu = vel0.function_space()
    Vr = rho0.function_space()
    Vt = theta0.function_space()

    # expressions for variables from paper
    s = r * cos(lat)

    Q_expr = (s / a)**2 * (u0**2 + 2 * omega * a * u0) / (2 * Rd * T0)

    # solving fields as per the staniforth paper
    q_expr = Q_expr + ((g * a**2) / (Rd * T0)) * (1/r - 1/a)
    p_expr = p0 * exp(q_expr)
    theta_expr = T0 * (p_expr / p0) ** (-params.kappa)
    pie_expr = T0 / theta_expr
    rho_expr = p_expr / (Rd * T0)

    # get components of u in spherical polar coordinates
    zonal_u = u0 * r / a * cos(lat)
    merid_u = Constant(0.0)
    radial_u = Constant(0.0)

    # Get spherical basis vectors, expressed in terms of (x,y,z) components
    e_lon = xyz_vector_from_lonlatr(1, 0, 0, (x, y, z))
    e_lat = xyz_vector_from_lonlatr(0, 1, 0, (x, y, z))
    e_r = xyz_vector_from_lonlatr(0, 0, 1, (x, y, z))

    # obtain initial conditions
    logger.info('Set up initial conditions')
    logger.debug('project u')
    test_u = TestFunction(Vu)
    dx_reduced = dx(degree=4)
    u_field = zonal_u*e_lon + merid_u*e_lat + radial_u*e_r
    u_proj_eqn = inner(test_u, vel0 - u_field)*dx_reduced
    u_proj_prob = NonlinearVariationalProblem(u_proj_eqn, vel0)
    u_proj_solver = NonlinearVariationalSolver(u_proj_prob)
    u_proj_solver.solve()

    theta0.interpolate(theta_expr)

    exner = Function(Vr).interpolate(pie_expr)
    rho0.interpolate(rho_expr)

    logger.info('find rho by solving hydrostatic balance')
    compressible_hydrostatic_balance(
        model.equation, theta0, rho0, exner_boundary=exner, solve_for_rho=True
    )

    rho_analytic = Function(Vr).interpolate(rho_expr)
    logger.info('Normalised rho error is: '
                + f'{errornorm(rho_analytic, rho0)/norm(rho_analytic)}')

    # make mean fields
    rho_b = Function(Vr).assign(rho0)
    theta_b = Function(Vt).assign(theta0)

    # assign reference profiles
    stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_b)])

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
        '--ncell_per_edge',
        help="The number of cells per panel edge of the cubed-sphere.",
        type=int,
        default=solid_body_sphere_defaults['ncell_per_edge']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=solid_body_sphere_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=solid_body_sphere_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=solid_body_sphere_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=solid_body_sphere_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=solid_body_sphere_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    solid_body_sphere(**vars(args))
