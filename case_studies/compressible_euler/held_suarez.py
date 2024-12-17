"""
The Held suarez test case from Held-Saurez 1994, "A Proposal for the and
intercomparison of the dynamical cores of Atmospheric general circulation models".

Initial conditions are from the unperturbed deep atmosphere dry baroclinic wave from Ullrich et al. 2014:
``A proposed baroclinic wave test case for deep- and shallow-atmosphere
dynamical cores'', QJRMS.

This is a 3D test on the sphere, with an initial state that is in unsteady
balance, forcings are then applied to the temperature and potential temperature
field to 'relax' the atmosphere to a given state.

This setup uses a cubed-sphere with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (ExtrudedMesh, SpatialCoordinate, cos, sqrt, exp, Constant,
                       Function, errornorm, norm, TestFunction, dx, inner,
                       NonlinearVariationalProblem, NonlinearVariationalSolver)
from gusto import (EmbeddedDGOptions, GeneralCubedSphereMesh, lonlatr_from_xyz, xyz_vector_from_lonlatr,
                   Domain, CompressibleEulerEquations, CompressibleParameters, OutputParameters,
                   MeridionalComponent, ZonalComponent, Pressure, RadialComponent, Temperature,
                   CourantNumber, CompressibleKineticEnergy, PotentialEnergy, CompressibleSolver,
                   DGUpwind, Relaxation, SSPRK3, SemiImplicitQuasiNewton, RayleighFriction, ForwardEuler,
                   BackwardEuler, compressible_hydrostatic_balance, IO, logger)

held_suarez_defaults = {
    'ncell_per_edge': 16,
    'nlayers': 15,
    'dt': 900.0,               # 15 minutes
    'tmax': 15.*24.*60.*60.,   # 15 days
    'dumpfreq': 48,            # Corresponds to every 12 hours with default options
    'dirname': 'held_suarez'
}


def held_suarez(
        ncell_per_edge=held_suarez_defaults['ncell_per_edge'],
        nlayers=held_suarez_defaults['nlayers'],
        dt=held_suarez_defaults['dt'],
        tmax=held_suarez_defaults['tmax'],
        dumpfreq=held_suarez_defaults['dumpfreq'],
        dirname=held_suarez_defaults['dirname']
):
    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #
    a = 6.371229e6    # radius of planet, in m
    htop = 3.0e4      # height of top of atmosphere above surface, in m
    omega = 7.292e-5  # rotation frequency of planet, in 1/s
    lapse = 0.005     # thermal lapse rate
    T0e = 310         # equatorial temperature, in K
    T0p = 240         # polar surface temperature, in K
    lapse = 0.005     # lapse rate of atmosphere, in temp / m
    T0p = 240         # stratosphere temp
    T0e = 310         # surface temperature at equator
    alpha = 0.50      # ratio between implicit and explict in solver
    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #
    horder = 1        # horizontal order of finite element de Rham complex
    vorder = 1        # vertical order of finite element de Rham complex
    u_eqn_type = 'vector_advection_form'  # Form of the momentum equation to use

    # -------------------------------------------------------------- #
    # Model Objects
    # -------------------------------------------------------------- #
    # Domain
    # Layers are not evenly based -- compute level heights here
    layer_height = []
    running_height = 0
    for m in range(1, nlayers+1):
        mu = 15
        height = htop * (
            ((mu * (m / nlayers)**2 + 1)**0.5 - 1)
            / ((mu + 1)**0.5 - 1)
        )
        depth = height - running_height
        running_height = height
        layer_height.append(depth)

    # Mesh
    base_mesh = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=ncell_per_edge, degree=2)
    mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=layer_height, extrusion_type='radial')
    domain = Domain(mesh, dt, "RTCF", horizontal_degree=horder, vertical_degree=vorder)

    # Equations
    params = CompressibleParameters(Omega=omega)

    eqn = CompressibleEulerEquations(
        domain, params, u_transport_option=u_eqn_type
    )

    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dump_nc=True,
                              dump_vtus=False)
    diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'), RadialComponent('u'),
                         CourantNumber(), Temperature(eqn), Pressure(eqn),
                         CompressibleKineticEnergy(), PotentialEnergy(eqn)]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport Schemes
    transported_fields = []
    transport_methods = []

    theta_opts = EmbeddedDGOptions()
    transported_fields = [
        SSPRK3(domain, "u"),
        SSPRK3(domain, "rho"),
        SSPRK3(domain, "theta", options=theta_opts)
    ]
    transport_methods = [
        DGUpwind(eqn, "u"),
        DGUpwind(eqn, "rho"),
        DGUpwind(eqn, "theta")
    ]

    # Linear Solver
    linear_solver = CompressibleSolver(eqn, alpha=alpha)

    physics_schemes = [(Relaxation(eqn, 'theta', parameters=params), ForwardEuler(domain)),
                       (RayleighFriction(eqn, parameters=params), BackwardEuler(domain))]
    # Time Stepper
    stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                      transport_methods,
                                      linear_solver=linear_solver, alpha=alpha,
                                      physics_schemes=physics_schemes,
                                      num_outer=4, num_inner=1)

    # -------------------------------------------------------------- #
    # Initial Conditions
    # -------------------------------------------------------------- #
    x, y, z = SpatialCoordinate(mesh)
    _, lat, r = lonlatr_from_xyz(x, y, z)

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vu = u0.function_space()
    Vr = rho0.function_space()
    Vt = theta0.function_space()

    # Steady state -------------------------------------------------------------

    # Extract default atmospheric parameters
    Rd = params.R_d
    g = params.g
    p0 = params.p_0

    # Some temporary variables
    T0 = 0.5*(T0e + T0p)
    H = Rd*T0/g      # scale height of atmosphere
    k = 3            # power of temperature field
    b = 2            # half width parameter

    # Expressions for temporary variables from paper
    s = (r/a)*cos(lat)
    A = 1/lapse
    B = (T0e - T0p)/((T0e + T0p)*T0p)
    C = ((k + 2)/2)*((T0e - T0p)/(T0e*T0p))

    tau1 = A*lapse*exp((r - a)*lapse/T0)/T0
    tau1 += B * (1 - 2*((r - a)/(b*H))**2) * exp(-((r - a) / (b*H))**2)

    tau2 = C * (1 - 2*((r - a)/(b*H))**2) * exp(-((r - a) / (b*H))**2)

    tau1_integral = A * (exp(lapse * (r - a) / T0) - 1)
    tau1_integral += B * (r - a) * exp(-((r - a) / (b*H))**2)

    tau2_integral = C * (r - a) * exp(-((r - a) / (b*H))**2)

    # Temperature and pressure fields
    T_expr = (a / r)**2 / (
        tau1 - tau2 * (s**k - (k/(k + 2)) * s**(k + 2))
    )
    P_expr = p0 * exp(
        - g/Rd * tau1_integral
        + g/Rd * tau2_integral * (s**k - (k / (k + 2)) * s**(k + 2))
    )

    # wind expression
    wind_proxy = (
        (g/a)*k*T_expr*tau2_integral*(
            ((r*cos(lat))/a)**(k - 1) - ((r*cos(lat))/a)**(k + 1)
        )
    )
    wind = (
        - omega*r*cos(lat)
        + sqrt((omega*r*cos(lat))**2 + r*cos(lat)*wind_proxy)
    )

    theta_expr = T_expr*(P_expr/p0)**(- params.kappa)
    exner_expr = T_expr/theta_expr
    rho_expr = P_expr/(Rd*T_expr)

    zonal_u = wind
    merid_u = Constant(0.0)
    radial_u = Constant(0.0)

    # Get spherical basis vectors, expressed in terms of (x,y,z) components
    e_lon = xyz_vector_from_lonlatr(1, 0, 0, (x, y, z))
    e_lat = xyz_vector_from_lonlatr(0, 1, 0, (x, y, z))
    e_r = xyz_vector_from_lonlatr(0, 0, 1, (x, y, z))

    # Obtain initial conditions -- set up projection manually to
    # manually specify a reduced quadrature degree
    logger.info('Set up initial conditions')
    logger.debug('project u')
    test_u = TestFunction(Vu)
    dx_reduced = dx(degree=4)
    u_field = zonal_u*e_lon + merid_u*e_lat + radial_u*e_r
    u_proj_eqn = inner(test_u, u0 - u_field)*dx_reduced
    u_proj_prob = NonlinearVariationalProblem(u_proj_eqn, u0)
    u_proj_solver = NonlinearVariationalSolver(u_proj_prob)
    u_proj_solver.solve()

    theta0.interpolate(theta_expr)
    exner = Function(Vr).interpolate(exner_expr)
    rho0.interpolate(rho_expr)

    logger.info('find rho by solving hydrostatic balance')
    compressible_hydrostatic_balance(
        eqn, theta0, rho0, exner_boundary=exner, solve_for_rho=True
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
        default=held_suarez_defaults['ncell_per_edge']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=held_suarez_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=held_suarez_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=held_suarez_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=held_suarez_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=held_suarez_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    held_suarez(**vars(args))
