"""
The 1 metre high mountain test case from Melvin et al, 2010:
``An inherently mass-conserving iterative semi-implicit semi-Lagrangian
discretization of the non-hydrostatic vertical-slice equations.'', QJRMS.

This test describes a wave over a 1m high mountain. The domain is smaller than
that in the "non-hydrostatic mountain" case, so the solutions between the
hydrostatic and non-hydrostatic equations should be different.

The setup used here uses the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    as_vector, VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh,
    SpatialCoordinate, tanh, exp, pi, cos, Function, conditional, Mesh, Constant, File
)
from gusto import (
    Domain, CompressibleParameters, CompressibleSolver, logger,
    CompressibleEulerEquations, HydrostaticCompressibleEulerEquations,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton,
    compressible_hydrostatic_balance, SpongeLayerParameters, Exner, ZComponent,
    Perturbation, SUPGOptions, TrapeziumRule, MaxKernel, MinKernel,
    hydrostatic_parameters, WaterVapour, SaturationAdjustment, ForwardEuler, CloudWater,
    SourceSink, unsaturated_hydrostatic_balance, PhysicsParametrisation, TimeDiscretisation
)

import gusto.equations.thermodynamics as td

mountain_nonhydrostatic_defaults = {
    'ncolumns': 100,
    'nlayers': 48,
    'dt': 2 + 1/12,
    'tmax': 500,
    'dumpfreq': 5,
    'dirname': 'moist_trapped_wave_modified',
    'hydrostatic': False
}


def mountain_nonhydrostatic(
        ncolumns=mountain_nonhydrostatic_defaults['ncolumns'],
        nlayers=mountain_nonhydrostatic_defaults['nlayers'],
        dt=mountain_nonhydrostatic_defaults['dt'],
        tmax=mountain_nonhydrostatic_defaults['tmax'],
        dumpfreq=mountain_nonhydrostatic_defaults['dumpfreq'],
        dirname=mountain_nonhydrostatic_defaults['dirname'],
        hydrostatic=mountain_nonhydrostatic_defaults['hydrostatic']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 80000.   # width of domain in x direction, in m
    domain_height = 15984.   # height of model top, in m
    a = 2500.                # scale width of mountain, in m
    hm = 1.                  # height of mountain, in m
    zh = 5000.               # height at which mesh is no longer distorted, in m
    Tsurf = 283.             # temperature of surface, in K
    initial_wind = 20.0      # initial horizontal wind, in m/s
    sponge_depth = 7992.0   # depth of sponge layer, in m
    g = 9.80665              # acceleration due to gravity, in m/s^2
    cp = 1004.               # specific heat capacity at constant pressure
    mu_dt = 0.15             # parameter for strength of sponge layer, no units
    exner_surf = 1.0         # maximum value of Exner pressure at surface
    max_iterations = 10      # maximum number of hydrostatic balance iterations
    tolerance = 1e-7         # tolerance for hydrostatic balance iteration

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1
    u_eqn_type = 'vector_invariant_form'
    alpha = 0.5

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    # Make normal extruded mesh which will be distorted to describe the mountain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    ext_mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=domain_height/nlayers
    )
    Vc = VectorFunctionSpace(ext_mesh, "DG", 2)

    # Describe the mountain
    xc = domain_width/2.
    x, z = SpatialCoordinate(ext_mesh)
    zs = hm * a**2 / ((x - xc)**2 + a**2)
    xexpr = as_vector(
        [x, conditional(z < zh, z + cos(0.5 * pi * z / zh)**6 * zs, z)]
    )

    # Make new mesh
    new_coords = Function(Vc).interpolate(xexpr)
    mesh = Mesh(new_coords)
    mesh._base_mesh = base_mesh  # Force new mesh to inherit original base mesh
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    parameters = CompressibleParameters(mesh, g=g, cp=cp)
    tracers = [WaterVapour() , CloudWater()]
    sponge = SpongeLayerParameters(
        mesh, H=domain_height, z_level=domain_height-sponge_depth, mubar=mu_dt/dt
    )
    if hydrostatic:
        eqns = HydrostaticCompressibleEulerEquations(
            domain, parameters, sponge_options=sponge,
            u_transport_option=u_eqn_type
        )
    else:
        eqns = CompressibleEulerEquations(
            domain, parameters, active_tracers=tracers, sponge_options=sponge,
            u_transport_option=u_eqn_type
        )

    # I/O
    # Adjust default directory name
    if hydrostatic and dirname == mountain_nonhydrostatic_defaults['dirname']:
        dirname = f'hyd_switch_{dirname}'

    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True
    )
    diagnostic_fields = [
        Exner(parameters), ZComponent('u'), Perturbation('theta')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    theta_opts = SUPGOptions()
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "rho"),
        SSPRK3(domain, "theta", options=theta_opts),
        SSPRK3(domain, "water_vapour", options=theta_opts),
        SSPRK3(domain, "cloud_water", options=theta_opts)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho"),
        DGUpwind(eqns, "theta", ibp=theta_opts.ibp),
        DGUpwind(eqns, "water_vapour", ibp=theta_opts.ibp),
        DGUpwind(eqns, "cloud_water", ibp=theta_opts.ibp)
    ]

    # Linear solver
    if hydrostatic:
        linear_solver = CompressibleSolver(
            eqns, alpha, solver_parameters=hydrostatic_parameters,
            overwrite_solver_parameters=True
        )
    else:
        linear_solver = CompressibleSolver(eqns, alpha)

    class MaintainRelativeHumidity(PhysicsParametrisation):

        def __init__(self, equation, value):
            label_name = 'maintain_rh'
            super().__init__(equation, label_name)

            W = equation.function_space
            self.rho = Function(W.sub(1))
            self.theta = Function(W.sub(2))
            self.r_v = Function(W.sub(3))
            self.exner = Function(W.sub(1))
            self.T = Function(W.sub(1))
            self.p = Function(W.sub(1))
            self.r_v_new = Function(W.sub(3))

        def evaluate(self, x_in, dt, x_out=None):
            self.rho.assign(x_in.subfunctions(1))
            self.theta.assign(x_in.subfunctions(2))
            self.r_v.assign(x_in.subfunctions(3))
            self.exner.interpolate(td.exner_pressure(self.parameters, self.rho, self.theta))
            self.T.interpolate(td.T(self.parameters, self.theta, self.exner, self.r_v))
            self.p.interpolate(td.p(self.parameters, self.exner))
            self.r_v_new.interpolate(td.r_v(1., T, p))
            r_v = x_in.subfunctions(3)
            print('before: ', r_v.at(1000.0))
            r_v.interpolate(conditional(x < xc-a, max(r_v, r_v_new), r_v))
            print('after: ', r_v.at(1000.0))

    class AdjustValue(TimeDiscretisation):

        def apply(self, x_out, x_in):
            print('hello', len(self.evaluate_source))
            for evaluate in self.evaluate_source:
                evaluate(x_in, self.dt)

    # Physics schemes
    physics_schemes = [
    	(SaturationAdjustment(eqns), ForwardEuler(domain)),
	(MaintainRelativeHumidity(eqns, 1), AdjustValue(domain, 'water_vapour'))
    ]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        physics_schemes=physics_schemes, linear_solver=linear_solver, alpha=alpha
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

    # Constants required for setting initial conditions
    # and reference profiles
    #T = 250
    k = 1
    m = 0.00001
    #l1 = 1*10**-6
    #l2 = 1.5*10**-7
    z1 = 3000
    N1 = 0.018**2
    N2= 0.008**2

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    x, z = SpatialCoordinate(mesh)
    thetab = conditional(z<z1, Tsurf*exp(N1*z/g), Tsurf*exp((z1/g)*(N1-N2))*exp(N2*z/g))
    theta_b = Function(Vt).interpolate(thetab)

    # Relative humidity
    #relhum = conditional(x < (xc - a), conditional(z<z1, -tanh(k*(z-z1))*-tanh(m*(x-(xc-a)))*tanh(m*x), 0), 0)
    relhum = conditional(x < (xc - a), conditional(z<z1, 1, 0), 0)

    relhum_layer = conditional(z<z1, 1, 0)
    rel_hum = Function(Vt).interpolate(relhum_layer)

    # Cloud content
    #cloud = conditional(x < (xc - a), conditional(z<z1, -0.0002*tanh(k*(z-z1))*-tanh(m*(x-(xc-a))), 0), 0)
    cloud = 0
    water_c0.interpolate(cloud)

    # Calculate hydrostatic exner
    exner_dry = Function(Vr)
    exner_moist = Function(Vr)
    rho_dry = Function(Vr)
    rho_moist = Function(Vr)
    theta_moist = Function(Vt)

    # Set up kernels to evaluate global minima and maxima of fields
    min_kernel = MinKernel()
    max_kernel = MaxKernel()

    # First solve hydrostatic balance that gives Exner = 1 at bottom boundary
    # This gives us a guess for the top boundary condition
    bottom_boundary = Constant(exner_surf, domain=mesh)
    logger.info(f'Solving hydrostatic with bottom Exner of {exner_surf}')
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_dry, exner_dry, top=False, exner_boundary=bottom_boundary
    )

    # Solve hydrostatic balance again, but now use minimum value from first
    # solve as the *top* boundary condition for Exner
    top_value = min_kernel.apply(exner_dry)
    top_boundary = Constant(top_value, domain=mesh)
    logger.info(f'Solving hydrostatic with top Exner of {top_value}')
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_dry, exner_dry, top=True, exner_boundary=top_boundary
    )

    max_bottom_value = max_kernel.apply(exner_dry)

    # Now we iterate, adjusting the top boundary condition, until this gives
    # a maximum value of 1.0 at the surface
    lower_top_guess = 0.9*top_value
    upper_top_guess = 1.2*top_value
    for i in range(max_iterations):
        # If max bottom Exner value is equal to desired value, stop iteration
        if abs(max_bottom_value - exner_surf) < tolerance:
            break

        # Make new guess by average of previous guesses
        top_guess = 0.5*(lower_top_guess + upper_top_guess)
        top_boundary.assign(top_guess)

        logger.info(
            f'Solving hydrostatic balance iteration {i}, with top Exner value '
            + f'of {top_guess}'
        )

        compressible_hydrostatic_balance(
            eqns, theta_b, rho_dry, exner_dry, top=True, exner_boundary=top_boundary
        )

        max_bottom_value = max_kernel.apply(exner_dry)

        # Adjust guesses based on new value
        if max_bottom_value < exner_surf:
            lower_top_guess = top_guess
        else:
            upper_top_guess = top_guess

    logger.info(f'Final max bottom Exner value of {max_bottom_value}')

    # Perform a final solve to obtain hydrostatically balanced rho
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_dry, exner_dry, top=True, exner_boundary=top_boundary,
        solve_for_rho=True
    )


    # First solve hydrostatic balance that gives Exner = 1 at bottom boundary
    # This gives us a guess for the top boundary condition
    bottom_boundary = Constant(exner_surf, domain=mesh)
    logger.info(f'Solving hydrostatic with bottom Exner of {exner_surf}')
    unsaturated_hydrostatic_balance(
        eqns, stepper.fields, theta_b, rel_hum, exner_moist, top=False, exner_boundary=bottom_boundary
    )

    # Solve hydrostatic balance again, but now use minimum value from first
    # solve as the *top* boundary condition for Exner
    top_value = min_kernel.apply(exner_moist)
    top_boundary = Constant(top_value, domain=mesh)
    logger.info(f'Solving hydrostatic with top Exner of {top_value}')
    unsaturated_hydrostatic_balance(
        eqns, stepper.fields, theta_b, rel_hum, exner_moist, top=True, exner_boundary=top_boundary
    )

    max_bottom_value = max_kernel.apply(exner_moist)

    # Now we iterate, adjusting the top boundary condition, until this gives
    # a maximum value of 1.0 at the surface
    lower_top_guess = 0.9*top_value
    upper_top_guess = 1.2*top_value
    for i in range(max_iterations):
        # If max bottom Exner value is equal to desired value, stop iteration
        if abs(max_bottom_value - exner_surf) < tolerance:
            break

        # Make new guess by average of previous guesses
        top_guess = 0.5*(lower_top_guess + upper_top_guess)
        top_boundary.assign(top_guess)

        logger.info(
            f'Solving hydrostatic balance iteration {i}, with top Exner value '
            + f'of {top_guess}'
        )

        unsaturated_hydrostatic_balance(
            eqns, stepper.fields, theta_b, rel_hum, exner_moist, top=True, exner_boundary=top_boundary
        )

        max_bottom_value = max_kernel.apply(exner_moist)

        # Adjust guesses based on new value
        if max_bottom_value < exner_surf:
            lower_top_guess = top_guess
        else:
            upper_top_guess = top_guess

    logger.info(f'Final max bottom Exner value of {max_bottom_value}')

    # Perform a final solve to obtain hydrostatically balanced rho
    #unsaturated_hydrostatic_balance(
    #    eqns, stepper.fields, theta_b, rel_hum, exner_moist, top=True, exner_boundary=top_boundary,
    #    solve_for_rho=True
    #)

    cond_exner = conditional(x < (xc - a), exner_moist, exner_dry)
    exner = Function(Vr).interpolate(cond_exner)
    out = File("exner.pvd")
    out.write(exner_dry, exner_moist, exner)

    cond_rho = conditional(x < (xc - a), rho0, rho_dry)
    rho_b = Function(Vr).interpolate(cond_rho)

    cond_theta = conditional(x < (xc - a), theta0, theta_b)
    theta_t = Function(Vt).interpolate(cond_theta)
    
    # Calculate mixing ratio
    pressure = td.p(parameters, exner)
    T = theta_t * ( pressure /parameters.p_0)**parameters.kappa
    #Tout = Function(Vt).interpolate(T)
    #outfile = File('tinit.pvd').write(Tout)

    #mixing_ratio.interpolate(td.r_v(parameters, relhum, T, pressure))
    #water_v0.assign(mixing_ratio)

    water_v0.interpolate(td.r_v(parameters, relhum, T, pressure))

    theta0.assign(theta_t)
    rho0.assign(rho_b)

    u0.project(as_vector([initial_wind, 0.0]), bcs=eqns.bcs['u'])

    stepper.set_reference_profiles([('rho', rho_b), ('theta', theta_t)])

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
        default=mountain_nonhydrostatic_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=mountain_nonhydrostatic_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=mountain_nonhydrostatic_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=mountain_nonhydrostatic_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=mountain_nonhydrostatic_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=mountain_nonhydrostatic_defaults['dirname']
    )
    parser.add_argument(
        '--hydrostatic',
        help=(
            "Whether to use the hydrostatic switch to emulate the "
            + "hydrostatic equations. Otherwise use the full non-hydrostatic"
            + "equations."
        ),
        action="store_true",
        default=mountain_nonhydrostatic_defaults['hydrostatic']
    )
    args, unknown = parser.parse_known_args()

    mountain_nonhydrostatic(**vars(args))
