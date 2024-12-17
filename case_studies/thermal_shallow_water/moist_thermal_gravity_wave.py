"""
A gravity wave on the sphere, solved with the moist thermal shallow water
equations. The initial conditions are saturated and cloudy everywhere.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, pi, sqrt, min_value, cos, Constant, Function, exp, sin
)
from gusto import (
    Domain, IO, OutputParameters, DGUpwind, ShallowWaterParameters,
    ShallowWaterEquations, lonlatr_from_xyz, GeneralIcosahedralSphereMesh,
    RungeKuttaFormulation, SSPRK3, ThermalSWSolver, MeridionalComponent,
    SemiImplicitQuasiNewton, ForwardEuler, WaterVapour, CloudWater,
    xyz_vector_from_lonlatr, SWSaturationAdjustment, ZonalComponent,
    SubcyclingOptions
)

moist_thermal_gw_defaults = {
    'ncells_per_edge': 16,     # number of cells per icosahedron edge
    'dt': 900.0,               # 15 minutes
    'tmax': 5.*24.*60.*60.,    # 5 days
    'dumpfreq': 96,            # dump once per day
    'dirname': 'moist_thermal_gw'
}


def moist_thermal_gw(
        ncells_per_edge=moist_thermal_gw_defaults['ncells_per_edge'],
        dt=moist_thermal_gw_defaults['dt'],
        tmax=moist_thermal_gw_defaults['tmax'],
        dumpfreq=moist_thermal_gw_defaults['dumpfreq'],
        dirname=moist_thermal_gw_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.           # planetary radius (m)
    mean_depth = 5960.          # reference depth (m)
    q0 = 0.0115                 # saturation curve coefficient (kg/kg)
    beta2 = 9.80616*10          # thermal feedback coefficient (m/s^2)
    nu = 1.5                    # dimensionless parameter in saturation curve
    R0 = pi/9.                  # radius of perturbation (rad)
    lamda_c = -pi/2.            # longitudinal centre of perturbation (rad)
    phi_c = pi/6.               # latitudinal centre of perturbation (rad)
    phi_0 = 3.0e4               # scale factor for poleward buoyancy gradient
    epsilon = 1/300             # linear air expansion coeff (1/K)
    u_max = 20.                 # max amplitude of the zonal wind (m/s)

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2)
    degree = 1
    domain = Domain(mesh, dt, "BDM", degree)
    xyz = SpatialCoordinate(mesh)

    # Equation parameters
    parameters = ShallowWaterParameters(H=mean_depth)
    Omega = parameters.Omega
    fexpr = 2*Omega*xyz[2]/radius

    # Equation
    tracers = [WaterVapour(space='DG'), CloudWater(space='DG')]
    eqns = ShallowWaterEquations(
        domain, parameters, fexpr=fexpr, thermal=True, active_tracers=tracers
    )

    # IO
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False,
        dumplist=['b', 'water_vapour', 'cloud_water', 'D']
    )
    diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    transport_methods = [
        DGUpwind(eqns, field_name) for field_name in eqns.field_names
    ]

    linear_solver = ThermalSWSolver(eqns)

    def sat_func(x_in):
        D = x_in.split()[1]
        b = x_in.split()[2]
        q_v = x_in.split()[3]
        b_e = b - beta2*q_v
        sat = q0*mean_depth/D * exp(nu*(1-b_e/g))
        return sat

    # Physics schemes
    sat_adj = SWSaturationAdjustment(
        eqns, sat_func, time_varying_saturation=True,
        parameters=parameters, thermal_feedback=True, beta2=beta2
    )

    physics_schemes = [(sat_adj, ForwardEuler(domain))]

    # ------------------------------------------------------------------------ #
    # Timestepper
    # ------------------------------------------------------------------------ #

    subcycling_opts = SubcyclingOptions(subcycle_by_courant=0.25)
    transported_fields = [
        SSPRK3(domain, "u", subcycling_options=subcycling_opts),
        SSPRK3(
            domain, "D", subcycling_options=subcycling_opts,
            rk_formulation=RungeKuttaFormulation.linear
        ),
        SSPRK3(domain, "b", subcycling_options=subcycling_opts),
        SSPRK3(domain, "water_vapour", subcycling_options=subcycling_opts),
        SSPRK3(domain, "cloud_water", subcycling_options=subcycling_opts)
    ]
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver, physics_schemes=physics_schemes,
        num_outer=2, num_inner=2
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    b0 = stepper.fields("b")
    v0 = stepper.fields("water_vapour")
    c0 = stepper.fields("cloud_water")

    lamda, phi, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    # Velocity -- a solid body rotation
    uexpr = xyz_vector_from_lonlatr(u_max*cos(phi), 0, 0, xyz)

    # Buoyancy -- dependent on latitude
    g = parameters.g
    w = Omega*radius*u_max + (u_max**2)/2
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
    b_guess = parameters.g * (1 - theta)

    # Depth -- in balance with the contribution of a perturbation
    Dexpr = mean_depth - (1/g)*(w + sigma)*((sin(phi))**2)

    # Perturbation
    lsq = (lamda - lamda_c)**2
    thsq = (phi - phi_c)**2
    rsq = min_value(R0**2, lsq+thsq)
    r = sqrt(rsq)
    pert = 2000 * (1 - r/R0)
    Dexpr += pert

    # Actual initial buoyancy is specified through equivalent buoyancy
    q_t = 0.03
    b_init = Function(b0.function_space()).interpolate(b_guess)
    b_e_init = Function(b0.function_space()).interpolate(b_init - beta2*q_t)
    q_v_init = Function(v0.function_space()).interpolate(q_t)

    # Iterate to obtain equivalent buoyancy and saturation water vapour
    n_iterations = 10

    for _ in range(n_iterations):
        q_sat_expr = q0*mean_depth/Dexpr * exp(nu*(1-b_e_init/g))
        dq_sat_dq_v_expr = nu*beta2/g*q_sat_expr
        q_v_init.interpolate(q_v_init - (q_sat_expr - q_v_init)/(dq_sat_dq_v_expr - 1.0))
        b_e_init.interpolate(b_init - beta2*q_v_init)

    # Water vapour set to saturation amount
    vexpr = q0*mean_depth/Dexpr * exp(nu*(1-b_e_init/g))

    # Back out the initial buoyancy using b_e and q_v
    bexpr = b_e_init + beta2*vexpr

    # Cloud is the rest of total liquid that isn't vapour
    cexpr = Constant(q_t) - vexpr

    u0.project(uexpr)
    D0.interpolate(Dexpr)
    b0.interpolate(bexpr)
    v0.interpolate(vexpr)
    c0.interpolate(cexpr)

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
        default=moist_thermal_gw_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=moist_thermal_gw_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=moist_thermal_gw_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=moist_thermal_gw_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=moist_thermal_gw_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    moist_thermal_gw(**vars(args))
