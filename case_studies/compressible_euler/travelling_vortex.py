"""
The travelling vortex test case of Kadioglu et al. (2008).

This test solves the compressible Euler equations in the absence of gravity,
in a vertical slice which is also periodic in the vertical direction. The vortex
is translated across the domain and should return to its initial condition, so
that there is an analytic solution.

The implementation here includes the option of changing the horizontal and
vertical element orders separately.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from gusto import (
    Domain, CompressibleParameters,
    OutputParameters, IO, SSPRK3, DGUpwind, SemiImplicitQuasiNewton,
    RungeKuttaFormulation, SteadyStateError, Pressure,
    CompressibleEulerEquations, SubcyclingOptions,
    RecoverySpaces, EmbeddedDGOptions
)
from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, Function,
    conditional, sqrt, as_vector, FunctionSpace
)
import numpy as np


travelling_vortex_defaults = {
    'ncolumns': 50,
    'nlayers': 50,
    'dx': 200,
    'dz': 200,
    'dt': 0.1,
    'tmax': 100.0,
    'dumpfreq': 250,
    'dirname': 'travelling_vortex',
    'horder': 1,
    'vorder': 1,
    'direction': 'diagonal'
}


def travelling_vortex(
        ncolumns=travelling_vortex_defaults['ncolumns'],
        nlayers=travelling_vortex_defaults['nlayers'],
        dt=travelling_vortex_defaults['dt'],
        tmax=travelling_vortex_defaults['tmax'],
        dumpfreq=travelling_vortex_defaults['dumpfreq'],
        dirname=travelling_vortex_defaults['dirname'],
        horder=travelling_vortex_defaults['horder'],
        vorder=travelling_vortex_defaults['vorder'],
        direction=travelling_vortex_defaults['direction']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 1.0e4                # width of domain in x direction, in m
    domain_height = 1.0e4               # height of top of domain, in m
    tau = 100.0                         # advection time scale, in s
    wind_speed = domain_width / tau     # magnitude of background wind, in m/s
    vortex_radius = 0.4 * domain_width  # radius of vortex

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    base_mesh = PeriodicIntervalMesh(ncolumns, domain_width)
    mesh = ExtrudedMesh(
        base_mesh, layers=nlayers, layer_height=domain_height/nlayers,
        periodic=True
    )
    domain = Domain(
        mesh, dt, 'CG', horizontal_degree=horder, vertical_degree=vorder
    )

    # Equations
    params = CompressibleParameters(mesh, g=0)
    eqns = CompressibleEulerEquations(
        domain, params, u_transport_option='vector_advection_form'
    )
    eqns.bcs['u'] = []  # For periodic vertical slice need to zero BCs

    # I/O
    if dirname == travelling_vortex_defaults['dirname']:
        dirname += f'_h{horder}_v{vorder}_{direction}'
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dumplist=['rho', 'theta', 'u'],
        dump_nc=True, dump_vtus=False,
    )
    diagnostics = [
        Pressure(eqns), SteadyStateError('theta'),
        SteadyStateError('u'), SteadyStateError('rho')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostics)

    # Transport schemes
    lowest_orders = ((0, 0), (0, 1), (1, 0))
    subcycling_opts = SubcyclingOptions(subcycle_by_courant=0.25)
    order = (horder, vorder)
    if order in lowest_orders:
        opts = RecoverySpaces(domain)
        transported_fields = [
            SSPRK3(
                domain, "u", options=opts.HDiv_options,
                subcycling_options=subcycling_opts
            ),
            SSPRK3(
                domain, "rho", options=opts.DG_options,
                subcycling_options=subcycling_opts,
                rk_formulation=RungeKuttaFormulation.linear
            ),
            SSPRK3(
                domain, "theta", options=opts.theta_options,
                subcycling_options=subcycling_opts
            )
        ]

    else:
        opts = EmbeddedDGOptions()
        transported_fields = [
            SSPRK3(
                domain, "u", options=opts,
                subcycling_options=subcycling_opts
            ),
            SSPRK3(
                domain, "rho", rk_formulation=RungeKuttaFormulation.linear,
                subcycling_options=subcycling_opts
            ),
            SSPRK3(
                domain, "theta", options=opts,
                subcycling_options=subcycling_opts
            )
        ]

    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho", advective_then_flux=True),
        DGUpwind(eqns, "theta")
    ]

    # Timestepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods, predictor='rho',
        tau_values={'rho': 1.0, 'theta': 1.0}
    )

    # ------------------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------------------

    x, y = SpatialCoordinate(mesh)

    u0 = stepper.fields('u')
    rho0 = stepper.fields('rho')
    theta0 = stepper.fields('theta')

    # spaces
    Vt = theta0.function_space()
    Vr = rho0.function_space()
    CG3 = FunctionSpace(mesh, "CG", 3)

    # Vortex position
    xc = domain_width / 2
    yc = domain_height / 2
    r = sqrt((x - xc)**2 + (y - yc)**2) / vortex_radius

    # rho expression
    rhoc = 1
    rhob = Function(Vr).interpolate(
        conditional(r >= 1, rhoc, 1.0 - 0.5*(1.0 - r**2)**6)
    )
    rho0.interpolate(rhob)

    # u expression
    vc = wind_speed if direction in ['diagonal', 'vertical'] else 0
    uc = wind_speed if direction in ['diagonal', 'horizontal'] else 0

    u_cond = (1024 * (1.0 - r)**6 * r**6)
    ux_expr = conditional(
        r >= 1, uc, uc + u_cond*(-(y - yc)/(r*vortex_radius + 0.0000001))
    )
    uy_expr = conditional(
        r >= 1, vc, vc + u_cond*((x - xc)/(r*vortex_radius + 0.0000001))
    )
    u0.project(as_vector([ux_expr, uy_expr]))

    # Pressure
    coe = np.zeros((25))
    coe[0] = 1.0 / 24.0
    coe[1] = -6.0 / 13.0
    coe[2] = 18.0 / 7.0
    coe[3] = -146.0 / 15.0
    coe[4] = 219.0 / 8.0
    coe[5] = -966.0 / 17.0
    coe[6] = 731.0 / 9.0
    coe[7] = -1242.0 / 19.0
    coe[8] = -81.0 / 40.0
    coe[9] = 64.
    coe[10] = -477.0 / 11.0
    coe[11] = -1032.0 / 23.0
    coe[12] = 737.0 / 8.0
    coe[13] = -204.0 / 5.0
    coe[14] = -510.0 / 13.0
    coe[15] = 1564.0 / 27.0
    coe[16] = -153.0 / 8.0
    coe[17] = -450.0 / 29.0
    coe[18] = 269.0 / 15.0
    coe[19] = -174.0 / 31.0
    coe[20] = -57.0 / 32.0
    coe[21] = 74.0 / 33.0
    coe[22] = -15.0 / 17.0
    coe[23] = 6.0 / 35.0
    coe[24] = -1.0 / 72.0

    p0 = 0
    for i in range(25):
        p0 += coe[i] * (r**(12+i)-1)
    mach = 0.341
    p = 1 + 1024**2 * mach**2 * conditional(r >= 1, 0, p0)

    # Potential temperature
    R0 = 287.
    pref = params.p_0
    T = p / (rhob*R0)

    theta_cg3 = Function(CG3).interpolate(T*(pref / p)**0.286)
    theta_exp = Function(Vt).interpolate(T*(pref / p)**0.286)
    thetab = Function(Vt).project(theta_cg3)
    theta0.interpolate(theta_exp)

    # assign reference profiles
    stepper.set_reference_profiles([('rho', rhob), ('theta', thetab)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)


# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=travelling_vortex_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=travelling_vortex_defaults['nlayers']
    )
    parser.add_argument(
        "--dt",
        help="The timestep",
        type=float,
        default=travelling_vortex_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=travelling_vortex_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=travelling_vortex_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=travelling_vortex_defaults['dirname']
    )
    parser.add_argument(
        '--horder',
        help='The horizontal polynomial order of the element',
        type=int,
        default=travelling_vortex_defaults['horder']
    )
    parser.add_argument(
        '--vorder',
        help='The vertical polynomial order of the element',
        type=int,
        default=travelling_vortex_defaults['vorder']
    )
    parser.add_argument(
        '--direction',
        help='The direction of the advecting winds',
        type=str,
        default=travelling_vortex_defaults['direction']
    )

    args, unknown = parser.parse_known_args()
    travelling_vortex(**vars(args))
