"""
An implementation of the Galewsky jet, with the thermal shallow water equations.
The initial conditions are taken from Hartney et al, 2024: ``A compatible finite
element discretisation for moist shallow water equations''.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import (
    SpatialCoordinate, pi, conditional, exp, cos, assemble, dx, Constant,
    Function, sqrt
)
from gusto import (
    Domain, IO, OutputParameters, DGUpwind, xyz_vector_from_lonlatr,
    ShallowWaterParameters, ThermalShallowWaterEquations, SubcyclingOptions,
    lonlatr_from_xyz, GeneralCubedSphereMesh, RelativeVorticity,
    ZonalComponent, MeridionalComponent, RungeKuttaFormulation, SSPRK3,
    SemiImplicitQuasiNewton, ThermalSWSolver, NumericalIntegral
)

import numpy as np

thermal_galewsky_defaults = {
    'ncells_per_edge': 32,     # number of cells per cubed sphere edge
    'dt': 600.0,               # 10 minutes
    'tmax': 6.*24.*60.*60.,    # 6 days
    'dumpfreq': 864,           # final time step with default options
    'dirname': 'thermal_galewsky'
}


def thermal_galewsky(
        ncells_per_edge=thermal_galewsky_defaults['ncells_per_edge'],
        dt=thermal_galewsky_defaults['dt'],
        tmax=thermal_galewsky_defaults['tmax'],
        dumpfreq=thermal_galewsky_defaults['dumpfreq'],
        dirname=thermal_galewsky_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    # Shallow water parameters
    radius = 6371220.       # planetary radius (m)
    mean_depth = 10000.     # reference depth (m)
    g = 9.80616             # acceleration due to gravity (m/s^2)
    umax = 80.0             # amplitude of jet wind speed, in m/s
    db = 1.0                # diff in buoyancy between equator and poles (m/s^2)
    phi0 = pi/7             # lower latitude of initial jet, in rad
    phi1 = pi/2 - phi0      # upper latitude of initial jet, in rad
    phi2 = pi/4             # central latitude of perturbation to jet, in rad
    alpha = 1.0/3           # zonal width parameter of perturbation, in rad
    beta = 1.0/15           # meridional width parameter of perturbation, in rad
    h_hat = 120.0           # strength of perturbation, in m

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
    mesh = GeneralCubedSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, "RTCF", element_order)
    xyz = SpatialCoordinate(mesh)

    # Equation
    parameters = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = parameters.Omega
    fexpr = 2*Omega*xyz[2]/radius
    eqns = ThermalShallowWaterEquations(
        domain, parameters, u_transport_option=u_eqn_type, fexpr=fexpr
    )

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True,
    )
    diagnostic_fields = [
        RelativeVorticity(), ZonalComponent('u'), MeridionalComponent('u')
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

    u0_field = stepper.fields("u")
    D0_field = stepper.fields("D")
    b0_field = stepper.fields("b")

    # Parameters
    g = parameters.g
    e_n = np.exp(-4./((phi1-phi0)**2))

    lon, lat, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    lat_VD = Function(D0_field.function_space()).interpolate(lat)

    # ------------------------------------------------------------------------ #
    # Obtain u, b and D (by integration of analytic expression)
    # ------------------------------------------------------------------------ #

    # Buoyancy
    bexpr = g - db*cos(lat)

    # Wind -- UFL expression
    u_zonal = conditional(
        lat <= phi0, 0.0,
        conditional(
            lat >= phi1, 0.0,
            umax / e_n * exp(1.0 / ((lat - phi0) * (lat - phi1)))
        )
    )
    uexpr = xyz_vector_from_lonlatr(u_zonal, Constant(0.0), Constant(0.0), xyz)

    # Numpy function
    def u_func(y):
        u_array = np.where(
            y <= phi0, 0.0,
            np.where(
                y >= phi1, 0.0,
                umax / e_n * np.exp(1.0 / ((y - phi0) * (y - phi1)))
            )
        )
        return u_array

    # Function for depth field in terms of u function
    def h_func(y):
        h_array = (
            1.0/(g - db*np.cos(y))**0.5 * u_func(y) * radius / g
            * (2*Omega*np.sin(y) + u_func(y) * np.tan(y)/radius)
        )

        return h_array

    # Find h from numerical integral
    D0_integral = Function(D0_field.function_space())
    h_integral = NumericalIntegral(-pi/2, pi/2)
    h_integral.tabulate(h_func)
    D0_integral.dat.data[:] = h_integral.evaluate_at(lat_VD.dat.data[:])
    Dexpr = (mean_depth*sqrt(g - db) - D0_integral)/sqrt(bexpr)

    # Obtain fields
    u0_field.project(uexpr)
    D0_field.interpolate(Dexpr)

    # Adjust mean value of initial D
    C = Function(D0_field.function_space()).assign(Constant(1.0))
    area = assemble(C*dx)
    Dmean = assemble(D0_field*dx)/area
    D0_field -= Dmean
    D0_field += Constant(mean_depth)

    # ------------------------------------------------------------------------ #
    # Apply perturbation
    # ------------------------------------------------------------------------ #

    h_pert = h_hat*cos(lat)*exp(-(lon/alpha)**2)*exp(-((phi2-lat)/beta)**2)
    D0_field.interpolate(Dexpr + h_pert)
    b0_field.interpolate(bexpr)

    # Background field, store in object for use in diagnostics
    Dbar = Function(D0_field.function_space()).assign(D0_field)
    bbar = Function(b0_field.function_space()).interpolate(b0_field)
    stepper.set_reference_profiles([('D', Dbar), ('b', bbar)])

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
        help="The number of cells per edge of cubed sphere",
        type=int,
        default=thermal_galewsky_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=thermal_galewsky_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=thermal_galewsky_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=thermal_galewsky_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=thermal_galewsky_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    thermal_galewsky(**vars(args))
