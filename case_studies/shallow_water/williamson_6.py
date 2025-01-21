"""
The Rossby-Haurwitz Wave test case (6) of Williamson et al. 1992.
A non-divergent wave pattern moves
eastwardly maintaining the wave structure. The test case is on the sphere.

The setup implemented here uses the cubed sphere mesh with the degree 1 spaces.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    SpatialCoordinate, Constant,
    Function, sin, cos, FunctionSpace, grad
)
from gusto import (
    Domain, IO, OutputParameters, GeneralCubedSphereMesh, RelativeVorticity,
    lonlatr_from_xyz,
    ShallowWaterEquations, ShallowWaterParameters, SSPRK3, DGUpwind,
    SemiImplicitQuasiNewton, ZonalComponent, MeridionalComponent
)
import numpy as np

williamson_6_defaults = {
    'ncells_per_edge': 32,     # number of cells per cubed sphere panel edge
    'dt': 300.0,               # 5 minutes
    'tmax': 14.*24.*60.*60.,   # 14 days
    'dumpfreq': 288,           # once per day with default options
    'dirname': 'williamson_6'
}


def williamson_6(
        ncells_per_edge=williamson_6_defaults['ncells_per_edge'],
        dt=williamson_6_defaults['dt'],
        tmax=williamson_6_defaults['tmax'],
        dumpfreq=williamson_6_defaults['dumpfreq'],
        dirname=williamson_6_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    radius = 6371220.      # radius of the planet, in m
    K = Constant(7.847e-6) # Frequency parameter, in sec^-1
    w = K                  # Set omega equal to K
    R = 4.                 # Wave number 4
    h0 = 8000.             # mean (and reference) depth, in m

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1
    hdiv_family = 'RTCF'
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = GeneralCubedSphereMesh(radius, ncells_per_edge, degree=2)
    domain = Domain(mesh, dt, hdiv_family, degree)

    # Equation
    xyz = SpatialCoordinate(mesh)
    parameters = ShallowWaterParameters(H=h0)
    Omega = parameters.Omega
    fexpr = 2*Omega*xyz[2]/radius
    eqns = ShallowWaterEquations(
        domain, parameters, fexpr=fexpr, u_transport_option=u_eqn_type
    )

    # I/O and diagnostics
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dumplist=['D']
    )
    diagnostic_fields = [
        RelativeVorticity(), ZonalComponent('u'), MeridionalComponent('u')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [SSPRK3(domain, "u"), SSPRK3(domain, "D")]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, spatial_methods=transport_methods
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0_field = stepper.fields("u")
    D0_field = stepper.fields("D")

    # Parameters
    g = parameters.g
    Omega = parameters.Omega

    lon, lat, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])

    # ------------------------------------------------------------------------ #
    # Obtain u and D
    # ------------------------------------------------------------------------ #

    # Intilising the velocity field from streamfunction
    CG2 = FunctionSpace(mesh, 'CG', 2)
    psi = Function(CG2)
    psiexpr = -radius**2 * w * sin(lat) + radius**2 * K * cos(lat)**R * sin(lat) * cos(R*lon)
    psi.interpolate(psiexpr)
    uexpr = domain.perp(grad(psi))

    # Initilising the depth field
    A = (w / 2) * (2 * Omega + w) * cos(lat)**2 + 0.25 * K**2 * cos(lat)**(2 * R) * ((R + 1) * cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * cos(lat)**(-2))
    B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
    B = B_frac * cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * cos(lat)**2)
    C = (1 / 4) * K**2 * cos(lat)**(2 * R) * ((R + 1)*cos(lat)**2 - (R + 2))
    Dexpr = h0 * g + radius**2 * (A + B*cos(lon*R) + C * cos(2 * R * lon))

    # Obtain fields
    u0_field.project(uexpr)
    D0_field.interpolate(Dexpr / g)

    # Dbar is a background field for diagnostics
    Dbar = Function(D0_field.function_space()).assign(h0)
    stepper.set_reference_profiles([('D', Dbar)])

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
        help="The number of cells per edge of cubed sphere panel",
        type=int,
        default=williamson_6_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=williamson_6_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=williamson_6_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=williamson_6_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=williamson_6_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    williamson_6(**vars(args))
