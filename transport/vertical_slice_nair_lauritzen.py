u"""
This script implements a the test given in the Bendall, Wood, Thuburn & Cotter,
2023:
``A solution to the trilemma of the moist Charney–Phillips staggering'', QJRMS.

The test is a version of the divergent Case 3 of Nair and Laurtizen (2010),
adapted to a vertical slice. This tests a coupled transport equation for
moisture and dry air density.

The mixing ratio obeys an advective transport equation:
∂/∂t (m_X) + (u.∇)m_X = 0

Whereas the dry density obeys the conservative form:
∂/∂t (ρ_d) + ∇.(ρ_d*u) = 0

There are two configurations that can be run:
    - The 'convergence' configuration has an initial condition of a linearly
      varying density field and two Gaussian bumps for the mixing ratio.
    - The 'consistency' configuration has an initial condition of a
      constant mixing ratio and two Gaussian bumps for the density.

The setup here uses a vertical slice with the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, cos, exp, sin, SpatialCoordinate, pi,
    min_value, as_vector
)
from gusto import (
    Domain, ActiveTracer, TracerVariableType, TransportEquationType,
    CoupledTransportEquation, OutputParameters, IO, TracerDensity, SSPRK3,
    DGUpwind, PrescribedTransport
)

vertical_slice_nair_lauritzen_defaults = {
    'configuration': 'convergence',   # 'convergence or 'consistency'
    'ncells_1d': 80,                  # number of points in x and z directions
    'dt': 4.0,
    'tmax': 2000.,
    'dumpfreq': 125,                  # with defaults gives four outputs
    'dirname': 'vertical_slice_nair_lauritzen'
}


def vertical_slice_nair_lauritzen(
        configuration=vertical_slice_nair_lauritzen_defaults['configuration'],
        ncells_1d=vertical_slice_nair_lauritzen_defaults['ncells_1d'],
        dt=vertical_slice_nair_lauritzen_defaults['dt'],
        tmax=vertical_slice_nair_lauritzen_defaults['tmax'],
        dumpfreq=vertical_slice_nair_lauritzen_defaults['dumpfreq'],
        dirname=vertical_slice_nair_lauritzen_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    tau = 2000.      # time period of reversible flow, in s
    Lx = 2000.       # width of domain, in m
    Hz = 2000.       # height of domain, in m
    w_factor = 0.1   # deformational factor, dimensionless

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1
    m_X_space = 'DG'   # 'theta' or 'DG'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    period_mesh = PeriodicIntervalMesh(ncells_1d, Lx)
    mesh = ExtrudedMesh(period_mesh, layers=ncells_1d, layer_height=Hz/ncells_1d)
    domain = Domain(mesh, dt, "CG", degree)
    x, z = SpatialCoordinate(mesh)

    # Define the mixing ratio and density as tracers
    m_X = ActiveTracer(
        name='m_X', space=m_X_space,
        variable_type=TracerVariableType.mixing_ratio,
        transport_eqn=TransportEquationType.advective
    )

    rho_d = ActiveTracer(
        name='rho_d', space='DG',
        variable_type=TracerVariableType.density,
        transport_eqn=TransportEquationType.conservative
    )

    tracers = [m_X, rho_d]

    # Equation
    V = domain.spaces("HDiv")
    eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu=V)

    # I/O
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False
    )

    # Use a tracer density diagnostic to track conservation
    diagnostic_fields = [TracerDensity('m_X', 'rho_d')]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Details of transport
    transport_scheme = SSPRK3(domain)
    transport_methods = [DGUpwind(eqn, "m_X"), DGUpwind(eqn, "rho_d")]

    # Transporting wind ------------------------------------------------------ #
    # Set up the divergent, time-varying, velocity field
    def u_t(t):
        umax = Lx/tau
        xd = x - umax*t

        u = umax - (w_factor*umax*pi*Lx/Hz)*cos(pi*t/tau)*cos(2*pi*xd/Lx)*cos(pi*z/Hz)
        w = 2*pi*w_factor*umax*cos(pi*t/tau)*sin(2*pi*xd/Lx)*sin(pi*z/Hz)

        return as_vector([u, w])

    # Time stepper
    stepper = PrescribedTransport(
        eqn, transport_scheme, io, transport_methods,
        prescribed_transporting_velocity=u_t
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    # Specify locations of the two Gaussians
    xc1 = 5.*Lx/8.
    zc1 = Hz/2.

    xc2 = 3.*Lx/8.
    zc2 = Hz/2.

    def l2_dist(xc, zc):
        return min_value(abs(x - xc), Lx - abs(x - xc))**2 + (z - zc)**2

    lc = 2.*Lx/25.
    m0 = 0.02

    # Set the initial states from the choice of configuration
    if configuration == 'convergence':
        f0 = 0.05

        rho_t = 0.5
        rho_b = 1.

        rho_d_0 = rho_b + z*(rho_t-rho_b)/Hz

        g1 = f0*exp(-l2_dist(xc1, zc1)/lc**2)
        g2 = f0*exp(-l2_dist(xc2, zc2)/lc**2)

        m_X_0 = m0 + g1 + g2

    elif configuration == 'consistency':
        f0 = 0.5
        rho_b = 0.5

        g1 = f0*exp(-l2_dist(xc1, zc1)/lc**2)
        g2 = f0*exp(-l2_dist(xc2, zc2)/lc**2)

        rho_d_0 = rho_b + g1 + g2

        m_X_0 = m0 + 0*x

    else:
        raise ValueError('Specified configuration is not valid')

    # Set fields
    stepper.fields("m_X").interpolate(m_X_0)
    stepper.fields("rho_d").interpolate(rho_d_0)
    u0 = stepper.fields("u")
    u0.project(u_t(0))

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
        '--configuration',
        help="The test configuration to use, 'convergence' or 'consistency'",
        type=str,
        default=vertical_slice_nair_lauritzen_defaults['configuration']
    )
    parser.add_argument(
        '--ncells_1d',
        help="The number of cells in the x and y directions",
        type=int,
        default=vertical_slice_nair_lauritzen_defaults['ncells_1d']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=vertical_slice_nair_lauritzen_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=vertical_slice_nair_lauritzen_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=vertical_slice_nair_lauritzen_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=vertical_slice_nair_lauritzen_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    vertical_slice_nair_lauritzen(**vars(args))
