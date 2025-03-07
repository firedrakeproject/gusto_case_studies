"""
The dry rising bubble test case of Robert, 1993:
``Bubble convection experiments with a semi-implicit formulation of the Euler
equations'', JAS.

The test simulates a rising thermal at high resolution.

This setup uses a vertical slice with the order 1 finite elements. The potential
temperature is transported using the embedded DG technique.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, Constant, pi, cos,
    Function, sqrt, conditional, as_vector
)
from gusto import (
    Domain, CompressibleParameters, CompressibleSolver,
    CompressibleEulerEquations, OutputParameters, IO, SSPRK3, TrapeziumRule,
    DGUpwind, SemiImplicitQuasiNewton, compressible_hydrostatic_balance,
    Perturbation, EmbeddedDGOptions
)

robert_bubble_defaults = {
    'ncolumns': 100,
    'nlayers': 100,
    'dt': 1.0,
    'tmax': 600.,
    'dumpfreq': 200,
    'dirname': 'robert_bubble'
}


def robert_bubble(
        ncolumns=robert_bubble_defaults['ncolumns'],
        nlayers=robert_bubble_defaults['nlayers'],
        dt=robert_bubble_defaults['dt'],
        tmax=robert_bubble_defaults['tmax'],
        dumpfreq=robert_bubble_defaults['dumpfreq'],
        dirname=robert_bubble_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 1000.   # width of domain in x direction, in m
    domain_height = 1000.  # height of top of domain, in m
    xc = 500.              # initial x coordinate of centre of the bubble, in m
    zc = 350.              # initial z coordinate of centre of the bubble, in m
    rc = 250.              # initial radius of bubble, in m
    T_pert = 0.25          # temperature perturbation of bubble, in K

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
    domain = Domain(mesh, dt, "CG", element_order)

    # Equation
    parameters = CompressibleParameters(mesh)
    eqns = CompressibleEulerEquations(
        domain, parameters, u_transport_option=u_eqn_type
    )

    # I/O
    dirname = 'robert_bubble'
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dump_vtus=False, dump_nc=True
    )
    diagnostic_fields = [Perturbation('theta')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    theta_opts = EmbeddedDGOptions()
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "rho"),
        SSPRK3(domain, "theta", options=theta_opts)
    ]

    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho"),
        DGUpwind(eqns, "theta")
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver, num_outer=4, num_inner=1
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b, solve_for_rho=True)

    x = SpatialCoordinate(mesh)
    r = sqrt((x[0] - xc)**2 + (x[1] - zc)**2)
    theta_pert = conditional(r > rc, 0., T_pert*(1. + cos(pi * r / rc)))

    theta0.interpolate(theta_b + theta_pert)
    rho0.interpolate(rho_b)
    u0.project(as_vector(
        [Constant(0.0, domain=mesh), Constant(0.0, domain=mesh)]
    ))

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
        '--ncolumns',
        help="The number of columns in the vertical slice mesh.",
        type=int,
        default=robert_bubble_defaults['ncolumns']
    )
    parser.add_argument(
        '--nlayers',
        help="The number of layers for the mesh.",
        type=int,
        default=robert_bubble_defaults['nlayers']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=robert_bubble_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=robert_bubble_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=robert_bubble_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=robert_bubble_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    robert_bubble(**vars(args))
