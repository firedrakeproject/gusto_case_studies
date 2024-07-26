"""
The 1 metre high mountain test case from Melvin et al, 2010:
``An inherently mass-conserving iterative semi-implicit semi-Lagrangian
discretization of the non-hydrostatic vertical-slice equations.'', QJRMS.

This test describes a wave over a mountain in a non-hydrostatic atmosphere.

The setup used here uses the order 1 finite elements.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import (
    as_vector, VectorFunctionSpace, PeriodicIntervalMesh, ExtrudedMesh,
    SpatialCoordinate, exp, pi, cos, Function, conditional, Mesh, op2
)
from gusto import (
    Domain, CompressibleParameters, CompressibleSolver,
    CompressibleEulerEquations, OutputParameters, IO, SSPRK3,
    DGUpwind, SemiImplicitQuasiNewton, compressible_hydrostatic_balance,
    SpongeLayerParameters, CourantNumber, ZComponent, Perturbation,
    SUPGOptions, TrapeziumRule, remove_initial_w
)

mountain_nonhydrostatic_defaults = {
    'ncolumns': 180,
    'nlayers': 70,
    'dt': 5.0,
    'tmax': 9000.,
    'dumpfreq': 450,
    'dirname': 'mountain_nonhydrostatic'
}


def mountain_nonhydrostatic(
        ncolumns=mountain_nonhydrostatic_defaults['ncolumns'],
        nlayers=mountain_nonhydrostatic_defaults['nlayers'],
        dt=mountain_nonhydrostatic_defaults['dt'],
        tmax=mountain_nonhydrostatic_defaults['tmax'],
        dumpfreq=mountain_nonhydrostatic_defaults['dumpfreq'],
        dirname=mountain_nonhydrostatic_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    domain_width = 144000.   # width of domain in x direction, in m
    domain_height = 35000.   # height of model top, in m
    a = 1000.                # scale width of mountain, in m
    hm = 1.                  # height of mountain, in m
    zh = 5000.               # height at which mesh is no longer distorted, in m
    Tsurf = 300.             # temperature of surface, in K
    initial_wind = 10.0      # initial horizontal wind, in m/s
    sponge_depth = 10000.0   # depth of sponge layer, in m
    g = 9.80665              # acceleration due to gravity, in m/s^2
    cp = 1004.               # specific heat capacity at constant pressure
    sponge_mu = 0.15         # parameter for strength of sponge layer, in J/kg/K

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #
    element_order = 1
    u_eqn_type = 'vector_invariant_form'

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
    parameters = CompressibleParameters(g=g, cp=cp)
    sponge = SpongeLayerParameters(
        H=domain_height, z_level=domain_height-sponge_depth, mubar=sponge_mu/dt
    )
    eqns = CompressibleEulerEquations(
        domain, parameters, sponge=sponge, u_transport_form=u_eqn_type
    )

    # I/O
    dirname = 'nonhydrostatic_mountain'
    output = OutputParameters(
        dirname=dirname, dumpfreq=dumpfreq, dumplist=['u']
    )
    diagnostic_fields = [
        CourantNumber(), ZComponent('u'), Perturbation('theta'),
        Perturbation('rho')
    ]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    theta_opts = SUPGOptions()
    transported_fields = [
        TrapeziumRule(domain, "u"),
        SSPRK3(domain, "rho"),
        SSPRK3(domain, "theta", options=theta_opts)
    ]
    transport_methods = [
        DGUpwind(eqns, "u"),
        DGUpwind(eqns, "rho"),
        DGUpwind(eqns, "theta", ibp=theta_opts.ibp)
    ]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(
        eqns, io, transported_fields, transport_methods,
        linear_solver=linear_solver
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

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    N = parameters.N

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    x, z = SpatialCoordinate(mesh)
    thetab = Tsurf*exp(N**2*z/g)
    theta_b = Function(Vt).interpolate(thetab)

    # Calculate hydrostatic exner
    exner = Function(Vr)
    rho_b = Function(Vr)

    exner_params = {'ksp_type': 'gmres',
                    'ksp_monitor_true_residual': None,
                    'pc_type': 'python',
                    'mat_type': 'matfree',
                    'pc_python_type': 'gusto.VerticalHybridizationPC',
                    # Vertical trace system is only coupled vertically in columns
                    # block ILU is a direct solver!
                    'vert_hybridization': {'ksp_type': 'preonly',
                                           'pc_type': 'bjacobi',
                                           'sub_pc_type': 'ilu'}}

    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=True, exner_boundary=0.5,
        params=exner_params
    )

    def minimum(f):
        fmin = op2.Global(1, [1000], dtype=float)
        op2.par_loop(op2.Kernel("""
    static void minify(double *a, double *b) {
        a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
    }
    """, "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
        return fmin.data[0]

    p0 = minimum(exner)
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=True, params=exner_params
    )
    p1 = minimum(exner)
    alpha = 2.*(p1 - p0)
    beta = p1 - alpha
    exner_top = (1. - beta) / alpha
    compressible_hydrostatic_balance(
        eqns, theta_b, rho_b, exner, top=True, exner_boundary=exner_top,
        solve_for_rho=True, params=exner_params
    )

    theta0.assign(theta_b)
    rho0.assign(rho_b)
    u0.project(as_vector([initial_wind, 0.0]))
    remove_initial_w(u0)

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
    args, unknown = parser.parse_known_args()

    mountain_nonhydrostatic(**vars(args))
