"""
A demonstration of the shallow water equations over topography, using the
topography that mimics the shape of the Pangea super-continent, and the initial
conditions of test case 5 of Williamson et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

This setup uses the cubed sphere with the degree 1 finite element spaces.

NB: the initial conditions for this test case should be generated from the
`create_pangea_dump.py` utilities script.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from firedrake import SpatialCoordinate, as_vector, CheckpointFile, Function
from gusto import (
    Domain, IO, OutputParameters, Sum, RelativeVorticity, ZonalComponent,
    MeridionalComponent, ShallowWaterEquations, ShallowWaterParameters,
    SSPRK3, DGUpwind, SemiImplicitQuasiNewton

)
import os.path as osp

shallow_water_pangea_defaults = {
    'ncells_per_edge': 24,     # number of cells per cubed sphere panel edge
    'dt': 600.0,               # 10 minutes
    'tmax': 6.*24.*60.*60.,    # 6 days
    'dumpfreq': 144,           # once per day with default options
    'dirname': 'shallow_water_pangea'
}


def shallow_water_pangea(
        ncells_per_edge=shallow_water_pangea_defaults['ncells_per_edge'],
        dt=shallow_water_pangea_defaults['dt'],
        tmax=shallow_water_pangea_defaults['tmax'],
        dumpfreq=shallow_water_pangea_defaults['dumpfreq'],
        dirname=shallow_water_pangea_defaults['dirname']
):

    # ------------------------------------------------------------------------ #
    # Test case parameters
    # ------------------------------------------------------------------------ #

    radius = 6371220.    # radius of the planet, in m
    H = 10000.           # mean (and reference) depth, in m
    u_max = 20.          # maximum amplitude of the initial zonal wind, in m/s

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    degree = 1
    hdiv_family = 'RTCF'
    u_eqn_type = 'vector_advection_form'

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain -- pick up mesh and topography from checkpoint
    chkfile = osp.join(
        osp.abspath(osp.dirname(__file__)),
        f"utilities/pangea_C{ncells_per_edge}_chkpt.h5"
    )
    with CheckpointFile(chkfile, 'r') as chk:
        # Recover all the fields from the checkpoint
        mesh = chk.load_mesh()
        b_field = chk.load_function(mesh, 'topography')

    domain = Domain(mesh, dt, hdiv_family, degree)

    # Equation
    xyz = SpatialCoordinate(mesh)
    parameters = ShallowWaterParameters(H=H)
    Omega = parameters.Omega
    fexpr = 2*Omega*xyz[2]/radius
    bexpr = b_field
    eqns = ShallowWaterEquations(
        domain, parameters, fexpr=fexpr, topog_expr=bexpr,
        u_transport_option=u_eqn_type
    )

    # I/O and diagnostics
    output = OutputParameters(
        dirname=dirname, dumplist=['D', 'topography'], dump_nc=True,
        dumpfreq=dumpfreq
    )
    diagnostic_fields = [
        Sum('D', 'topography'), RelativeVorticity(), MeridionalComponent('u'),
        ZonalComponent('u')
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

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    uexpr = as_vector([-u_max*xyz[1]/radius, u_max*xyz[0]/radius, 0.0])
    g = parameters.g
    Rsq = radius**2
    Dexpr = H - ((radius*Omega*u_max + 0.5*u_max**2)*xyz[2]**2/Rsq)/g - bexpr

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
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
        help="The number of cells per cubed sphere edge",
        type=int,
        default=shallow_water_pangea_defaults['ncells_per_edge']
    )
    parser.add_argument(
        '--dt',
        help="The time step in seconds.",
        type=float,
        default=shallow_water_pangea_defaults['dt']
    )
    parser.add_argument(
        "--tmax",
        help="The end time for the simulation in seconds.",
        type=float,
        default=shallow_water_pangea_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=shallow_water_pangea_defaults['dumpfreq']
    )
    parser.add_argument(
        '--dirname',
        help="The name of the directory to write to.",
        type=str,
        default=shallow_water_pangea_defaults['dirname']
    )
    args, unknown = parser.parse_known_args()

    shallow_water_pangea(**vars(args))
