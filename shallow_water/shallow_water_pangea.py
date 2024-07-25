"""
The Williamson 5 shallow-water test case (flow over topography), solved with a
discretisation of the non-linear shallow-water equations.

This uses an icosahedral mesh of the sphere, and runs a series of resolutions.
"""

from gusto import *
from firedrake import (SpatialCoordinate, as_vector, CheckpointFile)
import sys
from os.path import join, abspath, dirname

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    dt = 3000.
    ncell_1d = 4
    tmax = dt
    ndumps = 1
else:
    # setup resolution and timestepping parameters for convergence test
    dt = 300.0
    ncell_1d = 48
    tmax = 6*day
    ndumps = 12

# setup shallow water parameters
R = 6371220.
H = 10000.

# ------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #

# Domain -- pick up mesh and topography from checkpoint
chkfile = join(abspath(dirname(__file__)), "pangea_C48_chkpt.h5")
with CheckpointFile(chkfile, 'r') as chk:
    # Recover all the fields from the checkpoint
    mesh = chk.load_mesh()
    b_field = chk.load_function(mesh, 'topography')

x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)
domain = Domain(mesh, dt, 'RTCF', 1)

# Equation
parameters = ShallowWaterParameters(H=H)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
bexpr = b_field
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr,
                             u_transport_option='vector_advection_form')

# I/O
dirname = "shallow_water_pangea"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname, dumplist=['D','topography'],
                          dump_nc=True, dumpfreq=dumpfreq)
diagnostic_fields = [Sum('D', 'topography'), RelativeVorticity(),
                     MeridionalComponent('u'), ZonalComponent('u')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [SSPRK3(domain, "u"), SSPRK3(domain, "D")]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  spatial_methods=transport_methods)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)
