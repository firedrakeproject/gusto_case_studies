"""
An example experiment for volcanic ash dispersion.
"""

from gusto import *
from firedrake import (RectangleMesh, exp, SpatialCoordinate,
                       Constant, sin, cos, sqrt, grad)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 60.                       # 1 mins
tmax = 5*24*60*60              # End time
Lx = 1e6                       # Domain length in x direction
Ly = 1e6                       # Domain length in y direction
nx = 200                       # Number of cells in x direction
ny = 200                       # Number of cells in y direction
dumpfreq = int(tmax / (20*dt)) # Output dump frequency
tau = 2.0*24*60*60             # Half life of source
centre_x = 3 * Lx / 8.0        # x coordinate for volcano
centre_y = 2 * Ly / 3.0        # y coordinate for volcano
width = Lx / 50.0              # width of volcano
umax = 12.0                    # Representative wind value
twind = 5*24*60*60             # Time scale for wind components
omega22 = 3.0 / twind          # Frequency for sin(2*pi*x/Lx)*sin(2*pi*y/Ly)
omega21 = 0.9 / twind          # Frequency for sin(2*pi*x/Lx)*sin(pi*y/Ly)
omega12 = 0.6 / twind          # Frequency for sin(pi*x/Lx)*sin(2*pi*y/Ly)
omega44 = 0.1 / twind          # Frequency for sin(4*pi*x/Lx)*sin(4*pi*y/Ly)

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
    nx = 20
    ny = 20

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
domain = Domain(mesh, dt, "RTCF", 1)
x, y = SpatialCoordinate(mesh)

# Equation
V = domain.spaces('DG')
eqn = AdvectionEquation(domain, V, "ash")

# I/O
dirname = 'volcanic_ash'
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist=['ash'],
                          dump_nc=True,
                          dump_vtus=False)
diagnostic_fields = [CourantNumber(), VelocityX(), VelocityY()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transport_method = [DGUpwind(eqn, "ash")]

# Physics scheme ------------------------------------------------------------- #
# Source is a Lorentzian centred on a point
dist_x = x - centre_x
dist_y = y - centre_y
dist = sqrt(dist_x**2 + dist_y**2)
# Lorentzian function
basic_expression = -width / (dist**2 + width**2)

def time_varying_expression(t):
    return 2*basic_expression*exp(-t/tau)

physics_parametrisations = [SourceSink(eqn, 'ash', time_varying_expression,
                                       time_varying=True)]

# Transporting wind ---------------------------------------------------------- #
def transporting_wind(t):
    # Divergence-free wind. A series of sines/cosines with different time factors
    psi_expr = (0.25*Lx/pi)*umax*(
        sin(pi*x/Lx)*sin(pi*y/Ly)
        + 0.15*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*(1.0 + cos(2*pi*omega22*t))
        + 0.25*sin(2*pi*x/Lx)*sin(pi*y/Ly)*sin(2*pi*omega21*(t-0.7*twind))
        + 0.17*sin(pi*x/Lx)*sin(2*pi*y/Ly)*cos(2*pi*omega12*(t+0.2*twind))
        + 0.12*sin(4*pi*x/Lx)*sin(4*pi*y/Ly)*(1.0 + sin(2*pi*omega44*(t-0.83*twind)))
    )

    return domain.perp(grad(psi_expr))

# Time stepper
stepper = PrescribedTransport(eqn, SSPRK3(domain, limiter=DG1Limiter(V)),
                              io, transport_method,
                              physics_parametrisations=physics_parametrisations,
                              prescribed_transporting_velocity=transporting_wind)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

ash0 = stepper.fields("ash")
# Start with some ash over the volcano
ash0.interpolate(Constant(0.0)*-basic_expression)

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

num_steps = int(tmax / dt)
logger.info(f'Beginning run to do {num_steps} steps')
stepper.run(t=0, tmax=tmax)
