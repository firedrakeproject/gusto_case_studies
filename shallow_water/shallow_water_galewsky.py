"""
The Galewsky jet test case.
"""

from gusto import *
from firedrake import (SpatialCoordinate, pi, conditional, exp, cos, assemble,
                       dx, Constant, Function, sqrt, atan_2, asin)
import numpy as np
import sys

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
a = 6371220.
H = 10000.

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
mesh = GeneralCubedSphereMesh(a, ncell_1d, degree=2)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)
domain = Domain(mesh, dt, 'RTCF', 1)

# Equation
parameters = ShallowWaterParameters(H=H)
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/a
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr,
                             u_transport_option='vector_advection_form')

# I/O
dirname = "galewsky"
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq, dump_nc=True,
                          dumplist=['D'], log_level='INFO')
diagnostic_fields = [RelativeVorticity()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [SSPRK3(domain, "u"), SSPRK3(domain, "D")]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0_field = stepper.fields("u")
D0_field = stepper.fields("D")

# Parameters
umax = 80.0
g = parameters.g
Omega = parameters.Omega
phi0 = pi/7
phi1 = pi/2 - phi0
phi2 = pi/4
alpha = 1.0/3
beta = 1.0/15
h_hat = 120.0
e_n = np.exp(-4./((phi1-phi0)**2))

# Get spherical coordinates
e_x = as_vector([Constant(1.0), Constant(0.0), Constant(0.0)])
e_y = as_vector([Constant(0.0), Constant(1.0), Constant(0.0)])
e_z = as_vector([Constant(0.0), Constant(0.0), Constant(1.0)])
R = sqrt(x[0]**2 + x[1]**2)  # distance from z axis
r = sqrt(x[0]**2 + x[1]**2 + x[2]**2)  # distance from origin

lon = atan_2(x[1], x[0])
lat = asin(x[2]/r)
e_lon = (x[0] * e_y - x[1] * e_x) / R
lat_VD = Function(D0_field.function_space()).interpolate(lat)

# -------------------------------------------------------------------- #
# Obtain u and D (by integration of analytic expression)
# -------------------------------------------------------------------- #

# Wind -- UFL expression
uexpr = conditional(lat <= phi0, 0.0,
                    conditional(lat >= phi1, 0.0,
                                umax / e_n * exp( 1.0 / ((lat-phi0)*(lat-phi1)) )
                                )
                    )

# Numpy function
def u_func(y):
    return np.where(y <= phi0, 0.0,
                    np.where(y >= phi1, 0.0,
                                umax / e_n * np.exp( 1.0 / ((y-phi0)*(y-phi1)) )
                                )
                    )

# Function for depth field in terms of u function
def h_func(y):
    return a/g*(2*Omega*np.sin(y) + u_func(y)*np.tan(y)/a)*u_func(y)

# Find h from numerical integral
D0_integral = Function(D0_field.function_space())
h_integral = NumericalIntegral(-pi/2, pi/2)
h_integral.tabulate(h_func)
D0_integral.dat.data[:] = h_integral.evaluate_at(lat_VD.dat.data[:])
Dexpr = H - D0_integral

# Obtain fields
u0_field.project(uexpr*e_lon)
D0_field.interpolate(Dexpr)

# Adjust mean value of initial D
C = Function(D0_field.function_space()).assign(Constant(1.0))
area = assemble(C*dx)
Dmean = assemble(D0_field*dx)/area
D0_field -= Dmean
D0_field += Constant(H)

# Background field, store in object for use in diagnostics
Dbar = Function(D0_field.function_space()).assign(D0_field)

#----------------------------------------------------------------------#
# Apply perturbation
#----------------------------------------------------------------------#

h_pert = h_hat*cos(lat)*exp(-(lon/alpha)**2)*exp(-((phi2-lat)/beta)**2)
D0_field.interpolate(Dexpr + h_pert)

stepper.set_reference_profiles([('D', Dbar)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
