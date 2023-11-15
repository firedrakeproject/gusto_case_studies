"""
A vortex in a biperiodic domain (using compressible Euler equations with no
gravity).

The mesh is distorted.
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       Constant, pi, cos, Function, sqrt, conditional, as_vector,
                       atan2, drop, keep, name_label, Mesh, exp)
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 0.5
L = 10000.
H = 10000.
nlayers = 32
ncolumns = 32
tmax = 100.
dumpfreq = int(tmax / (10*dt))

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
m = PeriodicIntervalMesh(ncolumns, L)
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, periodic=True)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)

# Describe the deformation
x, y = SpatialCoordinate(ext_mesh)
x_c = L / 2
y_m = 500.0
sigma_m = 500.0
y_s = y_m*exp(-((x-x_c)/sigma_m)**2)
y_expr = y_s + y
new_coords = Function(Vc).interpolate(as_vector([x, y_expr] ))
mesh = Mesh(new_coords)
mesh._base_mesh = m  # Force new mesh to inherit original base mesh
domain = Domain(mesh, dt, "CG", 1)

# Equation
parameters = CompressibleParameters()
eqns = CompressibleEulerEquations(domain, parameters)
eqns.bcs['u'] = []

# Drop gravity term
eqns.residual = eqns.residual.label_map(
    lambda t: t.get(name_label) == "gravity",
    drop, keep)

# I/O
dirname = 'vortex_distortion'
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq)
diagnostic_fields = []
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
theta_opts = EmbeddedDGOptions()
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=theta_opts)]

transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta")]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("DG")

x, y = SpatialCoordinate(mesh)

uc = 100.
vc = 100.

xc = L/2 #+ t*uc
yc = L/2 #+ t*vc

R = 0.4*L #radius of vortex
r = sqrt((x - xc)**2 + (y - yc)**2)/R
phi = atan2((y - yc),(x-xc))

rhoc = 1.
rhob = Function(Vr).interpolate(conditional(r>=1, rhoc,  1- 0.5*(1-r**2)**6))

uth = (1024 * (1.0 - r)**6 * (r)**6)
ux = conditional(r>=1, uc, uc + uth * (-(y-yc)/(r*R)))
uy = conditional(r>=1, vc, vc + uth * (+(x-xc)/(r*R)))
u0.project(as_vector([ux,uy]))


coe = np.zeros((25))
coe[0]  =     1.0 / 24.0
coe[1]  = -   6.0 / 13.0
coe[2]  =    18.0 /  7.0
coe[3]  = - 146.0 / 15.0
coe[4]  =   219.0 / 8.0
coe[5]  = - 966.0 / 17.0
coe[6]  =   731.0 /  9.0
coe[7]  = -1242.0 / 19.0
coe[8]  = -  81.0 / 40.0
coe[9]  =   64.
coe[10] = - 477.0 / 11.0
coe[11] = -1032.0 / 23.0
coe[12] =   737.0 / 8.0
coe[13] = - 204.0 /  5.0
coe[14] = - 510.0 / 13.0
coe[15] =  1564.0 / 27.0
coe[16] = - 153.0 /  8.0
coe[17] = - 450.0 / 29.0
coe[18] =   269.0 / 15.0
coe[19] = - 174.0 / 31.0
coe[20] = -  57.0 / 32.0
coe[21] =    74.0 / 33.0
coe[22] = -  15.0 / 17.0
coe[23] =     6.0 / 35.0
coe[24] =  -  1.0 / 72.0

p = 0
for ip in range(25):
    p += coe[ip] * (r**(12+ip)-1)
mach = 0.341
p = 1 + 1024**2 * mach**2 *conditional(r>=1, 0, p)

R0 = 287.
gamma = 1.4
pref = parameters.p_0

T = p/(rhob*R0)
thetab = Function(Vt).interpolate(T*(pref/p)**0.286)

theta0.assign(thetab)
rho0.assign(rhob)

stepper.set_reference_profiles([('rho', rhob),
                                ('theta', thetab)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
