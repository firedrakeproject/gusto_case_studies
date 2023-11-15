"""
A case with a mountain and quadratic extrusion that won't get through a single time step.
"""

from gusto import *
from firedrake import (as_vector, VectorFunctionSpace, grad, pi,
                       PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, Function, conditional, Mesh)
from gusto import thermodynamics
import sys

# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #

quadratic_extrusion = False

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 300.0             # Time step (s)
L = 100000.            # Domain length (m)
H = 30000.             # Domain height (m)
z_m = 1000.             # Mountain height (m)
sigma_m = 5000.        # Width parameter for mountain (m)
x_c = L / 2            # Centre of mountain (m)
z_h = 30000.           # Height above which mesh is not distorted (m)
T_0 = 290.             # Temperature at sea level (K)
u_shear = 1.           # Max velocity of shear part (recommended is 1)

if '--running-tests' in sys.argv:
    tmax = dt
    dumpfreq = 1
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
else:
    tmax = 5*24*60*60
    dumpfreq = int(tmax / (5*dt))
    nlayers = 30  # horizontal layers
    columns = 100  # number of columns

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
# Make an normal extruded mesh which will be distorted to describe the mountain
m = PeriodicIntervalMesh(columns, L)
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)

# Describe the mountain
x, z = SpatialCoordinate(ext_mesh)
z_s = z_m*exp(-((x-x_c)/sigma_m)**2)
if quadratic_extrusion:
    # z_new = z_s @ z = 0
    # z_new = z_b**2/H = z_h @ z = z_b => z_b = sqrt(H*z_h)
    z_b = sqrt(H*z_h)
    z_bottom = z_s + (z_b - z_s)*(z / z_b)**2
    z_top = z**2/H
    xz_expr = as_vector([x, conditional(z < z_b, z_bottom, z_top)])
else:
    # Linear extrusion
    z_bottom = z_s + (z_h - z_s)/z_h*z
    xz_expr = as_vector([x, conditional(z < z_h, z_bottom, z)])

# Make new mesh
new_coords = Function(Vc).interpolate(xz_expr)
mesh = Mesh(new_coords)
mesh._base_mesh = m  # Force new mesh to inherit original base mesh
domain = Domain(mesh, dt, "CG", 1)

# spaces
Vu = domain.spaces("HDiv")
Vt = domain.spaces("theta")
Vr = domain.spaces("L2")
CG2 = FunctionSpace(mesh, "CG", 2, name="CG2")

# Equation
parameters = CompressibleParameters()
eqns = CompressibleEulerEquations(domain, parameters)

# I/O
dirname = 'big_mountain_quadratic'
output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          dumplist=['u', 'theta', 'rho'])
diagnostic_fields = []
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
theta_opts = SUPGOptions()
transported_fields = [TrapeziumRule(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=theta_opts)]
transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta", ibp=theta_opts.ibp)]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

x, z = SpatialCoordinate(mesh)

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# Temporary fields
rho_bar = Function(Vr)
theta_bar = Function(Vt)

# Thermodynamic constants required for setting initial conditions
# and reference profiles
g = parameters.g
R_d = parameters.R_d
p_0 = parameters.p_0

# ---------------------------------------------------------------------------- #
# Temperature and pressure profiles
T_expr = Constant(T_0)
p_expr = p_0*exp(-g*z/(R_d*T_expr))
theta_expr = thermodynamics.theta(parameters, T_expr, p_expr)
rho_expr = p_expr / (R_d * T_expr)

theta_bar.interpolate(theta_expr)
rho_bar.interpolate(rho_expr)

theta0.assign(theta_bar)
rho0.assign(rho_bar)

stepper.set_reference_profiles([('rho', rho_bar),
                                ('theta', theta_bar)])

# ---------------------------------------------------------------------------- #
# Wind profile

# Uniform shear above H/2
shear_expr = conditional(z > H / 2.0,
                         as_vector([u_shear*(z - H / 2.0)/H, Constant(0.0)]),
                         as_vector([Constant(0.0), Constant(0.0)]))

u0.project(shear_expr)

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
