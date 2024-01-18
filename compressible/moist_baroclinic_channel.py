from gusto import *
from gusto import thermodynamics
from firedrake import (PeriodicRectangleMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, sin, pi, sqrt,
                       ln, exp, Constant, Function, DirichletBC, as_vector,
                       FunctionSpace, BrokenElement, VectorFunctionSpace,
                       errornorm, norm, cross, grad)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
import sys

from gusto.diagnostics import CompressibleRelativeVorticity, Vorticity

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

days = 12 # suggested is 15
dt = 300.0
Lx = 4.0e7  # length
Ly = 6.0e6  # width
H = 3.0e4  # height
degree = 1
omega = Constant(7.292e-5)
phi0 = Constant(pi/4)

if '--running-tests' in sys.argv:
    tmax = 5*dt
    deltax = 2.0e6
    deltay = 1.0e6
    deltaz = 6.0e3
    dumpfreq = 5
else:
    tmax = days * 24 * 60 * 60
    deltax = 2.5e5
    deltay = deltax
    deltaz = 1.5e3
    dumpfreq = int(tmax / (3 * days * dt))

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain
nlayers = int(H/deltaz)
ncolumnsx = int(Lx/deltax)
ncolumnsy = int(Ly/deltay)
m = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly, "x", quadrilateral=True)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
domain = Domain(mesh, dt, "RTCF", degree)
x,y,z = SpatialCoordinate(mesh)

# Equation
params = CompressibleParameters()
tracers = [WaterVapour(), CloudWater()]
coriolis = 2*omega*sin(phi0)*domain.k
eqns = CompressibleEulerEquations(domain, params, active_tracers=tracers,
                                  Omega=coriolis/2, no_normal_flow_bc_ids=[1, 2])

# I/O
dirname = 'moist_baroclinic_channel'
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq, dump_nc=True,
                          dumplist=['cloud_water'])
diagnostic_fields = [Perturbation('theta'), CompressibleRelativeVorticity()]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport schemes
transported_fields = [SSPRK3(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta"),
                      SSPRK3(domain, "water_vapour"),
                      SSPRK3(domain, "cloud_water")]

# Linear solver
linear_solver = CompressibleSolver(eqns)

# Physics schemes
physics_schemes = [(SaturationAdjustment(eqns), ForwardEuler(domain))]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  linear_solver=linear_solver,
                                  physics_schemes=physics_schemes)

# ---------------------------------------------------------------------------- #
# Initial conditions
# ---------------------------------------------------------------------------- #

# Physical parameters
a = Constant(6.371229e6)  # radius of earth
b = Constant(2)  # vertical width parameter
beta0 = 2 * omega * cos(phi0) / a
T0 = Constant(288)
Ts = Constant(260)
u0 = Constant(35)
Gamma = Constant(0.005)
Rd = params.R_d
Rv = params.R_v
f0 = 2 * omega * sin(phi0)
y0 = Constant(Ly / 2)
g = params.g
p0 = Constant(100000.)
beta0 = Constant(0)
eta_w = Constant(0.3)
deltay_w = Constant(3.2e6)
q0 = Constant(0.016)
cp = params.cp

# Initial conditions
u = stepper.fields("u")
rho = stepper.fields("rho")
theta = stepper.fields("theta")
water_v = stepper.fields("water_vapour")
water_c = stepper.fields("cloud_water")

# spaces
Vu = u.function_space()
Vt = theta.function_space()
Vr = rho.function_space()

# set up background state expressions
eta = Function(Vt).interpolate(Constant(1e-7))
Phi = Function(Vt).interpolate(g * z)
q = Function(Vt)
T = Function(Vt)
Phi_prime = u0 / 2 * ((f0 - beta0 * y0) *(y - (Ly/2) - (Ly/(2*pi))*sin(2*pi*y/Ly))
                       + beta0/2*(y**2 - (Ly*y/pi)*sin(2*pi*y/Ly)
                                  - (Ly**2/(2*pi**2))*cos(2*pi*y/Ly) - (Ly**2/3) - (Ly**2/(2*pi**2))))
Phi_expr = (T0 * g / Gamma * (1 - eta ** (Rd * Gamma / g))
            + Phi_prime * ln(eta) * exp(-(ln(eta) / b) ** 2))

Tv_expr = T0 * eta ** (Rd * Gamma / g) + Phi_prime / Rd * ((2/b**2) * (ln(eta)) ** 2 - 1) * exp(-(ln(eta)/b)**2)
u_expr = as_vector([-u0 * (sin(pi*y/Ly))**2 * ln(eta) * eta ** (-ln(eta) / b ** 2), 0.0, 0.0])
q_bar = q0 / 2 * conditional(eta > eta_w, (1 + cos(pi * (1 - eta) / (1 - eta_w))), 0.0)
q_expr = q_bar * exp(-(y / deltay_w) ** 4)
r_expr = q / (1 - q)
T_expr = Tv_expr / (1 + q * (Rv / Rd - 1))

# do Newton method to obtain eta
eta_new = Function(Vt)
F = -Phi + Phi_expr
dF = -Rd * Tv_expr / eta
max_iterations = 40
tolerance = 1e-10
for i in range(max_iterations):
    eta_new.interpolate(eta - F/dF)
    if errornorm(eta_new, eta) / norm(eta) < tolerance:
        eta.assign(eta_new)
        break
    eta.assign(eta_new)

# make mean u and theta
u.project(u_expr)
q.interpolate(q_expr)
water_v.interpolate(r_expr)
T.interpolate(T_expr)
theta.interpolate(thermodynamics.theta(params, T_expr, p0 * eta) * (1 + water_v * Rv / Rd))
Phi_test = Function(Vt).interpolate(Phi_expr)
print("Error in setting up p:", errornorm(Phi_test, Phi) / norm(Phi))

# Calculate hydrostatic fields
compressible_hydrostatic_balance(eqns, theta, rho, mr_t=water_v, solve_for_rho=True)

# make mean fields
rho_b = Function(Vr).assign(rho)
u_b = stepper.fields("ubar", space=Vu, dump=False).project(u)
theta_b = Function(Vt).assign(theta)
water_vb = Function(Vt).assign(water_v)

# define perturbation
xc = 2.0e6
yc = 2.5e6
Lp = 6.0e5
up = Constant(1.0)
r = sqrt((x - xc) ** 2 + (y - yc) ** 2)
u_pert = Function(Vu).project(as_vector([up * exp(-(r/Lp)**2), 0.0, 0.0]))

# define initial u
u.assign(u_b + u_pert)

# initialise fields
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b),
                                ('water_vapour', water_vb)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)