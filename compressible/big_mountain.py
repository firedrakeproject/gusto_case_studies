"""
A case with a 1m high mountain that won't get through a single time step.
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
u_rot = 0.00           # Max velocity of rotational flow (recommended is 0.05)
l_a = 0.3              # Dimensionless factor for vortical area in rotational flow
l_b = 0.4              # Dimensionless factor for smoothing area in rotational flow

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
sponge = SpongeLayerParameters(H=H, z_level=H-10000, mubar=1.0/dt)
eqns = CompressibleEulerEquations(domain, parameters, sponge=sponge)

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

x, z = SpatialCoordinate(mesh)
T_surf = T_0 + 2.0*sin(2*x*pi/L)
bl_parameters = BoundaryLayerParameters(height_surface_layer=1000.)
physics_schemes = [(WindDrag(eqns, bl_parameters), BackwardEuler(domain)),
                   (SurfaceFluxes(eqns, T_surf, parameters=bl_parameters), BackwardEuler(domain))]

# Time stepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver,
                                  fast_physics_schemes=physics_schemes,
                                  alpha=0.55, num_outer=4, num_inner=1)

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

# Dimensionless distance from (x=L, z=H/2)
if np.isclose(u_rot, 0.0):
    stream_expr = Constant(pi)
else:
    l = sqrt((periodic_distance(x,L,L)/L)**2 + ((z - H/2)/H)**2)
    tau = 2*pi*(L*l_a/2.0)/u_rot
    psi_a = pi / tau
    psi_b = -pi*l_a / (tau*(l_b-l_a))
    psi_c = pi*l_a*l_b*L**2/tau
    psi_d = pi*l_a*l_b*L**2/tau
    stream_expr = conditional(l < l_a, psi_a*(L*l)**2,
                            conditional(l < l_b, psi_b*L**2*(l - l_b)**2 + psi_c,
                                        psi_d))
Vpsi = domain.spaces("H1")
stream = Function(Vpsi)
stream.interpolate(stream_expr)
u0.project(shear_expr - domain.perp(grad(stream)))

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)
