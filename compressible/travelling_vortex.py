"""
The Travelling Vortex test case in a three dimensional channel
"""

from gusto import *
from firedrake import (PeriodicRectangleMesh, ExtrudedMesh, SpatialCoordinate, 
                       Function, conditional, sqrt, as_vector, atan_2)

from gusto.diagnostics import (CompressibleAbsoluteVorticity, 
                               CompressibleRelativeVorticity)
from gusto.thermodynamics import theta


tmax = 100
dt = 1
dumpfreq = 25
degree = (1, 1)

Lx = 10000 # length
Ly = 5000 # width
Lz = 10000  # height
dx = Lx / 32
dy = Ly / 8
dz = Lz / 32
nlayers = int(Lz / dz)
ncolumnsx = int(Lx / dx)
ncolmunsy = int(Ly / dy)

# Domain
m = PeriodicRectangleMesh(ncolumnsx, ncolmunsy, Lx, Ly, 'both', quadrilateral=True)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=dz)
domain = Domain(mesh, dt, 'RTCF',  horizontal_degree=degree[0], 
                vertical_degree=degree[1])


# Equations
params = CompressibleParameters()
eqns = CompressibleEulerEquations(domain, params)
print(f'ideal number of processors = {eqns.X.function_space().dim() / 50000}')

#I/O
dirname='traveling_vortex'
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq, dump_nc=True)
diagnostics = [CompressibleAbsoluteVorticity, CompressibleRelativeVorticity, 
               XComponent('u'), ZComponent('u'), YComponent('u')]
io = IO(domain, output, diagnostic_fields=diagnostics)

#Transport options
theta_opts = EmbeddedDGOptions()
transported_fields = [SSPRK3(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=theta_opts)]
transport_methods = [DGUpwind(eqns, "u"),
                     DGUpwind(eqns, "rho"),
                     DGUpwind(eqns, "theta")]
# solver
linear_solver = CompressibleSolver(eqns)
# timestepper
stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                    transport_methods,
                                    linear_solver=linear_solver)
# ------------------------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------------------------
# initial fields
u = stepper.fields('u')
rho = stepper.fields('rho')
theta = stepper.fields('theta')
x, y, z = SpatialCoordinate(mesh)

# spaces
Vu = u.function_space()
Vt = theta.function_space()
Vr = rho.function_space()

# constants
kappa = params.kappa
p_0 = params.p_0
xc = Lx / 2
yc = Ly / 2
zc = Lz / 2 
R = 2500 #scaling
r = sqrt(x**2 + y**2 + z**2)
u0 = 100
v0 = 0
w0 = 0
T = 270

phi = atan_2((z - zc),(x - xc))
u_r_expr = -1024 * sin(phi) * (1*r**2)**6 * r**6 + u0
w_r_expr = 1024 * sin(phi) * (1*r**2)**6 * r**6 + w0

p_r_expr = -1/72*r**36 + 6/35*r**35 - 15/17*r**34 + 74/33*r**33- 57/32*r**32 - 174/31*r**31 +\
          269/15 *r**30 - 450/29*r**29 - 153/8*r**28 + 1564/27*r**27 - 510/13*r**26 -\
          204/5*r**25 + 737/8*r**24 - 1032/23*r**23 - 477/11*r**22 + 64*r**21 -\
          81/40*r**20 -1242/19*r**19 + 731/9*r**18 - 966/17*r**17 + 219/8*r**16 -\
          146/15*r**15 + 18/7*r**14 - 6/13*r**13 + 1/24*r**12
rho_r_expr = 1 - 1/2*(1 - r**2)**6

u_expr = conditional(r < R, u_r_expr, u0)
w_expr = conditional(r< R, w_r_expr, w0)
p_expr = conditional(r < R, p_r_expr, 0)
rho_expr = conditional(r < R, rho_r_expr, 0)
U_expr = as_vector([u_expr, v0, w_expr])
theta_expr = T*(p_0 / p_expr)**kappa

u.project(U_expr)
rho.interpolate(rho_expr)
theta.interpolate(theta_expr)
compressible_hydrostatic_balance(eqns, theta, rho, solve_for_rho=True)

# make mean fields
print('make mean fields')
rho_b = Function(Vr).assign(rho)
theta_b = Function(Vt).assign(theta)

# assign reference profiles
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])
# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)