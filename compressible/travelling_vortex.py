"""
The Travelling Vortex test case in a three dimensional channel
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, 
                       Function, conditional, sqrt, as_vector, atan2)

from gusto.diagnostics import (CompressibleAbsoluteVorticity, 
                               CompressibleRelativeVorticity)



tmax = 100
dt = 0.5
dumpfreq = int(tmax / (10*dt))
degree = (1, 1)

uc = 100.
vc = 100.
L = 10000.
H = 10000.
nlayers = 32
ncolumns = 32

t = 0.
delx = 100.
nx = int(L/delx)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, periodic=True)
x, y = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'CG',  horizontal_degree=degree[0], 
                vertical_degree=degree[1])


# Equations
params = CompressibleParameters(g=0)
eqns = CompressibleEulerEquations(domain, params)
eqns.bcs['u'] = []
print(f'ideal number of processors = {eqns.X.function_space().dim() / 50000}')

#I/O
dirname='traveling_vortex'
output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq, dump_nc=True)
diagnostics = [XComponent('u'), ZComponent('u'),  Pressure(eqns)]
io = IO(domain, output, diagnostic_fields=diagnostics)

#Transport options
theta_opts = EmbeddedDGOptions()
transported_fields = [TrapeziumRule(domain, "u"),
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
u0 = stepper.fields('u')
rho0 = stepper.fields('rho')
theta0 = stepper.fields('theta')

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()

# constants
kappa = params.kappa
p_0 = params.p_0
xc = L / 2  
yc = L / 2 

R = 0.4 * L # radius of voertex
r = sqrt(x**2 + y**2) / R
phi = atan2((y - yc),(x - xc))

# rho expression
rhoc = 1
rhob = Function(Vr).interpolate(conditional(r>=1, rhoc,  1- 0.5*(1-r**2)**6))

# u expression
u_cond = (1024 * (1.0 - r)**6 * r**6 )
ux_expr = conditional(r>=1, uc, uc + u_cond * (-(y-yc)/(r*R))) 
uy_expr = conditional(r>=1, vc, vc + u_cond * (-(x-xc)/(r*R))) 
u0.project(as_vector([ux_expr, uy_expr]))

#pressure and theta expressions
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

p0 = 0
for i in range(25):
    p0 += coe[i] * (r**(12+i)-1)
mach = 0.341
p = 1 + 1024**2 * mach**2 *conditional(r>=1, 0, p)

R0 = 287.
gamma=1.4
pref=params.p_0

T = p / (rhob*R0)
thetab = Function(Vt).interpolate(T*(pref / p)**0.286)
#compressible_hydrostatic_balance(eqns, theta, rho, solve_for_rho=True)

# make mean fields
print('make mean fields')
theta0.assign(thetab)
rho0.assign(rhob)

# assign reference profiles
stepper.set_reference_profiles([('rho', rhob),
                                ('theta', thetab)])
# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax)