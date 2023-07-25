from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, min_value, acos

import numpy as np

day = 24.*60.*60.
dt = 900.
tmax = 12*day
R = 6371220.

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
V = domain.spaces("DG")
eqn = AdvectionEquation(domain, V, "D")

# I/O
dirname = "NL"

output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          log_level="INFO")
io = IO(domain, output)

# get lat lon coordinates
theta, lamda = latlon_coords(mesh)
phi_c = 0.0
lamda_c1 = 5*pi/6
lamda_c2 = 7*pi/6
R_t = R/2.

def dist(lamda_, theta_, R0=R):
    return R0*acos(sin(theta_)*sin(theta) + cos(theta_)*cos(theta)*cos(lamda - lamda_))

d1 = min_value(1.0, dist(lamda_c1, phi_c)/R_t)
d2 = min_value(1.0, dist(lamda_c2, phi_c)/R_t)
Dexpr = 0.5*(1 + cos(pi*d1)) + 0.5*(1 + cos(pi*d2))

T = tmax
tc = Constant(0)
k = 10*R/T
u_zonal = k*pow(sin(lamda - 2*pi*tc/T), 2)*sin(2*theta)*cos(pi*tc/T) + ((2*pi*R)/T)*cos(theta)
u_merid = k*sin(2*(lamda - 2*pi*tc/T))*cos(theta)*cos(pi*tc/T)
cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
cartesian_w_expr = u_merid*cos(theta)
u_expr = as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))

def ubar(t):
    return u_expr

transport_scheme = SSPRK3(domain)
transport_method = DGUpwind(eqn, "D")
    
# Time stepper
stepper = PrescribedTransport(eqn, transport_scheme, io, transport_method,
                              prescribed_transporting_velocity=ubar)

# initial conditions
u0 = stepper.fields("u")
D0 = stepper.fields("D")
D0.interpolate(Dexpr)
u0.project(u_expr)

stepper.run(t=0, tmax=tmax)
