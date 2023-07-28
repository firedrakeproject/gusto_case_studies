from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    sin, conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, min_value, acos

import numpy as np

# This script runs the four part solid body rotation test
# given in the Bendall, Wimmer (2022) paper on improving
# the accuracy of vector transport with RT elements


# Time parameters
T = 200
dt = 0.05
tmax = 2*T

# Radius of the Earth
R = 6371220.

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
V = domain.spaces("HDiv")
eqn = AdvectionEquation(domain, V, "F")

# I/O
dirname = "four_part_sbr"

output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['F'],
                          log_level="INFO")
                          
diagnostic_fields = [ZonalComponent('F'), MeridionalComponent('F')]
                          
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# get lat lon coordinates
theta, lamda = latlon_coords(mesh)

# Specify locations of the bump
lamda_c = 0
theta_c = -pi/6

F0 = 3
r0 = 0.25

def dist(lamda_, theta_):
      return acos(sin(theta_)*sin(theta) + cos(theta_)*cos(theta)*cos(lamda - lamda_))

# Initialise the vector field to be transported
F_init_zonal = 0*lamda
F_init_merid = F0*exp(-((dist(lamda_c,theta_c)/r0)**2))

cartesian_F_u_expr = -F_init_zonal*sin(lamda) - F_init_merid*sin(theta)*cos(lamda)
cartesian_F_v_expr = F_init_zonal*cos(lamda) - F_init_merid*sin(theta)*sin(lamda)
cartesian_F_w_expr = F_init_merid*cos(theta)

F_init_expr = as_vector((cartesian_F_u_expr, cartesian_F_v_expr, cartesian_F_w_expr))

# Set up the advecting velocity field
# It is defined piecewise in time
U = 2*pi*R/T

def u_t(t):
  tc = float(2*t/T)
  if ( tc % 2 < 1 ):
    u_zonal = U*cos(theta)
    u_merid = 0*U
  else:
    u_zonal = -U*cos(lamda)*sin(theta)
    u_merid = U*sin(lamda)*(cos(theta)**2 - sin(theta)**2)
  
  cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
  cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
  cartesian_w_expr = u_merid*cos(theta)
  
  u_expr = as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))
  
  return u_expr


transport_scheme = SSPRK3(domain)
transport_method = DGUpwind(eqn, "F")
    
# Time stepper
stepper = PrescribedTransport(eqn, transport_scheme, io, transport_method,
                              prescribed_transporting_velocity=u_t)

# initial conditions
u0 = stepper.fields("u")
F0 = stepper.fields("F")
F0.interpolate(F_init_expr)
u0.project(u_t(0))

stepper.run(t=0, tmax=tmax)
