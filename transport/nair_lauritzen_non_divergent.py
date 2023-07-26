from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, min_value, acos

import numpy as np

# Time parameters
day = 24.*60.*60.
dt = 900.
tmax = 1*day

# Radius of the Earth
R = 6371220.

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
V = domain.spaces("DG")
eqn = AdvectionEquation(domain, V, "D")

#Specify the initial condition case for the scalar field
# Choose from:
# 1. cosine_bells - Cosine bells (Quasi-smooth scalar field)
# 2. gaussian_surfaces - Gaussian surfaces (Smooth scalar field)
# 3. slotted_cylinder - Slotted Cylinder (Non-smooth scalar field)

scalar_case = 'gaussian_surfaces'

# I/O
dirname = "nair_lauritzen_"+scalar_case

# Set dump_nc = True to use tomplot.
output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          log_level="INFO",
                          dump_nc = True,
                          dump_vtus = False)
                          
io = IO(domain, output)

# get lat lon coordinates
theta, lamda = latlon_coords(mesh)
phi_c = 0.0
lamda_c1 = 5*pi/6
lamda_c2 = 7*pi/6
R_t = R/2.

if scalar_case == 'cosine_bells': 

  def dist(lamda_, theta_, R0=R):
      return R0*acos(sin(theta_)*sin(theta) + cos(theta_)*cos(theta)*cos(lamda - lamda_))
  
  d1 = min_value(1.0, dist(lamda_c1, phi_c)/R_t)
  d2 = min_value(1.0, dist(lamda_c2, phi_c)/R_t)
  Dexpr = 0.5*(1 + cos(pi*d1)) + 0.5*(1 + cos(pi*d2))

elif scalar_case == 'gaussian_surfaces':
  X = R*cos(theta)*cos(lamda)
  Y = R*cos(theta)*sin(lamda)
  Z = R*sin(theta)

  X1 = R*cos(pi/3)*cos(pi)
  Y1 = R*cos(pi/3)*sin(pi)
  Z1 = R*sin(pi/3)  
  
  X2 = R*cos(-pi/3)*cos(pi)
  Y2 = R*cos(-pi/3)*sin(pi)
  Z2 = R*sin(-pi/3)   
  
  # Define the two Gaussian bumps
  g1 = exp(-5*((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2))
  g2 = exp(-5*((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2))
  
  Dexpr = g1 + g2

elif scalar_case == 'slotted_cylinder':
  Dexpr = 0.5

else:
  raise NotImplementedError('Scalar case specified has not been implemented')

T = 12*day#tmax
k = 10*R/T

# Set up the non-divergent, time-varying, velocity field
def u_t(t):
  u_zonal = k*pow(sin(lamda - 2*pi*t/T), 2)*sin(2*theta)*cos(pi*t/T) + ((2*pi*R)/T)*cos(theta)
  u_merid = k*sin(2*(lamda - 2*pi*t/T))*cos(theta)*cos(pi*t/T)
  
  cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
  cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
  cartesian_w_expr = u_merid*cos(theta)
  
  u_expr = as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))
  
  return u_expr


transport_scheme = SSPRK3(domain)
transport_method = DGUpwind(eqn, "D")
    
# Time stepper
stepper = PrescribedTransport(eqn, transport_scheme, io, transport_method,
                              prescribed_transporting_velocity=u_t)

# initial conditions
u0 = stepper.fields("u")
D0 = stepper.fields("D")
D0.interpolate(Dexpr)
u0.project(u_t(0))

stepper.run(t=0, tmax=tmax)
