from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    sin, conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, min_value, acos

import numpy as np

# This script runs the Nair_Laurtizen (2010) test cases with 
# a divergent velocity field.

######################
# Specify the initial condition case for the scalar field
# Choose from (these all give two bumps):
# 1. cosine_bells - Cosine bells (Quasi-smooth scalar field)
# 2. gaussian - Gaussian surfaces (Smooth scalar field)
# 3. slotted_cylinder - Slotted Cylinder (Non-smooth scalar field)

scalar_case = 'slotted_cylinder'

######################

# Time parameters
day = 24.*60.*60.
dt = 300.
tmax = 12*day

# Radius of the Earth
R = 6371220.

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=5, degree=2)
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
V = domain.spaces("DG")
eqn = AdvectionEquation(domain, V, "D")

# I/O
dirname = "nair_lauritzen_div_"+scalar_case

# Set dump_nc = True to use tomplot.
output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          log_level="INFO",
                          dump_nc = True,
                          dump_vtus = False)
                          
io = IO(domain, output)

# get lat lon coordinates
theta, lamda = latlon_coords(mesh)

# Specify locations of the two bumps
theta_c1 = 0.0
theta_c2 = 0.0
lamda_c1 = -pi/4
lamda_c2 = pi/4

if scalar_case == 'cosine_bells': 

  R_t = R/2.

  def dist(lamda_, theta_, R0=R):
      return R0*acos(sin(theta_)*sin(theta) + cos(theta_)*cos(theta)*cos(lamda - lamda_))
  
  d1 = min_value(1.0, dist(lamda_c1, theta_c1)/R_t)
  d2 = min_value(1.0, dist(lamda_c2, theta_c2)/R_t)
  Dexpr = 0.5*(1 + cos(pi*d1)) + 0.5*(1 + cos(pi*d2))

elif scalar_case == 'gaussian':

  X = cos(theta)*cos(lamda)
  Y = cos(theta)*sin(lamda)
  Z = sin(theta)

  X1 = cos(theta_c1)*cos(lamda_c1)
  Y1 = cos(theta_c1)*sin(lamda_c1)
  Z1 = sin(theta_c1)
  
  X2 = cos(theta_c2)*cos(lamda_c2)
  Y2 = cos(theta_c2)*sin(lamda_c2)
  Z2 = sin(theta_c2)   
  
  # Define the two Gaussian bumps
  g1 = exp(-5*((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2))
  g2 = exp(-5*((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2))
  
  Dexpr = g1 + g2

elif scalar_case == 'slotted_cylinder':
      
  def dist1(lamda, theta, R0=R):
      return acos(sin(theta_c1)*sin(theta) + cos(theta_c1)*cos(theta)*cos(lamda - lamda_c1))
      
  def dist2(lamda, theta, R0=R):
      return acos(sin(theta)*sin(theta_c2) + cos(theta)*cos(theta_c2)*cos(lamda_c2 - lamda))
                  
  Dexpr = conditional( dist1(lamda, theta) < (1./2.), \
            conditional( (abs(lamda - lamda_c1) < (1./12.) ), \
              conditional( (theta - theta_c1) < (-5./24.), 1.0, 0.1), 1.0), \
                conditional ( dist2(lamda, theta) < (1./2.), \
                  conditional( (abs(lamda - lamda_c2) < (1./12.) ), \
                    conditional( (theta - theta_c2) > (5./24.), 1.0, 0.1), 1.0 ), 0.1 ))  

else:
  raise NotImplementedError('Scalar case specified has not been implemented')

T = 12*day#tmax
k = 10*R/T

# Set up the divergent, time-varying, velocity field
def u_t(t):
  u_zonal = -k*(sin(lamda/2)**2)*sin(2*theta)*(cos(theta)**2)*cos(pi*t/T)
  u_merid = k*sin(lamda)*(cos(theta)**3)*cos(pi*t/T)
  
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
