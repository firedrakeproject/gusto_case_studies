from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    sin, conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, min_value, acos

import numpy as np

# This script implements a the test given in the 'Charnery-Phillips trilemma'
# paper by Bendall, Wood, Thuburn, and Cotter. This considers a planar version of
# the Gaussian test case given by Nair and Laurtizen. 

# Specify whether to run the 'convergence' or 'consistency' version of the test.
case = 'convergence'

# Domain
Lx = 2000
Hz = 2000

dx = 160
dz = 160

nlayers = int(Hz/dz)  # horizontal layers
columns = int(Lx/dx)  # number of columns

m = PeriodicIntervalMesh(columns, Lx)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=Hz/nlayers)
domain = Domain(mesh, dt, "CG", 1)
    
x,z = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', 1)

# Equation
V = domain.spaces("DG")
eqn = AdvectionEquation(domain, V, "D")

# I/O
dirname = "planar_nair_gaussian"

# Time parameters
dt = 2.
tmax = 2000.

# Dump the solution at each day
dumpfreq = int(100./dt)

# Set dump_nc = True to use tomplot.
output = OutputParameters(dirname=dirname,
                          dumplist_latlon=['D'],
                          dumpfreq = dumpfreq,
                          log_level="INFO",
                          dump_nc = True,
                          dump_vtus = False)
                          
io = IO(domain, output)

# Set up the divergent, time-varying, velocity field
U = Lx/tmax
W = U/10.

def u_t(t):
  xd = x - (Lx/2) - U*t
  u = U - (W*pi*Lx/Hz)*cos(pi*t/tmax)*cos(2*pi*xd/Lx)*cos(pi*z/Hz)
  w = 2*pi*W*cos(pi*t/tmax)*sin(2*pi*xd/Lx)*sin(pi*z/Hz)
  
  u_expr = as_vector((u,w))
  
  return u_expr

# Specify locations of the two Gaussians
x_c1 = Lx/8.
z_c1 = Hz/2.

x_c2 = -Lx/8.
z_c2 = Hz/2.

def l2_dist(xc,zc):
  return min(abs(x-xc), Lx-abs(x-xc))**2 + (z-zc)**2

if case == 'convergence':
  f0 = 0.05
elif case == 'consistency':
  f0 = 0.5
else:
  raise NotImplementedError('Specified case is not recognised.')

# Construct the two Gaussians
g1 = 


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

T = tmax
k = 10*R/T




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