from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Constant, ge, le, exp, cos, \
    sin, conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, min_value, acos

import numpy as np

u"""
This script implements a the test given in the 'Charney-Phillips trilemma'
paper by Bendall, Wood, Thuburn, and Cotter. This considers a planar version of
the Gaussian test case given by Nair and Laurtizen. 

This tests a coupled transport equation for moisture.

The mixing ratio obeys an advective transport equation:
∂/∂t (m_X) + (u.∇)m_X = 0

Whereas the dry density obeys the conservative form:
∂/∂t (ρ_d) + ∇.(ρ_d*u) = 0

The relation between these is given by 

m_X = ρ_X/ρ_d

"""


# Specify whether to run the 'convergence' or 'consistency' version of the test.
case = 'convergence'

# Choose space for the mixing ratio.
# Theta if want staggered, DG if want colocated
m_X_space = 'DG'

# Domain
Lx = 2000
Hz = 2000

dx = 160
dz = 160

# Time parameters
dt = 2.
tmax = 2000.



nlayers = int(Hz/dz)  # horizontal layers
columns = int(Lx/dx)  # number of columns

period_mesh = PeriodicIntervalMesh(columns, Lx)
mesh = ExtrudedMesh(period_mesh, layers=nlayers, layer_height=Hz/nlayers)
domain = Domain(mesh, dt, "CG", 1)
x,z = SpatialCoordinate(mesh)

m_X = ActiveTracer(name='m_X', space=m_X_space,
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_eqn=TransportEquationType.conservative)
                 
rho_d = ActiveTracer(name='rho_d', space='DG',
                 variable_type=TracerVariableType.density,
                 transport_eqn=TransportEquationType.conservative)

# Define the tracers of mass and dry density
tracers = [m_X,rho_d]

# Equation
V = domain.spaces("DG")
eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu = V)

# I/O
dirname = "vertical_slice_nair_lauritzen"

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
xc1 = Lx/8.
zc1 = Hz/2.

xc2 = -Lx/8.
zc2 = Hz/2.

def l2_dist(xc,zc):
  return min_value(abs(x-xc), Lx-abs(x-xc))**2 + (z-zc)**2

if case == 'convergence':
  f0 = 0.05
  
  rho_t = 0.5
  rho_b = 1.
  
  rho_d_0 = rho_b + z*(rho_b-rho_t)/Hz
  
  m0 = 0.02
  
  lc = 2.*Lx/25.
  
  g1 = f0*exp(l2_dist(xc1,zc1)/(lc**2))
  g2 = f0*exp(l2_dist(xc2,zc2)/(lc**2))
  
  m_X_0 = m0 + g1 + g2
  
elif case == 'consistency':
  f0 = 0.5
  
  m0 = 0.02
  rho_b = 0.05
  
  g1 = f0*exp(l2_dist(xc1,zc1)/(lc**2))
  g2 = f0*exp(l2_dist(xc2,zc2)/(lc**2))
  
  rho_d = rho_b + g1 + g2
  
  m_X_0 = m0
  
else:
  raise NotImplementedError('Specified case is not recognised.')


transport_scheme = SSPRK3(domain)
#transport_methods = [DGUpwind(eqn, "m_X"), DGUpwind(eqn, "rho_d")]
transport_methods = []
    
# Time stepper
stepper = PrescribedTransport(eqn, transport_scheme, io, transport_methods,
                              prescribed_transporting_velocity=u_t)

# initial conditions
m_X_0 = stepper.fields("m_X")
rho_d_0 = stepper.fields("rho_d")
D0.interpolate(Dexpr)
u0.project(u_t(0))

stepper.run(t=0, tmax=tmax)