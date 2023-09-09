from gusto import *
from firedrake import PeriodicIntervalMesh, ExtrudedMesh, Constant, ge, le, exp, cos, \
    sin, conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, min_value, acos, as_vector

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

There are two configurations that can be run:
  The 'convergence' configuration has an initial condition of a linearly 
  varying density field and two Gaussian bumps for the mixing ratio.
  The 'consistency' configuration has an initial condition of a 
  constant mixing ratio and two Gaussian bumps for the density.

"""

# Specify whether to run the 'convergence' or 'consistency' version of the test.
case = 'convergence'

# Choose space for the mixing ratio.
# Theta will make it staggered relative to the density
# DG will make it colocated with the density
m_X_space = 'DG'

# Domain
Lx = 2000.
Hz = 2000.

# Time parameters
dt = 2.
tmax = 2000.

nlayers = 200.  # horizontal layers
columns = 200.  # number of columns

dx = Lx/nlayers
dz = Hz/columns

period_mesh = PeriodicIntervalMesh(columns, Lx)
mesh = ExtrudedMesh(period_mesh, layers=nlayers, layer_height=Hz/nlayers)
domain = Domain(mesh, dt, "CG", 1)
x,z = SpatialCoordinate(mesh)

# Define the mixing ratio and density as tracers
m_X = ActiveTracer(name='m_X', space=m_X_space,
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_eqn=TransportEquationType.advective)
                 
rho_d = ActiveTracer(name='rho_d', space='DG',
                 variable_type=TracerVariableType.density,
                 transport_eqn=TransportEquationType.conservative)

tracers = [m_X,rho_d]

# Equation
V = domain.spaces("HDiv")
eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu = V)

# I/O
dirname = "vertical_slice_nair_lauritzen_diff_diags_"+case

# Dump the solution at each day
dumpfreq = int(100./dt)

# Set dump_nc = True to use tomplot.
output = OutputParameters(dirname=dirname,
                          dumpfreq = dumpfreq,
                          dump_nc = True,
                          dump_vtus = False)

# Use a tracer density diagnostic to track conservation
diagnostic_fields_interpolate = TracerDensity('m_X','rho_d',method='interpolate')
diagnostic_fields_project = TracerDensity('m_X','rho_d',method='project')
#diagnostic_fields_assign = TracerDensity('m_X','rho_d',method='assign')
#diagnostic_fields_solve = TracerDensity('m_X','rho_d',method='solve')

diagnostic_fields_list = [diagnostic_fields_interpolate, diagnostic_fields_project] #diagnostic_fields_assign]#, diagnostic_fields_solve]

io = IO(domain, output, diagnostic_fields=diagnostic_fields_list)

# Set up the divergent, time-varying, velocity field
U = Lx/tmax
W = U/10.

def u_t(t):
  xd = x - U*t
  u = U - (W*pi*Lx/Hz)*cos(pi*t/tmax)*cos(2*pi*xd/Lx)*cos(pi*z/Hz)
  w = 2*pi*W*cos(pi*t/tmax)*sin(2*pi*xd/Lx)*sin(pi*z/Hz)
  
  u_expr = as_vector((u,w))
  
  return u_expr

# Specify locations of the two Gaussians
xc1 = 5.*Lx/8.
zc1 = Hz/2.

xc2 = 3.*Lx/8.
zc2 = Hz/2.

def l2_dist(xc,zc):
  return min_value(abs(x-xc), Lx-abs(x-xc))**2 + (z-zc)**2

lc = 2.*Lx/25.
m0 = 0.02

# Set the initial states from the choice of configuration
if case == 'convergence':
  f0 = 0.05
  
  rho_t = 0.5
  rho_b = 1.
  
  rho_d_0 = rho_b + z*(rho_t-rho_b)/Hz  
  
  g1 = f0*exp(-l2_dist(xc1,zc1)/(lc**2))
  g2 = f0*exp(-l2_dist(xc2,zc2)/(lc**2))
  
  m_X_0 = m0 + g1 + g2
  
elif case == 'consistency':
  f0 = 0.5
  rho_b = 0.5
  
  g1 = f0*exp(-l2_dist(xc1,zc1)/(lc**2))
  g2 = f0*exp(-l2_dist(xc2,zc2)/(lc**2))
  
  rho_d_0 = rho_b + g1 + g2
  
  m_X_0 = m0 + 0*x
  
else:
  raise NotImplementedError('Specified case is not recognised.')


transport_scheme = SSPRK3(domain)
transport_methods = [DGUpwind(eqn, "m_X"), DGUpwind(eqn, "rho_d")]
    
# Time stepper
stepper = PrescribedTransport(eqn, transport_scheme, io, transport_methods,
                              prescribed_transporting_velocity=u_t)

# initial conditions
stepper.fields("m_X").interpolate(m_X_0)
stepper.fields("rho_d").interpolate(rho_d_0)
u0 = stepper.fields("u")
u0.project(u_t(0))

stepper.run(t=0, tmax=tmax)