from gusto import *
from firedrake import IcosahedralSphereMesh, Constant, ge, le, exp, cos, \
    sin, conditional, interpolate, SpatialCoordinate, VectorFunctionSpace, \
    Function, assemble, dx, FunctionSpace, pi, max_value, acos, as_vector

import numpy as np

# This script runs the Laurtizen et al. (2015) Terminator Toy
# test case. This examines the interaction of two species
# in the transport equation. There is coupling 
# between the two species to model combination 
# and dissociation.

######################

# Time parameters
day = 24.*60.*60.
dt = 900.
tmax = 12*day # this is 1036800s

# Radius of the Earth
R = 6371220.

# Domain
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=3, degree=2)
                             
x = SpatialCoordinate(mesh)

# get lat lon coordinates
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])

domain = Domain(mesh, dt, 'BDM', 1)

# Define the dry density and 
# the two species as tracers
rho_d = ActiveTracer(name='rho_d', space='DG',
                 variable_type=TracerVariableType.density,
                 transport_eqn=TransportEquationType.conservative)

X = ActiveTracer(name='X', space='DG',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_eqn=TransportEquationType.advective)
                 
X2 = ActiveTracer(name='X2', space='DG',
                 variable_type=TracerVariableType.mixing_ratio,
                 transport_eqn=TransportEquationType.advective)
                 
tracers = [rho_d, X, X2]

# Equation
V = domain.spaces("HDiv")
eqn = CoupledTransportEquation(domain, active_tracers=tracers, Vu = V)

# I/O
dirname = "terminator_toy_implicit_form"

# Dump the solution at each day
dumpfreq = int(day/dt)

# Set dump_nc = True to use tomplot.
output = OutputParameters(dirname=dirname,
                          dumpfreq = dumpfreq,
                          dump_nc = True,
                          dump_vtus = True)               
                         
X_twice = Sum('X', 'X')
X2_twice = Sum('X2', 'X2')
X_plus_X2 = Sum('X', 'X2')
X_plus_X2_plus_X2 = Sum('X_plus_X2', 'X2') 
tracer_diagnostic = TracerDensity('X_plus_X2_plus_X2', 'rho_d')
                          
io = IO(domain, output, diagnostic_fields = [X_twice, X2_twice, X_plus_X2, X_plus_X2_plus_X2, tracer_diagnostic])

# Define the reaction rates:
theta_c = np.pi/9.
lamda_c = -np.pi/3.

k1 = max_value(0, sin(theta)*sin(theta_c) + cos(theta)*cos(theta_c)*cos(lamda-lamda_c))
k2 = 1

# This is used in PrescribedTransport()
physics_parameterisations = [TerminatorToy(eqn, k1=k1, k2=k2, species1_name='X',
                    species2_name='X2')]

#terminator_stepper = ForwardEuler(domain)
#terminator_stepper = SSPRK3(domain)
#terminator_stepper = BackwardEuler(domain)

implicit_formulation = True
#terminator_stepper = ForwardEuler(domain) if implicit_formulation else BackwardEuler(domain)

terminator_stepper = BackwardEuler(domain)

# This is used in SplitPhysicsTimestepper
physics_schemes = [(TerminatorToy(eqn, k1=k1, k2=k2, species1_name='X',
                    species2_name='X2', implicit_formulation=implicit_formulation)
                    , terminator_stepper)]

# Set up two Gaussian bumps for the initial density field
theta_c1 = 0.0
theta_c2 = 0.0
lamda_c1 = -pi/4
lamda_c2 = pi/4

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

rho_expr = g1 + g2

# Define the initial amounts of the species:
X_T_0 = 4e-6

#r = k1/(4*k2)

#D_val = sqrt(r**2 + 2*X_T_0*r)

#X_0 = D_val - r
#X2_0 = 0.5*(X_T_0 - D_val + r)

X_0 = X_T_0 + 0*theta
X2_0 = 0*theta

T = tmax
k = 10*R/T

# Set up a non-divergent, time-varying, velocity field
def u_t(t):
  u_zonal = k*(sin(lamda - 2*pi*t/T)**2)*sin(2*theta)*cos(pi*t/T) + ((2*pi*R)/T)*cos(theta)
  u_merid = k*sin(2*(lamda - 2*pi*t/T))*cos(theta)*cos(pi*t/T)
  
  cartesian_u_expr = -u_zonal*sin(lamda) - u_merid*sin(theta)*cos(lamda)
  cartesian_v_expr = u_zonal*cos(lamda) - u_merid*sin(theta)*sin(lamda)
  cartesian_w_expr = u_merid*cos(theta)
  
  u_expr = as_vector((cartesian_u_expr, cartesian_v_expr, cartesian_w_expr))
  
  # Standard velocity profile
  return u_expr
  
  # No flow test:
  #return Constant(0)*u_expr
  
SUPG_options = SUPGOptions()

transport_scheme = SSPRK3(domain)
#transport_scheme = SSPRK3(domain, options=SUPG_options)
#transport_scheme = ForwardEuler(domain)
#transport_scheme = TrapeziumRule(domain)
#transport_method = [DGUpwind(eqn, 'rho_d', ibp=SUPG_options.ibp), DGUpwind(eqn, 'X', ibp=SUPG_options.ibp), DGUpwind(eqn, 'X2', ibp=SUPG_options.ibp)]
#transport_method = [DGUpwind(eqn, 'rho_d'), DGUpwind(eqn, 'X'), DGUpwind(eqn, 'X2')]
transport_method = [DGUpwind(eqn, 'rho_d'), DGUpwind(eqn, 'X'), DGUpwind(eqn, 'X2')]

# Time stepper
#stepper = PrescribedTransport(eqn, transport_scheme, io, transport_method,
#                              physics_parametrisations=physics_parameterisations,
#                              prescribed_transporting_velocity=u_t)

# When using SplitPhysicsTimestepper, we define an initial velocity
# that won't change

stepper = SplitPhysicsTimestepper(eqn, transport_scheme, io, 
                                    spatial_methods=transport_method,
                                    physics_schemes=physics_schemes,
                                    prescribed_transporting_velocity=u_t)
                                    
# The new hybrid timestepper

#stepper = SplitPrescribedTransport(eqn, transport_scheme, io, 
#                                    spatial_methods=transport_method,
#                                    physics_schemes=physics_schemes,
#                                    prescribed_transporting_velocity=u_t)

# initial conditions
stepper.fields("rho_d").interpolate(rho_expr)
stepper.fields("X").interpolate(X_0)
stepper.fields("X2").interpolate(X2_0)

stepper.run(t=0, tmax=tmax)
