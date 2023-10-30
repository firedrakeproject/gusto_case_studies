"""
The dry Held-Suazrez test case.
Script by Daniel Witt. 
Last Updated 25/09/2023

3 Different configuration are provided for transport set up:

Config 1: Lowest order finite element space, with vector advection form 
          and no theta limiter
Config 4: Next to lowest order space with vector advection form, EmbeddedDG for 
          theta transport and no limiter
Config 7: Next to lowest order space with vector invariant form, EmbeddedDG for 
          theta transport and no limiter


In addition to these configuration there are optional adjustments

Variable height: Applies a non uniform height field.  
                 Default = True
Alpha:           Adjusts the ratio of implicit to explicit in the solver. 
                 Default = 0.5
"""

from firedrake import (ExtrudedMesh,  TensorProductElement, ln,
                       SpatialCoordinate, cos, sin, pi, sqrt, HDiv, HCurl,
                       exp, Constant, Function, as_vector, acos, interval,
                       errornorm, norm, min_value, max_value, le, ge, FiniteElement,
                       NonlinearVariationalProblem, NonlinearVariationalSolver)
from gusto import *                                            #

# --------------------------------------------------------------#
# Configuratio Options
# -------------------------------------------------------------- #
config = 'config4'
dt = 1200.
days = 200.
tmax = days * 24. * 60. * 60.
n = 5  # cells per cubed sphere face edge
nlayers = 5 # vertical layers
alpha = 0.50 # ratio between implicit and explict in solver

# Lowest order vector advection
if config == 'config1':   # lowest order no limiter
    DGdegree = 0
    u_form = 'vector_advection_form'
    transport_name = 'recovered'
    limited = False
    n = n*2
    nlayers = nlayers*2

# Vectir advection
elif config =='config4': # Vector advection embedded not limited
    DGdegree = 1
    u_form = 'vector_advection_form' 
    theta_transport = EmbeddedDGOptions()
    transport_name = 'embedded'
    limited = False

# Vector invariant
elif config =='config7': # vector invariant embedded not limited
    DGdegree = 1
    u_form = 'vector_invariant_form'
    theta_transport = EmbeddedDGOptions()
    transport_name = 'embedded'
    limited = False

# -------------------------------------------------------------- #
# Script Options
# -------------------------------------------------------------- #

a = 6.371229e6  # radius of earth
ztop = 3.2e4  # max height

# Height field
layerheight = ztop / nlayers

# Mesh 
base_mesh = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=n, degree=2)
mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=layerheight, extrusion_type='radial')
xyz= SpatialCoordinate(mesh)
lon, lat, r = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
domain = Domain(mesh, dt, "RTCF", degree=DGdegree)

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector((0, 0, omega))
print('making eqn')    
eqn = CompressibleEulerEquations(domain, params, Omega=Omega, u_transport_option=u_form)
print(f'Number of DOFs = {eqn.X.function_space().dim()}')


dirname = f'Held_suarez_no_relaxation'
output = OutputParameters(dirname=dirname,
                          dumpfreq=9, # every 3 hours
                          dump_nc=True,
                          dump_vtus=False)
diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'),RadialComponent('u'),
                    CourantNumber(), Temperature(eqn), Pressure(eqn)]
          
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Schemes 
transported_fields = []
transport_methods = []
if u_form == 'vector_invariant_form':
    transported_fields.append(TrapeziumRule(domain, "u"))
    transport_methods.append(DGUpwind(eqn, 'u'))
else:
    transported_fields.append(TrapeziumRule(domain, "u", options=SUPGOptions()))
    transport_methods.append(DGUpwind(eqn, 'u', ibp=SUPGOptions().ibp))
    
transported_fields.append(SSPRK3(domain, "rho"))
transported_fields.append(SSPRK3(domain, "theta", options=theta_transport))

transport_methods.append(DGUpwind(eqn, 'rho'))
transport_methods.append(DGUpwind(eqn, 'theta', ibp=IntegrateByParts.ONCE))

# Linear Solver
linear_solver = CompressibleSolver(eqn, alpha=alpha)

# set up parameters
Rd = params.R_d
cp = params.cp
kappa = Rd / cp
g = params.g
p0 = Constant(100000)

lapse = 0.005
T0init = 300
d = 24*60*60 # A day
T0stra = 200 # Stratosphere temp
T0surf = 315 # Surface temperature at equator
T0horiz = 60 # Equator to pole temperature difference
T0vert = 10 # Stability parameter
k = 3
H = Rd * T0surf / g # scale height of atmosphere
b = 2 # half width parameter
sigmab = 0.7
taod = 40 * d
taou = 4 * d
taofric = d 
s = (r / a) * cos(lat)
A = 1 / lapse
tao1 = A * lapse / T0init * exp((r - a)*lapse / T0init)
tao1_int = A * (exp(lapse * (r - a) / T0init) - 1)
P_expr = p0 * exp(-g / Rd * tao1_int)
exner = (P_expr / p0) ** (params.kappa)
theta_expr = T0init / exner
pie_expr = T0init / theta_expr

T_condition = (T0surf - T0horiz * sin(lat)**2 - T0vert * ln(P_expr/p0) * cos(lat)**2) * (P_expr / p0)**kappa
Teq = conditional(ge(T0stra, T_condition), T0stra, T_condition)
equilibrium_expr = Teq / exner
# timescale of temperature forcing
sigma = P_expr / p0
tao_cond = (sigma - sigmab) / (1 - sigmab)*cos(lat)**4
tau_rad_inverse = 1 / taod + (1/taou - 1/taod) * conditional(ge(0, tao_cond), 0, tao_cond)
temp_coeff = exner * tau_rad_inverse



physics_schemes = [(Relaxation(eqn, 'theta', equilibrium_expr, coeff=temp_coeff), ForwardEuler(domain))]

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver, alpha=alpha,
                                  physics_schemes=physics_schemes)

# -------------------------------------------------------------- #
# Parameter
# -------------------------------------------------------------- #
x = xyz[0]
y = xyz[1]
z = xyz[2]
l = sqrt(x**2 + y**2)

# set up parameters
Rd = params.R_d
cp = params.cp
kappa = Rd / cp
g = params.g
p0 = Constant(100000)

lapse = 0.005
T0init = 300
d = 24*60*60 # A day
T0stra = 200 # Stratosphere temp
T0surf = 315 # Surface temperature at equator
T0horiz = 60 # Equator to pole temperature difference
T0vert = 10 # Stability parameter
k = 3
H = Rd * T0surf / g # scale height of atmosphere
b = 2 # half width parameter
sigmab = 0.7
taod = 40 * d
taou = 4 * d
taofric = d 
# -----------------------------------------------------------------------------
# Background Profiles
# -----------------------------------------------------------------------------
s = (r / a) * cos(lat)
A = 1 / lapse
tao1 = A * lapse / T0init * exp((r - a)*lapse / T0init)
tao1_int = A * (exp(lapse * (r - a) / T0init) - 1)
P_expr = p0 * exp(-g / Rd * tao1_int)
exner = (P_expr / p0) ** (params.kappa)
theta_expr = T0init / exner
pie_expr = T0init / theta_expr

rho_expr = P_expr / (Rd * T0init)
# Spaces
u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")
Vu = u0.function_space()
Vr = rho0.function_space()
Vt = theta0.function_space()
# ------------------------------------------------------------------------------
# Relxation conditions
# ------------------------------------------------------------------------------

# temperature
T_condition = (T0surf - T0horiz * sin(lat)**2 - T0vert * ln(P_expr/p0) * cos(lat)**2) * (P_expr / p0)**kappa
Teq = conditional(ge(T0stra, T_condition), T0stra, T_condition)
equilibrium_expr = Function(Vt).interpolate(Teq / exner)
# timescale of temperature forcing
sigma = P_expr / p0
tao_cond = (sigma - sigmab) / (1 - sigmab)*cos(lat)**4
tau_rad_inverse = 1 / taod + (1/taou - 1/taod) * conditional(ge(0, tao_cond), 0, tao_cond)
temp_coeff = exner * tau_rad_inverse
# Velocity
wind_timescale = 1 / taofric * conditional(ge(0, tao_cond), 0, tao_cond)

# ------------------------------------------------------------------------------
# Field Initilisation
# ------------------------------------------------------------------------------

theta0.interpolate(theta_expr)
pie = Function(Vr).interpolate(pie_expr)
rho0.interpolate(rho_expr)
compressible_hydrostatic_balance(eqn, theta0, rho0, exner_boundary = pie, solve_for_rho=True)
rho_b = Function(Vr).assign(rho0)
theta_b = Function(Vt).assign(theta0)
stepper.set_reference_profiles([('rho', rho_b), 
                                ('theta', theta_b)])
print('Intialise Windy Boi')
stepper.run(t=0, tmax=tmax)