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
from gusto import *
from firedrake import (ExtrudedMesh, ln,
                       SpatialCoordinate, cos, sin, sqrt,
                       exp, Constant, Function, as_vector, ge,
                       NonlinearVariationalProblem, NonlinearVariationalSolver)

# --------------------------------------------------------------#
# Configuratio Options
# -------------------------------------------------------------- #
config = 'config4'
dt = 1200.
days = 200.
tmax = days * 24. * 60. * 60.
n = 12  # cells per cubed sphere face edge
nlayers = 15 # vertical layers
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
print(f'ideal cores = {eqn.X.function_space().dim() / 50000}')


dirname = f'HS_vel_config4'
output = OutputParameters(dirname=dirname,
                          dumpfreq=9, # every hour
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

physics_schemes = [(RayleighFriction(eqn, parameters=params), ForwardEuler(domain))]
#physics_schemes = [(Relaxation(eqn, 'theta', parameters=params), ForwardEuler(domain))]
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
g = params.g
p0 = Constant(100000)

lapse = 0.005
T0e = 310 # Equatorial temp
T0p = 240 # Polar surface temp
T0 = 0.5 * (T0e + T0p)
H = Rd * T0 / g # scale height of atmosphere
k = 3 # power of temp field
b = 2 # half width parameter

u0 = stepper.fields("u")
rho0 = stepper.fields("rho")
theta0 = stepper.fields("theta")

# spaces
Vu = u0.function_space()
Vr = rho0.function_space()
Vt = theta0.function_space()

# -------------------------------------------------------------- #
# Base State
# -------------------------------------------------------------- #

# expressions for variables from paper
s = (r / a) * cos(lat)
A = 1 / lapse
B = (T0e - T0p) / ((T0e + T0p)*T0p)
C = ((k + 2) / 2)*((T0e - T0p) / (T0e * T0p))

tao1 = A * lapse / T0 * exp((r - a)*lapse / T0) + B * (1 - 2*((r-a)/(b*H))**2)*exp(-((r-a) / (b*H))**2)
tao2 = C * (1 - 2*((r-a)/(b*H))**2)*exp(-((r - a) / (b*H))**2)

tao1_int = A * (exp(lapse * (r - a) / T0) - 1) + B * (r - a) * exp(-((r-a)/(b*H))**2)
tao2_int = C * (r - a)  * exp(-((r-a) / (b*H))**2)

# Variable fields
Temp = (a / r)**2 * (tao1 - tao2 * ( s**k - (k / (k+2)) *s**(k+2)))**(-1)
P_expr = p0 * exp(-g / Rd * tao1_int + g / Rd * tao2_int * (s**k - (k / (k+2)) *s**(k+2)))

# wind expression
wind_proxy = (g / a) * k * Temp * tao2_int * (((r * cos(lat)) / a)**(k-1) - ((r * cos(lat)) / a)**(k+1))
wind = -omega * r * cos(lat) + sqrt((omega * r * cos(lat))**2 + r * cos(lat) * wind_proxy )

theta_expr = Temp * (P_expr / p0) ** (-params.kappa) 
pie_expr = Temp / theta_expr
rho_expr = P_expr / (Rd * Temp)

# ------------------------------------------------------------------------------
# Relxation conditions
# ------------------------------------------------------------------------------

# temperature
#T_condition = (T0surf - T0horiz * sin(lat)**2 - T0vert * ln(P_expr/p0) * cos(lat)**2) * (P_expr / p0)**kappa
#Teq = conditional(ge(T0stra, T_condition), T0stra, T_condition)
#equilibrium_expr = Function(Vt).interpolate(Teq / exner)
# timescale of temperature forcing
#sigma = P_expr / p0
#tao_cond = (sigma - sigmab) / (1 - sigmab)*cos(lat)**4
#tau_rad_inverse = 1 / taod + (1/taou - 1/taod) * conditional(ge(0, tao_cond), 0, tao_cond)
#temp_coeff = exner * tau_rad_inverse
# Velocity
#wind_timescale = 1 / taofric * conditional(ge(0, tao_cond), 0, tao_cond)

# ------------------------------------------------------------------------------
# Field Initilisation
# ------------------------------------------------------------------------------
zonal_u = wind 
merid_u = Constant(0.0) 
radial_u = Constant(0.0)

e_lon = xyz_vector_from_lonlatr(1, 0, 0, xyz)
e_lat = xyz_vector_from_lonlatr(0, 1, 0, xyz)
e_r = xyz_vector_from_lonlatr(0, 0, 1, xyz)

print('Set up initial conditions')
print('project u')
test_u = TestFunction(Vu)
dx_reduced = dx(degree=4)
u_field = zonal_u*e_lon + merid_u * e_lat + radial_u * e_r 
u_proj_eqn = inner(test_u,u0 - u_field)*dx_reduced
u_proj_prob = NonlinearVariationalProblem(u_proj_eqn, u0)
u_proj_solver = NonlinearVariationalSolver(u_proj_prob)
u_proj_solver.solve()

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
