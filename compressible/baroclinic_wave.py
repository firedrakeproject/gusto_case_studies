"""
The deep atmosphere dry baroclinic wave Test case from Ullich et al. 2013
Script by Daniel Witt
Last Updated 22/07/2024

Variable height: Applies a non uniform height field.
                 Default = True
Alpha:           Adjusts the ratio of implicit to explicit in the solver.
                 Default = 0.5
"""

from firedrake import (ExtrudedMesh, SpatialCoordinate, cos, sin, pi, sqrt,
                       exp, Constant, Function, as_vector, acos, errornorm, norm,
                       le, ge, NonlinearVariationalProblem, NonlinearVariationalSolver)
from gusto import *
# --------------------------------------------------------------#
# Configuratio Options
# -------------------------------------------------------------- #

order = (1, 1)
c = 16  # number of cells per cube face of cubed sphere
nlayers = 15
dt = 900.
days = 15.
tmax = days * 24. * 60. * 60.
alpha = 0.50  # ratio between implicit and explict in solver
variable_height = True
u_form = 'vector_advection_form'

# -------------------------------------------------------------- #
# Set up Model
# -------------------------------------------------------------- #
# Domain
a = 6.371229e6  # radius of earth
ztop = 3.0e4  # max height
dirname = f'baroclinic_wave_c={c}_dt={dt}_{u_form}_order={order[0]}_{order[1]}'

if variable_height is True:
    layerheight = []
    runningheight = 0
    # Calculating Non-uniform height field
    for m in range(1, nlayers+1):
        mu = 3
        if nlayers == 15:
            mu = 15
        if nlayers == 30:
            mu = 15
        height = ztop * ((mu * (m / nlayers)**2 + 1)**0.5 - 1) / ((mu + 1)**0.5 - 1)
        width = height - runningheight
        runningheight = height
        layerheight.append(width)
else:
    layerheight = ztop / nlayers

base_mesh = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=c, degree=2)
mesh = ExtrudedMesh(base_mesh, layers=nlayers, layer_height=layerheight, extrusion_type='radial')
domain = Domain(mesh, dt, "RTCF", horizontal_degree=order[0], vertical_degree=order[1])

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector(0, 0, omega)
print('making eqn')
eqn = CompressibleEulerEquations(domain, params, Omega=Omega, u_transport_option=u_form)
print(f'Optimal Cores = {eqn.X.function_space().dim() / 50000}')

output = OutputParameters(dirname=dirname,
                          dumpfreq=(3600 * 8) / dt,  # every 8 hours
                          dump_nc=True,
                          dump_vtus=False)
diagnostic_fields = []
io = IO(domain, output, diagnostic_fields=diagnostic_fields)
# Transport options
theta_opts = EmbeddedDGOptions()
transported_fields = [SSPRK3(domain, "u"),
                      SSPRK3(domain, "rho"),
                      SSPRK3(domain, "theta", options=theta_opts)]
transport_methods = [DGUpwind(eqn, "u"),
                     DGUpwind(eqn, "rho"),
                     DGUpwind(eqn, "theta")]

# Linear Solver
linear_solver = CompressibleSolver(eqn, alpha=alpha)

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver, alpha=alpha,
                                  num_outer=4, num_inner=1)

# -------------------------------------------------------------- #
# Initial Conditions
# -------------------------------------------------------------- #
xyz = SpatialCoordinate(mesh)
lon, lat, _ = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
x = xyz[0]
y = xyz[1]
z = xyz[2]
r = sqrt(x**2 + y**2 + z**2)
l = sqrt(x**2 + y**2)

# set up parameters
Rd = params.R_d
cp = params.cp
g = params.g
p0 = Constant(100000)

lapse = 0.005
T0e = 310  # Equatorial temp
T0p = 240  # Polar surface temp
T0 = 0.5 * (T0e + T0p)
H = Rd * T0 / g  # scale height of atmosphere
k = 3  # power of temp field
b = 2  # half width parameter
Vp = 1  # maximum velocity of perturbation

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
tao2_int = C * (r - a) * exp(-((r-a) / (b*H))**2)

# Variable fields
Temp = (a / r)**2 * (tao1 - tao2 * (s**k - (k / (k + 2)) * s**(k + 2)))**(-1)
P_expr = p0 * exp(-g / Rd * tao1_int + g / Rd * tao2_int * (s**k - (k / (k+2)) * s**(k+2)))

# wind expression
wind_proxy = (g / a) * k * Temp * tao2_int * (((r * cos(lat)) / a)**(k-1) - ((r * cos(lat)) / a)**(k+1))
wind = -omega * r * cos(lat) + sqrt((omega * r * cos(lat))**2 + r * cos(lat) * wind_proxy)

theta_expr = Temp * (P_expr / p0) ** (-params.kappa)
pie_expr = Temp / theta_expr
rho_expr = P_expr / (Rd * Temp)

# -------------------------------------------------------------- #
# Perturbation
# -------------------------------------------------------------- #

base_zonal_u = wind
base_merid_u = Constant(0.0)

a = 6.371229e6
zt = 1.5e4     # top of perturbation
d0 = a / 6     # horizontal radius of perturbation
a = 6.371229e6
err_tol = 1e-12
lon_c, lat_c = pi/9, 2*pi/9  # location of perturbation centre

d = a * acos(sin(lat_c)*sin(lat) + cos(lat_c)*cos(lat)*cos(lon - lon_c))  # distance from centre of perturbation

depth = r - a  # The distance from origin subtracted from earth radius
zeta = conditional(ge(depth, zt-err_tol), 0, 1 - 3*(depth / zt)**2 + 2*(depth / zt)**3)  # peturbation vertical taper

perturb_magnitude = (16*Vp/(3*sqrt(3))) * zeta * sin((pi * d) / (2 * d0)) * cos((pi * d) / (2 * d0))**3


zonal_pert = conditional(le(d, err_tol), 0,
                         conditional(ge(d, (d0-err_tol)), 0, -perturb_magnitude * (-sin(lat_c)*cos(lat) + cos(lat_c)*sin(lat)*cos(lon - lon_c)) / sin(d / a)))
meridional_pert = conditional(le(d, err_tol), 0,
                              conditional(ge(d, d0-err_tol), 0, perturb_magnitude * cos(lat_c)*sin(lon - lon_c) / sin(d / a)))

zonal_u = base_zonal_u + zonal_pert
merid_u = base_merid_u + meridional_pert
radial_u = Constant(0.0)
# Get spherical basis vectors, expressed in terms of thier (x,y,z) components:
e_lon = xyz_vector_from_lonlatr(1, 0, 0, xyz)
e_lat = xyz_vector_from_lonlatr(0, 1, 0, xyz)
e_r = xyz_vector_from_lonlatr(0, 0, 1, xyz)

# obtain initial conditions
print('Set up initial conditions')
print('project u')
test_u = TestFunction(Vu)
dx_reduced = dx(degree=4)
u_field = zonal_u*e_lon + merid_u * e_lat + radial_u * e_r
u_proj_eqn = inner(test_u, u0 - u_field) * dx_reduced
u_proj_prob = NonlinearVariationalProblem(u_proj_eqn, u0)
u_proj_solver = NonlinearVariationalSolver(u_proj_prob)
u_proj_solver.solve()

print('interpolate theta')
theta0.interpolate(theta_expr)
print('find pi')
pie = Function(Vr).interpolate(pie_expr)
print('find rho')
rho0.interpolate(rho_expr)
compressible_hydrostatic_balance(eqn, theta0, rho0, exner_boundary=pie, solve_for_rho=True)

print('make analytic rho')
rho_analytic = Function(Vr).interpolate(rho_expr)
print('Normalised rho error is:', errornorm(rho_analytic, rho0) / norm(rho_analytic))

# make mean fields
print('make mean fields')
rho_b = Function(Vr).assign(rho0)
theta_b = Function(Vt).assign(theta0)

# assign reference profiles
stepper.set_reference_profiles([('rho', rho_b),
                                ('theta', theta_b)])

stepper.run(t=0, tmax=tmax)
