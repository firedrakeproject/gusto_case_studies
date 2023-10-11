"""
The deep atmosphere dry baroclinic wave Test case from Ullich et al. 2013 
Script by Daniel Witt 
Last Updated 25/09/2023

7 Different configuration are provided for transport set up:

Config 1: Lowest order finite element space, with vector advection form 
          and no theta limiter
Config 2: Lowest order finite element space with vector advection 
          form and a DG1Limiter on theta
Config 3: Next to lowest order space with vector advection form with SUPGOptions
          for theta transport and no limiter
Config 4: Next to lowest order space with vector advection form, EmbeddedDG for 
          theta transport and no limiter
Config 5: Next to lowest order space with vector advection form, EmbeddedDG for 
          theta transport and a limiter on the theta field
Config 6: Next to lowest order space with vector invariant form with SUPGOptions
          for theta transport and no limiter
Config 7: Next to lowest order space with vector invariant form, EmbeddedDG for 
          theta transport and no limiter
Config 8: Next to lowest order space with vector invariant form, EmbeddedDG for 
          theta transport and a limiter on the theta field

In addition to these configuration there are optional adjustments

Perturbation:    The properties of the perturbation
                 None: No perturbation 
                 single: Velocity perturbation in the upper mid lattitude
                 double: Symetrical Perturbations in the N/S hemispheres
Variable height: Applies a non uniform height field.  
                 Default = True
Alpha:           Adjusts the ratio of implicit to explicit in the solver. 
                 Default = 0.5
"""

from firedrake import (ExtrudedMesh,  TensorProductElement,
                       SpatialCoordinate, cos, sin, pi, sqrt, HDiv, HCurl,
                       exp, Constant, Function, as_vector, acos, interval,
                       errornorm, norm, min_value, max_value, le, ge, FiniteElement,
                       NonlinearVariationalProblem, NonlinearVariationalSolver)
from gusto import *                                            #
# --------------------------------------------------------------#
# Configuratio Options
# -------------------------------------------------------------- #
config = 'config4'
dt = 225.
days = 15.
tmax = days * 24. * 60. * 60.
n = 24   # cells per cubed sphere face edge
nlayers = 15 # vertical layers
alpha = 0.50 # ratio between implicit and explict in solver
variable_height = True
perturbed = True
perturbation = 'single'


if perturbed:
    if perturbation == 'single':
        location = [(pi/9, 2*pi/9)]
    elif perturbation == 'double':
        location = [(pi/9, 2*pi/9), (pi/9, -2*pi/9)]
    else:
        raise ValueError('Please select a valid perturbation option')


# Lowest Order Configs
if config == 'config1':   # lowest order no limiter
    DGdegree = 0
    u_form = 'vector_advection_form'
    transport_name = 'recovered'
    limited = False
    n = n*2
    nlayers = nlayers*2

elif config == 'config2': # lowest order theta limited
    DGdegree = 0
    u_form = 'vector_advection_form'
    transport_name = 'recovered'
    limited = True
    n = n*2
    nlayers = nlayers*2

# Vector advection form options
elif config =='config3': # vector advection SUPG 
    DGdegree = 1
    u_form = 'vector_advection_form'
    theta_transport = SUPGOptions()
    transport_name = 'SUPG'
    limited = False

elif config =='config4': # Vector advection embedded not limited
    DGdegree = 1
    u_form = 'vector_advection_form' 
    theta_transport = EmbeddedDGOptions()
    transport_name = 'embedded'
    limited = False

elif config =='config5': # Vector advection embedded theta limited
    DGdegree = 1
    u_form = 'vector_advection_form'
    theta_transport = EmbeddedDGOptions()
    transport_name = 'embedded'
    limited = True
# Vector invariant options
elif config =='config6': # vector invariant SUPG 
    DGdegree = 1
    u_form = 'vector_invariant_form'
    theta_transport = SUPGOptions()
    transport_name = 'SUPG'
    limited = False

elif config =='config7': # vector invariant embedded not limited
    DGdegree = 1
    u_form = 'vector_invariant_form'
    theta_transport = EmbeddedDGOptions()
    transport_name = 'embedded'
    limited = False

elif config =='config8': # vector invariant embedded theta limited 
    DGdegree = 1
    u_form = 'vector_invariant_form'
    theta_transport = EmbeddedDGOptions()
    transport_name = 'embedded'
    limited = True
# -------------------------------------------------------------- #
# Script Options
# -------------------------------------------------------------- #

dirname = f'{config}_{perturbation}_wave_n={n}_dt={dt}'

if variable_height:
    dirname = f'{dirname}_varied_height'

if limited:
    dirname = f'{dirname}_limited' 
if not perturbed:
    dirname = f'SBR_{config}_{u_form}_{transport_name}'

# function for building lowest order function spaces
def buildUrecoverySpaces(mesh, degree):
    # horizontal base spaces
    hdiv_family = "RTCF" 
    hcurl_family = "RTCE"

    cell = mesh._base_mesh.ufl_cell().cellname()
    u_div_hori = FiniteElement(hdiv_family, cell, degree + 1)
    w_div_hori = FiniteElement("DG", cell, degree)
    u_curl_hori = FiniteElement(hcurl_family, cell, degree+1)
    w_curl_hori = FiniteElement("CG", cell, degree + 1)

    # vertical base spaces
    u_div_vert = FiniteElement("DG", interval, degree)
    w_div_vert = FiniteElement("CG", interval, degree + 1)
    u_curl_vert = FiniteElement("CG", interval, degree + 1)
    w_curl_vert = FiniteElement("DG", interval, degree)

    # build elements
    u_div_element = HDiv(TensorProductElement(u_div_hori, u_div_vert))
    w_div_element = HDiv(TensorProductElement(w_div_hori, w_div_vert))
    u_curl_element = HCurl(TensorProductElement(u_curl_hori, u_curl_vert))
    w_curl_element = HCurl(TensorProductElement(w_curl_hori, w_curl_vert))
    hdiv_element = u_div_element + w_div_element
    hcurl_element = u_curl_element + w_curl_element

    VHDiv = FunctionSpace(mesh, hdiv_element)
    VHCurl = FunctionSpace(mesh, hcurl_element)

    return VHDiv, VHCurl

# -------------------------------------------------------------- #
# Set up Model
# -------------------------------------------------------------- #
# Domain
a = 6.371229e6  # radius of earth
ztop = 3.0e4  # max height

if variable_height == True: 
    layerheight=[]
    runningheight=0
    # Calculating Non-uniform height field
    for m in range(1,nlayers+1):
        mu = 8
        height = ztop * ((mu * (m / nlayers)**2 + 1)**0.5 - 1) / ((mu + 1)**0.5 - 1)
        width = height - runningheight
        runningheight = height
        layerheight.append(width)
else: 
    layerheight = ztop / nlayers

m = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=n, degree=2)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=layerheight, extrusion_type='radial')
x ,y, z= SpatialCoordinate(mesh)
lat, lon, _ = lonlatr_from_xyz(x, y, z)
domain = Domain(mesh, dt, "RTCF", degree=DGdegree)

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector((0, 0, omega))
print('making eqn')    
eqn = CompressibleEulerEquations(domain, params, Omega=Omega, u_transport_option=u_form)
print(f'Number of DOFs = {eqn.X.function_space().dim()}')

output = OutputParameters(dirname=dirname,
                          dumpfreq=48, # every 3 hours
                          dump_nc=True,
                          dump_vtus=False)
diagnostic_fields = [MeridionalComponent('u'), ZonalComponent('u'),RadialComponent('u'),
                    CourantNumber(), Temperature(eqn), Gradient('Temperature'), Pressure(eqn),
                    SteadyStateError('Pressure_Vt'), CompressibleKineticEnergy(), PotentialEnergy(eqn)]
          
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport Schemes 

if DGdegree == 0:

    VDG1 = domain.spaces('DG1_equispaced')
    VCG1 = FunctionSpace(mesh, 'CG', 1)
    VHDiv1, VHcurl = buildUrecoverySpaces(mesh, 1)
    VHDiv0, VHcurl0 = buildUrecoverySpaces(mesh, 0)
    if limited:
        limiter = DG1Limiter(VDG1)
    else:
        limiter = None


    u_opts = RecoveryOptions(embedding_space=VHDiv1,
                            recovered_space=VHcurl0,
                            boundary_method=BoundaryMethod.hcurl, # try on 1 if doesnt work
                            injection_method = 'project',
                            project_high_method = 'project',
                            broken_method='project') 

    rho_opts = RecoveryOptions(embedding_space=VDG1,
                            recovered_space=VCG1,
                            boundary_method=BoundaryMethod.extruded)

    theta_opts = RecoveryOptions(embedding_space=VDG1,
                                recovered_space=VCG1)
    
    transported_fields = []
    transported_fields.append(SSPRK3(domain, "u", options=u_opts))
    transported_fields.append(SSPRK3(domain, "rho", options=rho_opts))
    transported_fields.append(SSPRK3(domain, "theta", options=theta_opts, limiter=limiter))

    transport_methods = [DGUpwind(eqn, 'u'),
                        DGUpwind(eqn, 'rho'),
                        DGUpwind(eqn, 'theta')]
else:
    if limited:
        Vtheta = domain.spaces("theta")
        limiter = ThetaLimiter(Vtheta)
    else:
        limiter = None

    if isinstance(theta_transport, SUPGOptions):
        theta_ibp = theta_transport.ibp
    else:
        theta_ibp = IntegrateByParts.ONCE
    
    transported_fields = []
    transport_methods = []
    if u_form == 'vector_invariant_form':
        transported_fields.append(TrapeziumRule(domain, "u"))
        transport_methods.append(DGUpwind(eqn, 'u'))
    else:
        transported_fields.append(TrapeziumRule(domain, "u", options=SUPGOptions()))
        transport_methods.append(DGUpwind(eqn, 'u', ibp=SUPGOptions().ibp))
    
    transported_fields.append(SSPRK3(domain, "rho"))
    transported_fields.append(SSPRK3(domain, "theta", options=theta_transport, limiter=limiter))

    transport_methods.append(DGUpwind(eqn, 'rho'))
    transport_methods.append(DGUpwind(eqn, 'theta', ibp=theta_ibp))

# Linear Solver
linear_solver = CompressibleSolver(eqn, alpha=alpha)

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
                                  transport_methods,
                                  linear_solver=linear_solver, alpha=alpha)

# -------------------------------------------------------------- #
# Initial Conditions
# -------------------------------------------------------------- #

x, y, z = SpatialCoordinate(mesh)
lat, lon, _ = lonlatr_from_xyz(x, y, z)
r = sqrt(x**2 + y**2 + z**2)
l = sqrt(x**2 + y**2)
unsafe_x = x / l
unsafe_y = y / l
safe_x = min_value(max_value(unsafe_x, -1), 1)
safe_y = min_value(max_value(unsafe_y, -1), 1)

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


approx_wind = ((g*k) / (2 * omega * a)) * (cos(lat)**(k-1) - cos(lat)**(k+1))*tao2_int*Temp

theta_expr = Temp * (P_expr / p0) ** (-params.kappa) 
pie_expr = Temp / theta_expr
rho_expr = P_expr / (Rd * Temp)

# -------------------------------------------------------------- #
# Perturbation
# -------------------------------------------------------------- #


def VelocityPerturbation(base_state, location, mesh, Vp=1):
    '''
    A function which applies a velocity perturbation in the zonal and merirdioanl velocity 
    fields
    args:
        wind: The current velocity state : type: Tuple of zonal, meridional and radial fields
        location: Location of perturbation: type: tuple of co-ordinates in lat / lon
        mesh: requires the mesh to calculate lat, lon, r
        r: Location vector
    Optional: 
        Vp: Maximum velocity of perturbation
    '''

    x, y, z = SpatialCoordinate(mesh)
    lat, lon, _ = lonlatr_from_xyz(x, y, z)
    r = sqrt(x**2 + y**2 + z**2)
    a = 6.371229e6
    zt = 1.5e4     # top of perturbation
    d0 = a / 6     # horizontal radius of perturbation
    zonal_u, merid_u, _ = base_state
    a = 6.371229e6   
    err_tol = 1e-12
    
    for co_ords in location:
        lon_c , lat_c = co_ords # location of perturbation centre
        d = a * acos(sin(lat_c)*sin(lat) + cos(lat_c)*cos(lat)*cos(lon - lon_c)) # distance from centre of perturbation

        depth = r - a # The distance from origin subtracted from earth radius
        zeta = conditional(ge(depth,zt-err_tol), 0, 1 - 3*(depth / zt)**2 + 2*(depth / zt)**3) # peturbation vertical taper

        perturb_magnitude = (16*Vp/(3*sqrt(3))) * zeta * sin((pi * d) / (2 * d0)) * cos((pi * d) / (2 * d0)) ** 3


        zonal_pert = conditional(le(d,err_tol), 0, 
                                conditional(ge(d,(d0-err_tol)), 0, -perturb_magnitude * (-sin(lat_c)*cos(lat) + cos(lat_c)*sin(lat)*cos(lon - lon_c)) / sin(d / a)))
        meridional_pert = conditional(le(d,err_tol), 0, 
                                    conditional(ge(d,d0-err_tol), 0, perturb_magnitude * cos(lat_c)*sin(lon - lon_c) / sin(d / a)))

        zonal_u = zonal_u + zonal_pert
        merid_u = merid_u + meridional_pert
    return (zonal_u, merid_u)

# -------------------------------------------------------------- #
# Configuring fields
# -------------------------------------------------------------- #
# get components of u in spherical polar coordinates

zonal_u = wind 
merid_u = Constant(0.0) 
radial_u = Constant(0.0)

base_state = (zonal_u, merid_u, radial_u)

if perturbed == True:
    zonal_u, merid_u, = VelocityPerturbation(base_state, location, mesh)

(u_expr, v_expr, w_expr) = xyz_from_lonlatr(zonal_u, merid_u, radial_u)


# obtain initial conditions
print('Set up initial conditions')
print('project u')
test_u = TestFunction(Vu)
dx_reduced = dx(degree=4)
u_field = as_vector([u_expr, v_expr, w_expr])
u_proj_eqn = inner(test_u,u0 - u_field)*dx_reduced
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
print('Intialise Windy Boi')
stepper.run(t=0, tmax=tmax)
