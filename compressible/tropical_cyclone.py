from firedrake import (ExtrudedMesh, SpatialCoordinate, cos, sin, pi, sqrt,
                       exp, Constant, Function, as_vector, inner,
                       errornorm, norm, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, FiniteElement, HDiv, HCurl,
                       TensorProductElement, interval, VertexBasedLimiter,
                       max_value)
from gusto import *
import gusto.thermodynamics as tde
import sys

pick_up = ('--pick_up' in sys.argv)

# ---------------------------------------------------------------------------- #
# Script Options
# ---------------------------------------------------------------------------- #
element_degree = 1
vector_invariant = False
variable_height = False
limit_theta = False

# ---------------------------------------------------------------------------- #
# Test case Parameters
# ---------------------------------------------------------------------------- #
dt = 600.
days = 10.
tmax = days * 24. * 60. * 60.
dumpfreq = int(tmax / (2*days*dt))
ncells = 11
nlayers = 5
# For short simulations
tmax = 24*60*60
dumpfreq = int(tmax / (24*dt))
ncells = 25
nlayers = 15
# ---------------------------------------------------------------------------- #
# Generate directory name to capture parameters
# ---------------------------------------------------------------------------- #
dirname = f'tropical_cyclone_dt{dt:.0f}_n{ncells}_degree{element_degree}'

if limit_theta:
    dirname = f'{dirname}_theta_limited'
if vector_invariant:
    u_transport_option = 'vector_invariant'
    dirname += '_vector_invariant'
else:
    u_transport_option = 'vector_advection_form'
    dirname += '_vector_advective'
if variable_height:
    dirname += '_variable_height'

output = OutputParameters(dirname=dirname,
                          dumpfreq=dumpfreq,
                          chkptfreq=dumpfreq,
                          dump_nc=True,
                          dump_vtus=False)

mesh_name = 'gusto_mesh'

# ---------------------------------------------------------------------------- #
# Routine to make HDiv/HCurl spaces for velocity recovery
# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #
# Set up Model
# ---------------------------------------------------------------------------- #
# Domain
a = 6.371229e6  # radius of earth
ztop = 3.0e4  # max height

if variable_height == True:
    dirname = f'{dirname}variable_height_'
    layerheight=[]
    runningheight=0
    # Calculating Non-uniform height field
    for n in range(1,16):
        mu = 8
        height = ztop * ((mu * (n / 15)**2 + 1)**0.5 - 1) / ((mu + 1)**0.5 - 1)
        width = height - runningheight
        runningheight = height
        layerheight.append(width)
else:
    layerheight = ztop / nlayers

if pick_up:
    mesh = pick_up_mesh(output, mesh_name)
else:
    m = GeneralCubedSphereMesh(a, num_cells_per_edge_of_panel=ncells, degree=2)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=layerheight, extrusion_type='radial', name=mesh_name)

xyz = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, "RTCF", degree=element_degree)

Vu = domain.spaces('HDiv')
Vr = domain.spaces('DG')
Vt = domain.spaces('theta')

# Equations
params = CompressibleParameters()
omega = Constant(7.292e-5)
Omega = as_vector((0, 0, omega))
active_tracers = [WaterVapour(), CloudWater(), Rain()]
eqn = CompressibleEulerEquations(domain, params, Omega=Omega,
                                 u_transport_option=u_transport_option,
                                 active_tracers=active_tracers)

diagnostic_fields = [MeridionalComponent('u', space=Vr),
                     ZonalComponent('u', space=Vr),
                     RadialComponent('u', space=Vr),
                     CourantNumber(),
                     Temperature(eqn, space=Vt),
                     Pressure(eqn, space=Vt),
                     Perturbation('Pressure_Vt'),
                     Theta_d(eqn, space=Vt)]

io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Transport options
transport_schemes = []
transport_methods = []

# u transport
if element_degree == 0:
    assert not vector_invariant, 'Cannot run with vector invariant and lowest-order spaces'
    VHDiv1, VHcurl = buildUrecoverySpaces(mesh, 1)
    VHDiv0, VHcurl0 = buildUrecoverySpaces(mesh, 0)

    u_opts = RecoveryOptions(embedding_space=VHDiv1,
                            recovered_space=VHcurl0,
                            boundary_method=BoundaryMethod.hcurl,
                            injection_method = 'project',
                            project_high_method = 'project',
                            broken_method='project')

    transport_schemes.append(TrapeziumRule(domain, "u", options=u_opts))
    transport_methods.append(DGUpwind(eqn, "u"))
else:
    # SUPG, needs IBP specifying
    u_opts = SUPGOptions()
    transport_schemes.append(TrapeziumRule(domain, "u", options=u_opts))
    transport_methods.append(DGUpwind(eqn, "u", ibp=u_opts.ibp))

# rho transport
if element_degree == 0:
    VDG1 = domain.spaces('DG1_equispaced')
    VCG1 = FunctionSpace(mesh, 'CG', 1)

    rho_opts = RecoveryOptions(embedding_space=VDG1,
                               recovered_space=VCG1,
                               boundary_method=BoundaryMethod.extruded)
else:
    rho_opts = None

transport_schemes.append(SSPRK3(domain, "rho", options=rho_opts))
transport_methods.append(DGUpwind(eqn, "rho"))

# theta transport
if element_degree == 0:
    theta_opts = RecoveryOptions(embedding_space=VDG1,
                                 recovered_space=VCG1)
    if limit_theta:
        limiter = VertexBasedLimiter(VDG1)
    else:
        limiter = None
    moisture_limiter = VertexBasedLimiter(VDG1)
else:
    theta_opts = EmbeddedDGOptions()
    if limit_theta:
        limiter = ThetaLimiter(Vt)
    else:
        limiter = None
    moisture_limiter = ThetaLimiter(Vt)

transport_methods.append(DGUpwind(eqn, "theta"))
transport_methods.append(DGUpwind(eqn, "water_vapour"))
transport_methods.append(DGUpwind(eqn, "cloud_water"))
transport_methods.append(DGUpwind(eqn, "rain"))
transport_schemes.append(SSPRK3(domain, "theta", options=theta_opts, limiter=limiter))
transport_schemes.append(SSPRK3(domain, "water_vapour", options=theta_opts, limiter=moisture_limiter))
transport_schemes.append(SSPRK3(domain, "cloud_water", options=theta_opts, limiter=moisture_limiter))
transport_schemes.append(SSPRK3(domain, "rain", options=theta_opts, limiter=moisture_limiter))

# Linear Solver
linear_solver = CompressibleSolver(eqn)

# Physics
T_surf = Constant(302.15)
rainfall_method = DGUpwind(eqn, 'rain', outflow=True)
physics_schemes = [(Fallout(eqn, 'rain', domain, rainfall_method, moments=AdvectedMoments.M0), SSPRK3(domain)),
                   (Coalescence(eqn), ForwardEuler(domain)),
                   (EvaporationOfRain(eqn), ForwardEuler(domain)),
                   (SaturationAdjustment(eqn), ForwardEuler(domain)),
                   (SurfaceFluxes(eqn, T_surf, 'water_vapour', implicit_formulation=True), ForwardEuler(domain)),
                   (WindDrag(eqn, implicit_formulation=True), ForwardEuler(domain))]

# Time Stepper
stepper = SemiImplicitQuasiNewton(eqn, io, transport_schemes, transport_methods,
                                  linear_solver=linear_solver, physics_schemes=physics_schemes)

# ---------------------------------------------------------------------------- #
# Initial Conditions
# ---------------------------------------------------------------------------- #

if not pick_up:
    # set up parameters
    Rd = params.R_d
    Rv = params.R_v
    cp = params.cp
    g = params.g
    p0 = Constant(100000)
    kappa = params.kappa

    X = 1.0        # Small-planet scaling factor (regular-size Earth)
    zt = 15000.    # Tropopause height
    q0 = 0.021     # Maximum specific humidity amplitude kg/kg
    qt = 1e-11     # Specific humidity in the upper atmosphere
    T0 = 302.15    # Surface temperature of the air
    Ts = 302.15    # Sea surface temperature
    zq1 = 3000.    # Height related to linear decrease of q with height
    zq2 = 8000.    # Height related to quadratic decrease of q with height
    Gamma = 0.007  # Virtual temperature lapse rate
    pb = 101500.   # Background surface pressure
    lat_c = pi/18  # Initial latitude of vortex centre
    lon_c = 0      # Initial longitude of vortex centre
    deltap = 1115. # Pressure perturbation at vortex center
    rp = 282000.   # Horizontal half-width of pressure perturbation
    zp = 7000.     # Height related to the vertical decay of pressure perturbation
    eps = 1e-25    # Small value threshold


    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")
    vapour0 = stepper.fields("water_vapour")

    # ------------------------------------------------------------------------ #
    # Base State
    # ------------------------------------------------------------------------ #

    lon, lat, radial_coord = lonlatr_from_xyz(xyz[0], xyz[1], xyz[2])
    z = radial_coord - a

    # Specific humidity
    q_expr = conditional(z < zt, q0*exp(-z/zq1)*exp(-(z/zq2)**2), qt)
    mv_expr = q_expr / (1 - q_expr)
    # Virtual temperature
    mv0 = q0/(1-q0)
    Tv0 = T0*(1 + Rv*mv0/Rd)
    Tvt = Tv0 - Gamma*zt
    Tvd_bar_expr = conditional(z < zt, Tv0 - Gamma*z, Tvt)
    # Pressure
    pt = pb*(Tvt/Tv0)**(g/(Rd*Gamma))
    p_bar_expr = conditional(z < zt, pb*((Tv0 - Gamma*z)/Tv0)**(g/(Rd*Gamma)),
                            pt*exp(g*(zt-z)/(Rd*Tvt)))

    # Background state fields
    # Set background pressure to use in diagnostics
    pressure_b_Vt = stepper.fields('Pressure_Vt_bar', space=Vt, dump=True, pick_up=True)
    exner_b = Function(Vr)
    rho_b = Function(Vr)
    thetavd_b = Function(Vt)
    rho_b_Vt = Function(Vt)
    boundary_method = BoundaryMethod.extruded if element_degree == 0 else None
    rho_recoverer = Recoverer(rho_b, rho_b_Vt, boundary_method=boundary_method)

    # Evaluate background states
    thetavd_bar_expr = Tvd_bar_expr * (p0 / p_bar_expr) ** kappa
    rho_bar_expr = p_bar_expr / (Rd*Tvd_bar_expr)
    vapour0.interpolate(mv_expr)
    thetavd_b.interpolate(thetavd_bar_expr)
    # First guesses for rho and exner for hydrostatic balance solve
    rho_b.interpolate(rho_bar_expr)
    exner_b.interpolate(tde.exner_pressure(params, rho_b, thetavd_b))
    # Ensure base state is in numerical hydrostatic balance
    print('Compressible hydrostatic balance solve')
    compressible_hydrostatic_balance(eqn, thetavd_b, rho_b, exner0=exner_b,
                                    exner_boundary=exner_b, mr_t=vapour0,
                                    solve_for_rho=True)
    # Back out background pressure from prognostic variables
    rho_recoverer.project()
    exner_bar_expr = tde.exner_pressure(params, rho_b_Vt, thetavd_b)
    pressure_b_Vt.interpolate(tde.p(params, exner_bar_expr))

    # ------------------------------------------------------------------------ #
    # Perturbation
    # ------------------------------------------------------------------------ #

    r = a * great_arc_angle(lon, lat, lon_c, lat_c)

    # Pressure perturbation
    p_pert_expr = conditional(z > zt, Constant(0.0),
                            -deltap*exp(-(r/rp)**1.5-(z/zp)**2)*((Tv0-Gamma*z)/Tv0)**(g/(Rd*Gamma)))
    p_expr = p_bar_expr + p_pert_expr
    ### TODO: to remove (this diagnostic)
    pressure_pert = stepper.fields('Pressure_Vt_raw_pert', space=Vr, dump=True, pick_up=True)
    pressure_pert.interpolate(p_pert_expr)

    # Temperature perturbation
    Tvd_pert_expr = conditional(z > zt, Constant(0.0),
                            (Tv0 - Gamma*z)*(-1 + 1 / (1 + 2*Rd*(Tv0-Gamma*z)*z
                                                            /(g*zp**2*(1 - pb/deltap*exp((r/rp)**1.5 + (z/zp)**2))))))
    Tvd_expr = Tvd_bar_expr + Tvd_pert_expr

    # Wind field
    fc = 2*omega*sin(lat_c)
    tangent_u = conditional(z > zt, Constant(0.0),
                            -fc*r/2 + sqrt((fc*r/2)**2
                                        - 1.5*(r/rp)**1.5*(Tv0 - Gamma*z)*Rd
                                        / (1 + 2*Rd*(Tv0-Gamma*z)*z/(g*zp**2)
                                            - pb/deltap*exp((r/rp)**1.5+(z/zp)**2))))

    d1 = sin(lat_c)*cos(lat) - cos(lat_c)*sin(lat)*cos(lon-lon_c)
    d2 = cos(lat_c)*sin(lon-lon_c)
    d = max_value(eps, sqrt(d1**2 + d2**2))
    zonal_u = tangent_u*d1/d
    meridional_u = tangent_u*d2/d

    # Scalar prognostics: perturbations are made to base state
    thetavd_expr = Tvd_expr * (p0 / p_expr) ** kappa
    thetavd_pert_expr = thetavd_expr - thetavd_bar_expr
    rho_expr = p_expr / (Rd*Tvd_expr)
    rho_pert_expr = rho_expr - rho_bar_expr

    # ------------------------------------------------------------------------ #
    # Configuring fields
    # ------------------------------------------------------------------------ #

    e_lon_tuple = xyz_vector_from_lonlatr([Constant(1.0), Constant(0.0), Constant(0.0)], xyz)
    e_lat_tuple = xyz_vector_from_lonlatr([Constant(0.0), Constant(1.0), Constant(0.0)], xyz)
    e_lon = as_vector(e_lon_tuple)
    e_lat = as_vector(e_lat_tuple)

    # obtain initial conditions
    print('Set up initial conditions')
    print('project u')
    test_u = TestFunction(Vu)
    dx_reduced = dx(degree=4)
    u_proj_eqn = inner(test_u, u0 - zonal_u*e_lon - meridional_u*e_lat)*dx_reduced
    u_proj_prob = NonlinearVariationalProblem(u_proj_eqn, u0)
    u_proj_solver = NonlinearVariationalSolver(u_proj_prob)
    u_proj_solver.solve()
    print('interpolate theta')
    # Generate initial conditions for prognostics by adding perturbations to numerical base state
    theta0.interpolate(thetavd_b + thetavd_pert_expr)
    print('find rho')
    rho0.interpolate(rho_b + rho_pert_expr)

    print('make analytic rho')
    rho_analytic = Function(Vr).interpolate(rho_expr)
    print('Normalised rho error is:', errornorm(rho_analytic, rho0) / norm(rho_analytic))

    # assign reference profiles
    stepper.set_reference_profiles([('rho', rho_b),
                                    ('theta', thetavd_b)])

# ---------------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------------- #

stepper.run(t=0, tmax=tmax, pick_up=pick_up)
