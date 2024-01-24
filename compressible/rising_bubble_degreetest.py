"""
The dry rising bubble test case of Robert (1993).

Potential temperature is transported using the embedded DG technique.
"""
from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       Constant, pi, cos, Function, sqrt, conditional, FiniteElement,
                       HDiv, HCurl, interval, TensorProductElement)
import sys
# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #
def RecoverySpaces(mesh, vertical_degree, horizontal_degree, BC=None):
    # rho space
    cell = mesh._base_mesh.ufl_cell().cellname()
    DG_hori_ele = FiniteElement('DG', cell, horizontal_degree+1, variant='equispaced')
    DG_vert_ele = FiniteElement('DG', interval, vertical_degree+1, variant='equispaced')
    CG_hori_ele = FiniteElement('CG', cell, horizontal_degree+1)
    CG_vert_ele = FiniteElement('CG', interval, vertical_degree+1)

    VDG_ele = TensorProductElement(DG_hori_ele, DG_vert_ele)
    VCG_ele = TensorProductElement(CG_hori_ele, CG_vert_ele)
    VDG = FunctionSpace(mesh, VDG_ele)
    VCG = FunctionSpace(mesh, VCG_ele)

    # VR space for u transport
    Vrh_hori_ele = FiniteElement('DG', cell, horizontal_degree+1)
    Vrh_vert_ele = FiniteElement('CG', interval, vertical_degree+2)

    Vrv_hori_ele = FiniteElement('CG', cell, horizontal_degree+2)
    Vrv_vert_ele = FiniteElement('DG', interval, horizontal_degree+1)

    Vrh_ele = HCurl(TensorProductElement(Vrh_hori_ele, Vrh_vert_ele))
    Vrv_ele = HCurl(TensorProductElement(Vrv_hori_ele, Vrv_vert_ele))

    Vrh_ele = Vrh_ele + Vrv_ele
    Vu_VR = FunctionSpace(mesh, Vrh_ele)

    # Vh space for u transport
    VHh_hori_ele = FiniteElement('CG', cell, horizontal_degree+2)
    VHh_vert_ele = FiniteElement('DG', interval, vertical_degree+1)

    VHv_hori_ele = FiniteElement('DG', cell, horizontal_degree+1)
    VHv_vert_ele = FiniteElement('CG', interval, horizontal_degree+2)

    VHh_ele = HDiv(TensorProductElement(VHh_hori_ele, VHh_vert_ele))
    VHv_ele = HDiv(TensorProductElement(VHv_hori_ele, VHv_vert_ele))

    VHh_ele = VHh_ele + VHv_ele
    Vu_VH = FunctionSpace(mesh, VHh_ele)


    u_opts = RecoveryOptions(embedding_space=Vu_VH,
                             recovered_space=Vu_VR,
                             injection_method='recover',
                             project_high_method='project',
                             project_low_method='project',
                             broken_method='project'
                             )
    rho_opts = RecoveryOptions(embedding_space=VDG,
                               recovered_space=VCG,
                               )
    theta_opts = RecoveryOptions(embedding_space=VDG,
                                 recovered_space=VCG)

    return u_opts, rho_opts, theta_opts


dt = 1
L = 1000.
H = 1000.
tmax = 600.
dumpfreq = int(tmax / (60*dt))
dz = 50.
dx = 50.
nlayers = int(H / dz)
ncolumns = int(L / dx)

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #
degrees = [(0,0), (0,1), (1,0), (1,1)]
for degree in degrees:
	# Domain
    h_degree = degree[0]
    v_degree = degree[1]
    if v_degree == 0:
       nlayers = nlayers * 2 
    if h_degree ==0:
       ncoloumns = ncolumns * 2
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG", 
        			horizontal_degree=h_degree, 
		            vertical_degree=v_degree)

	# Equation
    parameters = CompressibleParameters()
    eqn = CompressibleEulerEquations(domain, parameters)
	# I/O
    dirname = f'RB_horiz={h_degree}_vertical={v_degree}_fixed'
    output = OutputParameters(dirname=dirname,
				  dumpfreq=dumpfreq,
				  dumplist=['u'],
				  dump_nc = True,
				  dump_vtus = False)
    diagnostic_fields = [CourantNumber(), Perturbation('theta'), Perturbation('rho')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    recovery_spaces = ((0, 0), (0, 1), (1, 0))

    if degree in recovery_spaces:
        u_opts, rho_opts, theta_opts = RecoverySpaces(mesh, v_degree, h_degree)

        transported_fields = [SSPRK3(domain, "u", options=u_opts),
                              SSPRK3(domain, "rho", options=rho_opts),
                              SSPRK3(domain, "theta", options=theta_opts)]
        transport_methods = [DGUpwind(eqn, "u"),
                             DGUpwind(eqn, "rho"),
                             DGUpwind(eqn, "theta")]
    else:

        theta_opts = EmbeddedDGOptions()
        transported_fields = [SSPRK3(domain, "u"),
                              SSPRK3(domain, "rho"),
                              SSPRK3(domain, "theta", options=theta_opts)]
        transport_methods = [DGUpwind(eqn, "u"),
                             DGUpwind(eqn, "rho"),
                             DGUpwind(eqn, "theta")]

    # Linear solver
    linear_solver = CompressibleSolver(eqn)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqn, io, transported_fields,
					transport_methods,
					linear_solver=linear_solver)

    # ---------------------------------------------------------------------------- #
    # Initial conditions
    # ---------------------------------------------------------------------------- #

    u0 = stepper.fields("u")
    rho0 = stepper.fields("rho")
    theta0 = stepper.fields("theta")

    # spaces
    Vu = domain.spaces("HDiv")
    Vt = domain.spaces("theta")
    Vr = domain.spaces("DG")

    # Isentropic background state
    Tsurf = Constant(300.)

    theta_b = Function(Vt).interpolate(Tsurf)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqn, theta_b, rho_b, solve_for_rho=True)

    x = SpatialCoordinate(mesh)
    xc = 500.
    zc = 350.
    rc = 250.
    r = sqrt((x[0]-xc)**2 + (x[1]-zc)**2)
    theta_pert = conditional(r > rc, 0., 0.25*(1. + cos((pi/rc)*r)))

    theta0.interpolate(theta_b + theta_pert)
    rho0.interpolate(rho_b)

    stepper.set_reference_profiles([('rho', rho_b),
	     			                ('theta', theta_b)])

    # ---------------------------------------------------------------------------- #
    # Run
    # ---------------------------------------------------------------------------- #

    stepper.run(t=0, tmax=tmax)
