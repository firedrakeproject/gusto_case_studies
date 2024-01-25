"""
The non-linear gravity wave test case of Skamarock and Klemp (1994).

Potential temperature is transported using SUPG.
"""
from petsc4py import PETSc
from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh, TensorProductElement,
                       ExtrudedMesh, exp, sin, Function, pi, FiniteElement, quadrilateral, interval, HCurl, HDiv)
import numpy as np
import icecream as ic

def MinimumRecoverySpaces(mesh, v_degree, h_degree, BC=None):
    # rho space
    h_order_dic = {}
    v_order_dic = {}
    h_order_dic['rho'] = {0: '1', 1: '1'}
    h_order_dic['theta'] = {0: '1', 1: '1'}
    h_order_dic['u'] = {0: '1', 1: '1'}
    v_order_dic['rho'] = {0: '1', 1: '1'}
    v_order_dic['theta'] = {0: '1', 1: '2'}
    v_order_dic['u'] = {0: '1', 1: '1'}
    cell = mesh._base_mesh.ufl_cell().cellname()

    # rho space
    DG_hori_ele = FiniteElement('DG', cell, h_order_dic['rho'][h_degree], variant='equispaced')
    DG_vert_ele = FiniteElement('DG', interval, v_order_dic['rho'][v_degree], variant='equispaced')
    CG_hori_ele = FiniteElement('CG', cell, h_order_dic['rho'][h_degree])
    CG_vert_ele = FiniteElement('CG', interval, v_order_dic['rho'][v_degree])

    VDG_ele = TensorProductElement(DG_hori_ele, DG_vert_ele)
    VCG_ele = TensorProductElement(CG_hori_ele, CG_vert_ele)
    VDG_rho= FunctionSpace(mesh, VDG_ele)
    VCG_rho = FunctionSpace(mesh, VCG_ele)

    # theta space
    DG_hori_ele = FiniteElement('DG', cell, h_order_dic['theta'][h_degree], variant='equispaced')
    DG_vert_ele = FiniteElement('DG', interval, v_order_dic['theta'][v_degree], variant='equispaced')
    CG_hori_ele = FiniteElement('CG', cell, h_order_dic['theta'][h_degree])
    CG_vert_ele = FiniteElement('CG', interval, v_order_dic['thea'][v_degree])

    VDG_ele = TensorProductElement(DG_hori_ele, DG_vert_ele)
    VCG_ele = TensorProductElement(CG_hori_ele, CG_vert_ele)
    VDG_theta= FunctionSpace(mesh, VDG_ele)
    VCG_theta = FunctionSpace(mesh, VCG_ele)

    # VR space for u transport
    Vrh_hori_ele = FiniteElement('DG', cell, h_order_dic['u'][h_degree])
    Vrh_vert_ele = FiniteElement('CG', interval, v_order_dic['u'][v_degree]+1)

    Vrv_hori_ele = FiniteElement('CG', cell, h_order_dic['u'][h_degree]+1)
    Vrv_vert_ele = FiniteElement('DG', interval, v_order_dic['u'][v_degree])

    Vrh_ele = HCurl(TensorProductElement(Vrh_hori_ele, Vrh_vert_ele))
    Vrv_ele = HCurl(TensorProductElement(Vrv_hori_ele, Vrv_vert_ele))

    Vrh_ele = Vrh_ele + Vrv_ele
    Vu_VR = FunctionSpace(mesh, Vrh_ele)

    # Vh space for u transport
    VHh_hori_ele = FiniteElement('CG', cell, h_order_dic['u'][h_degree]+1)
    VHh_vert_ele = FiniteElement('DG', interval,  v_order_dic['u'][v_degree])

    VHv_hori_ele = FiniteElement('DG', cell, h_order_dic['u'][h_degree])
    VHv_vert_ele = FiniteElement('CG', interval, v_order_dic['u'][v_degree]+1)

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
    rho_opts = RecoveryOptions(embedding_space=VDG_rho,
                               recovered_space=VCG_rho,
                               )
    theta_opts = RecoveryOptions(embedding_space=VDG_theta,
                                 recovered_space=VCG_theta)

    return u_opts, rho_opts, theta_opts

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


# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 1.2
L = 3.0e5  # Domain length
H = 1.0e4  # Height position of the model top
tmax = 3600
dumpfreq = int(tmax / (4*dt))

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain -- 3D volume mesh
degrees = [(0, 0), (1, 0), (0, 1), (1, 1)]
for degree in degrees:
    h_degree = degree[0]
    v_degree = degree[1]
    dx = 200
    dz = 200
    if v_degree == 0:
       dz = dz / 2
    if h_degree ==0:
       dx = dx / 2
    nlayers = int(H / dz)
    columns = int(L / dx)
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG",
                    horizontal_degree=h_degree,
                    vertical_degree=v_degree)
    # Equation
    Tsurf = 300.
    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)
    print(f'Ideal number of cores = {eqns.X.function_space().dim() / 50000}')

    # I/O
    dirname = f'gravwave_HighRes_h={h_degree}_v={v_degree}'
    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist=['u'],
                              dump_nc=True,
                              dump_vtus=False)

    diagnostic_fields = [CourantNumber(), XComponent('u'), ZComponent('u'),
                         Perturbation('theta'), Perturbation('rho')]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    # Transport options
    recovery_spaces = ((0, 0), (0, 1), (1, 0))

    if degree in recovery_spaces:
        u_opts, rho_opts, theta_opts = RecoverySpaces(mesh, v_degree, h_degree)

        transported_fields = [SSPRK3(domain, "u", options=u_opts),
                              SSPRK3(domain, "rho", options=rho_opts),
                              SSPRK3(domain, "theta", options=theta_opts)]
        transport_methods = [DGUpwind(eqns, "u"),
                             DGUpwind(eqns, "rho"),
                             DGUpwind(eqns, "theta")]
    else:

        theta_opts = EmbeddedDGOptions()
        transported_fields = [SSPRK3(domain, "u"),
                              SSPRK3(domain, "rho"),
                              SSPRK3(domain, "theta", options=theta_opts)]
        transport_methods = [DGUpwind(eqns, "u"),
                             DGUpwind(eqns, "rho"),
                             DGUpwind(eqns, "theta")]

    # Linear solver
    linear_solver = CompressibleSolver(eqns)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
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

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g
    N = parameters.N

    x, z = SpatialCoordinate(mesh)

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic exner
    compressible_hydrostatic_balance(eqns, theta_b, rho_b)

    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = deltaTheta*sin(pi*z/H)/(1 + (x - L/2)**2/a**2)
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([20.0, 0.0]))

    stepper.set_reference_profiles([('rho', rho_b),
                                    ('theta', theta_b)])

    # ---------------------------------------------------------------------------- #
    # Run
    # ---------------------------------------------------------------------------- #

    stepper.run(t=0, tmax=tmax)
