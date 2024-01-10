"""
The non-linear gravity wave test case of Skamarock and Klemp (1994).

Potential temperature is transported using SUPG.
"""
from petsc4py import PETSc
from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh, TensorProductElement,
                       ExtrudedMesh, exp, sin, Function, pi, FiniteElement, cell, interval)
import numpy as np
import icecream as ic


def RecoverySpaces(mesh, vertical_degree, horizontal_degree, BC=None):
    # scaler spaces
    DG_vert_ele = FiniteElement('DG', interval, vertical_degree+1, variant='equispaced')
    DG_hori_ele = FiniteElement('DG', cell, horizontal_degree+1, variant='equispaced')
    CG_hori_ele = FiniteElement('CG', cell, vertical_degree+1)
    CG_vert_ele = FiniteElement('CG', interval, vertical_degree+1)

    VDG_ele = TensorProductElement(DG_hori_ele, DG_vert_ele)
    VCG_ele = TensorProductElement(CG_hori_ele, CG_vert_ele)
    VDG = FunctionSpace(mesh, VDG_ele)
    VCG = FunctionSpace(mesh, VCG_ele)

    # vector_spaces
    DG_hori_vec_ele = FiniteElement('DG', cell, horizontal_degree+2)
    DG_vert_vec_ele = FiniteElement('DG', interval, horizontal_degree+2)
    CG_hori_vec_ele = FiniteElement('hdiv', cell, horizontal_degree+2)
    CG_vert_vec_ele = FiniteElement('hdiv', interval, horizontal_degree+2)

    VDG_Vec_ele = TensorProductElement(DG_hori_vec_ele, DG_vert_vec_ele)
    VCG_Vec_ele = TensorProductElement(CG_hori_vec_ele, CG_vert_vec_ele)

    Vu_DG = VectorFunctionSpace(mesh, VDG_Vec_ele)
    Vu_CG = VectorFunctionSpace(mesh, VCG_Vec_ele)

    u_opts = RecoveryOptions(embedding_space=Vu_DG,
                             recovered_space=Vu_CG,
                             boundary_method=BC)
    rho_opts = RecoveryOptions(embedding_space=VDG,
                               recovered_space=VCG,
                               boundary_method=BC)
    theta_opts = RecoveryOptions(embedding_space=VDG,
                                 recovered_space=VCG)

    return u_opts, rho_opts, theta_opts


# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 6.
L = 3.0e5  # Domain length
H = 1.0e4  # Height position of the model top

nlayers = 10
columns = 150
tmax = 3600
dumpfreq = int(tmax / (2*dt))

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain -- 3D volume mesh
degrees = [(0, 0), (1, 0), (0, 1)]
for degree in degrees:
    h_degree = degree[0]
    v_degree = degree[1]
    nlayers = 10
    columns = 150
    if h_degree > 1:
        columns = columns / 2
    if v_degree > 1:
        nlayers = nlayers / 2
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG",
                    horizontal_degree=h_degree,
                    vertical_degree=v_degree)
    # Equation
    Tsurf = 300.
    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)
    ic(eqns.X.function_space().dim())
    print(f'Ideal number of cores = {eqns.X.function_space().dim() / 50000}')

    # I/O
    points_x = np.linspace(0., L, 100)
    points_z = [H/2.]
    points = np.array([p for p in itertools.product(points_x, points_z)])
    dirname = f'gravwave={h_degree}_v_order={v_degree}'
    output = OutputParameters(dirname=dirname,
                              dumpfreq=dumpfreq,
                              dumplist=['u'],
                              dump_nc=True,
                              dump_vtus=False)

    diagnostic_fields = [CourantNumber(), XComponent('u'), YComponent('u'), ZComponent('u'),
                         Perturbation('theta'), Perturbation('rho')]

    io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    # Transport options
    recovery_spaces = ((0, 0), (0, 1), (1, 0))

    if degree in recovery_spaces:
        if v_degree == 0:
            u_opts, rho_opts, theta_opts = RecoverySpaces(mesh, v_degree, h_degree, BC=BoundaryMethod.taylor)
        else:
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
