"""
The non-linear gravity wave test case of Skamarock and Klemp (1994).

Potential temperature is transported using SUPG.
"""

from petsc4py import PETSc
from pyrsistent import v
PETSc.Sys.popErrorHandler()
from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, exp, sin, Function, pi, COMM_WORLD)
import numpy as np
import icecream as ic
import sys

def lowestorderspaces(domain, BC=None):
    # creates lowest order spaces
    mesh=domain.mesh
    VDG1 = domain.spaces("DG1_equispaced")
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    u_opts = RecoveryOptions(embedding_space=Vu_DG1,
				             recovered_space=Vu_CG1,
                             boundary_method=BC)
    rho_opts = RecoveryOptions(embedding_space=VDG1,
				               recovered_space=VCG1,
                               boundary_method=BC)
    theta_opts = RecoveryOptions(embedding_space=VDG1,
				                 recovered_space=VCG1)
    
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
degrees = [(1, 1), (1,2), (2,1), (2,2)]
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
                              dump_vtus = False)
    
    diagnostic_fields = [CourantNumber(), ZonalComponent('u'), MeridionalComponent('u'),
                         RadialComponent('u'), Perturbation('theta'), Perturbation('rho'),
                         CompressibleKineticEnergy('u'), PotentialEnergy(eqns), InternalEnergy(eqns),
                         RichardsonNumber('theta', parameters.g/Tsurf)]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    # Transport options
    if degree ==(0,0):
        u_opts, rho_opts, theta_opts = lowestorderspaces(domain)

    if v_degree == 0:
       u_opts, rho_opts, theta_opts = lowestorderspaces(domain, BoundaryMethod.taylor)
   

    theta_opts = EmbeddedDGOptions()
    transported_fields = [SSPRK3(domain, "u"),
                            SSPRK3(domain, "rho"),
                            SSPRK3(domain, "theta", options=theta_opts)]
    transport_methods = [DGUpwind(eqns, "u"),
                            DGUpwind(eqns, "rho"),
                            DGUpwind(eqns, "theta", ibp=theta_opts.ibp)]

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
