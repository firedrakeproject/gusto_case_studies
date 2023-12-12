"""
The non-linear gravity wave test case of Skamarock and Klemp (1994).

Potential temperature is transported using SUPG.
"""

from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
from gusto import *
import itertools
from firedrake import (as_vector, SpatialCoordinate, PeriodicIntervalMesh,
                       ExtrudedMesh, exp, sin, Function, pi, COMM_WORLD)
import numpy as np
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

dt = 6.
L = 3.0e5  # Domain length
H = 1.0e4  # Height position of the model top

if '--running-tests' in sys.argv:
    nlayers = 5
    columns = 30
    tmax = dt
    dumpfreq = 1
else:
    nlayers = 10
    columns = 150
    tmax = 3600
    dumpfreq = int(tmax / (2*dt))

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #

# Domain -- 3D volume mesh
degrees = [(0, 0), (0,1), (1,0), (1,1)]
for degree in degrees:
    h_degree = degree[0]
    v_degree = degree[1]
    m = PeriodicIntervalMesh(columns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    domain = Domain(mesh, dt, "CG", 
                    horizontal_degree=h_degree, 
		            vertical_degree=v_degree)

    # Equation
    Tsurf = 300.
    parameters = CompressibleParameters()
    eqns = CompressibleEulerEquations(domain, parameters)
    print(f'Ideal number of cores = {eqns.X.function_space().dim() / 50000} ')

    # I/O
    points_x = np.linspace(0., L, 100)
    points_z = [H/2.]
    points = np.array([p for p in itertools.product(points_x, points_z)])
    dirname = f'skamarock_klemp_nonlinear_h_order={h_degree}_v_order={v_degree}'
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

    VDG1 = domain.spaces("DG1_equispaced")
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    u_opts = RecoveryOptions(embedding_space=Vu_DG1,
				             recovered_space=Vu_CG1)
    rho_opts = RecoveryOptions(embedding_space=VDG1,
				               recovered_space=VCG1)
    
    if v_degree == 0:
        u_opts = RecoveryOptions(embedding_space=Vu_DG1,
				                 recovered_space=Vu_CG1,
				                 boundary_method=BoundaryMethod.taylor)
        rho_opts = RecoveryOptions(embedding_space=VDG1,
                                   recovered_space=VCG1,
                                   boundary_method=BoundaryMethod.taylor)

    theta_opts = RecoveryOptions(embedding_space=VDG1,
				                 recovered_space=VCG1)
    transported_fields = [TrapeziumRule(domain, "u", options=u_opts),
                        SSPRK3(domain, "rho", options=rho_opts),
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
