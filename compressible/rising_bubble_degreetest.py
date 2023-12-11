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

dt = 0.1
L = 1000.
H = 1000.
tmax = 600.
dumpfreq = int(tmax / (60*dt))
res = 5.
nlayers = int(H / res)
ncolumns = int(L  /res)

# ---------------------------------------------------------------------------- #
# Set up model objects
# ---------------------------------------------------------------------------- #
degrees = [(0,1), (1,0), (1,1)]
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
	dirname = f'RB_horiz={h_degree}_vertical={v_degree}_res={res}'
	output = OutputParameters(dirname=dirname,
				  dumpfreq=dumpfreq,
				  dumplist=['u'],
				  dump_nc = True,
				  dump_vtus = False)
	diagnostic_fields = [CourantNumber(), Perturbation('theta'), Perturbation('rho')]
	io = IO(domain, output, diagnostic_fields=diagnostic_fields)

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

	transported_fields = []
	transported_fields.append(SSPRK3(domain, "u", options=u_opts))
	transported_fields.append(SSPRK3(domain, "rho", options=rho_opts))
	transported_fields.append(SSPRK3(domain, "theta", options=theta_opts))

	transport_methods = [DGUpwind(eqn, 'u'),
			    DGUpwind(eqn, 'rho'),
			    DGUpwind(eqn, 'theta')]

	# Transport schemes
	theta_opts = EmbeddedDGOptions()
	transported_fields = [TrapeziumRule(domain, "u"),
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
