"""
The Travelling Vortex test case in a three dimensional channel
"""

from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate, 
                       Function, conditional, sqrt, as_vector, atan2, FiniteElement, interval,
                       TensorProductElement, HCurl, HDiv, FunctionSpace)

def ConstRecoverySpaces(mesh, v_degree, h_degree):
    # rho space
    h_order_dic = {}
    v_order_dic = {}
    h_order_dic['rho'] = {0: 1, 1: 1}
    h_order_dic['theta'] = {0: 1, 1: 1}
    h_order_dic['u'] = {0: 1, 1: 1}
    v_order_dic['rho'] = {0: 1, 1: 1}
    v_order_dic['theta'] = {0: 2, 1: 2}
    v_order_dic['u'] = {0: 1, 1: 1}
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
    CG_vert_ele = FiniteElement('CG', interval, v_order_dic['theta'][v_degree])

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



tmax = 100
dt = 0.1
dumpfreq = int(tmax / (10*dt))
orders = [(0, 0), (0, 1), (1, 0), (1, 1)]
resolutions = [60, 80, 100, 120, 140]

uc = 100.
vc = 100.
L = 10000.
H = 10000.

for order in orders:
    for res in resolutions:
        nlayers = res
        ncolumns = res
        if order[0] == 1:
            ncolumns = res / 2
        if order[1] == 1:
            nlayers = res / 2

        m = PeriodicIntervalMesh(ncolumns, L)
        mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers, periodic=True)
        x, y = SpatialCoordinate(mesh)
        domain = Domain(mesh, dt, 'CG',  horizontal_degree=order[0], 
                        vertical_degree=order[1])

        # Equations
        params = CompressibleParameters(g=0)
        eqns = CompressibleEulerEquations(domain, params)
        eqns.bcs['u'] = []
        print(f'ideal number of processors = {eqns.X.function_space().dim() / 50000}')

        #I/O
        dirname=f'TV_chkpoint_order={order[0]}_{order[1]}_res={res}'
        output = OutputParameters(dirname=dirname, dumpfreq=dumpfreq, dump_nc=True, dump_vtus=False)
        diagnostics = [XComponent('u'), ZComponent('u'),  Pressure(eqns), SteadyStateError('theta'),
                    SteadyStateError('u')]
        io = IO(domain, output, diagnostic_fields=diagnostics)

        # Transport options
        recovery_spaces = ((0, 0), (0, 1), (1, 0))

        if order in recovery_spaces:
            u_opts, rho_opts, theta_opts = ConstRecoverySpaces(mesh, order[1], order[0])

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
        # solver
        linear_solver = CompressibleSolver(eqns)
        # timestepper
        stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields,
                                            transport_methods,
                                            linear_solver=linear_solver)
        # ------------------------------------------------------------------------------
        # Initial conditions
        # ------------------------------------------------------------------------------
        # initial fields
        u0 = stepper.fields('u')
        rho0 = stepper.fields('rho')
        theta0 = stepper.fields('theta')

        # spaces
        Vu = u0.function_space()
        Vt = theta0.function_space()
        Vr = rho0.function_space()

        # constants
        kappa = params.kappa
        p_0 = params.p_0
        xc = L / 2  
        yc = H / 2 

        R = 0.4 * L # radius of voertex
        r = sqrt((x - xc)**2 + (y - yc)**2) / R
        phi = atan2((y - yc),(x - xc))

        # rho expression
        rhoc = 1
        rhob = Function(Vr).interpolate(conditional(r>=1, rhoc,  1- 0.5*(1-r**2)**6))

        # u expression
        u_cond = (1024 * (1.0 - r)**6 * r**6 )
        ux_expr = conditional(r>=1, uc, uc + u_cond * (-(y-yc)/(r*R))) 
        uy_expr = conditional(r>=1, vc, vc + u_cond * (-(x-xc)/(r*R))) 
        u0.project(as_vector([ux_expr, uy_expr]))

        #pressure and theta expressions
        coe = np.zeros((25))
        coe[0]  =     1.0 / 24.0
        coe[1]  = -   6.0 / 13.0
        coe[2]  =    18.0 /  7.0
        coe[3]  = - 146.0 / 15.0
        coe[4]  =   219.0 / 8.0
        coe[5]  = - 966.0 / 17.0
        coe[6]  =   731.0 /  9.0
        coe[7]  = -1242.0 / 19.0
        coe[8]  = -  81.0 / 40.0
        coe[9]  =   64.
        coe[10] = - 477.0 / 11.0
        coe[11] = -1032.0 / 23.0
        coe[12] =   737.0 / 8.0
        coe[13] = - 204.0 /  5.0
        coe[14] = - 510.0 / 13.0
        coe[15] =  1564.0 / 27.0
        coe[16] = - 153.0 /  8.0
        coe[17] = - 450.0 / 29.0
        coe[18] =   269.0 / 15.0
        coe[19] = - 174.0 / 31.0
        coe[20] = -  57.0 / 32.0
        coe[21] =    74.0 / 33.0
        coe[22] = -  15.0 / 17.0
        coe[23] =     6.0 / 35.0
        coe[24] =  -  1.0 / 72.0

        p0 = 0
        for i in range(25):
            p0 += coe[i] * (r**(12+i)-1)
        mach = 0.341
        p = 1 + 1024**2 * mach**2 *conditional(r>=1, 0, p0)

        R0 = 287.
        gamma=1.4
        pref=params.p_0

        T = p / (rhob*R0)
        
        cg3 = FunctionSpace(mesh, "CG", 3 )


        theta_cg3 = Function(cg3).interpolate(T*(pref / p)**0.286)
        thetab = Function(Vt).project(theta_cg3)


        #compressible_hydrostatic_balance(eqns, theta, rho, solve_for_rho=True)

        # make mean fields
        print('make mean fields')
        theta0.assign(thetab)
        rho0.assign(rhob)

        # assign reference profiles
        stepper.set_reference_profiles([('rho', rhob),
                                        ('theta', thetab)])
        # ---------------------------------------------------------------------------- #
        # Run
        # ---------------------------------------------------------------------------- #

        stepper.run(t=0, tmax=tmax)