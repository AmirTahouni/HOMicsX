import numpy as np
import ufl
from dolfinx import fem
import dolfinx_mpc

from config import *

#-------------------------------------------------| FEM |--------------------------------------------------#
def fem_solve(mesh, ct, ft, E_particle, E_matrix, nu_particle, nu_matrix):   
    vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=mesh)))

    def create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
        V0 = fem.functionspace(domain, ("DG", 0))
        k = fem.Function(V0, name=name)
        for tag, value in property_dict.items():
            cells = cell_markers.find(tag)
            k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
        return k

    E = create_piecewise_constant_field(
        mesh, ct, {1: E_matrix, 2: E_particle}, name="YoungModulus"
    )
    nu = create_piecewise_constant_field(
        mesh, ct, {1: nu_matrix, 2: nu_particle}, name="PoissonRatio"
    )

    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)


    Eps = fem.Constant(mesh, np.zeros((3, 3)))
    Eps_ = fem.Constant(mesh, np.zeros((3, 3)))
    y = ufl.SpatialCoordinate(mesh)


    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        eps = Eps + epsilon(v)
        return lmbda * ufl.tr(eps) * ufl.Identity(gdim) + 2 * mu * eps

    V = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    du = ufl.TrialFunction(V)
    u_ = ufl.TestFunction(V)
    a_form, L_form = ufl.system(ufl.inner(sigma(du), epsilon(u_)) * ufl.dx)


    bcs = []
    point_dof = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0) & np.isclose(x[2], 0)
    )
    bcs.append(fem.dirichletbc(np.zeros((gdim,)), point_dof, V))


    def periodic_relation_left_right(x):
        out_x = np.zeros(x.shape)
        out_x[0] = x[0] - 1
        out_x[1] = x[1] 
        out_x[2] = x[2]
        return out_x

    def periodic_relation_bottom_top(x):
        out_x = np.zeros(x.shape)
        out_x[0] = x[0]
        out_x[1] = x[1] - 1
        out_x[2] = x[2]
        return out_x

    def periodic_relation_near_far(x):
        out_x = np.zeros(x.shape)
        out_x[0] = x[0]
        out_x[1] = x[1]
        out_x[2] = x[2] - 1
        return out_x

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(
        V, ft, 4, periodic_relation_left_right, bcs
    )
    mpc.create_periodic_constraint_topological(
        V, ft, 6, periodic_relation_bottom_top, bcs
    )
    mpc.create_periodic_constraint_topological(
        V, ft, 8, periodic_relation_near_far, bcs
    )
    mpc.finalize()


    u = fem.Function(mpc.function_space, name="Displacement")
    v = fem.Function(mpc.function_space, name="Periodic_fluctuation")
    problem = dolfinx_mpc.LinearProblem(
        a_form,
        L_form,
        mpc,
        bcs=bcs,
        u=v,
        #petsc_options={"ksp_type": "cg", "-ksp_monitor_true_residual": "", "pc_type": "hypre", },
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", },
    )


    #problematic block
    # Define the 3D elementary strain tensors
    elementary_load = np.array([
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # Exx
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),  # Eyy
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),  # Ezz
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),  # Exy
        np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),  # Exz
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),  # Eyz
    ]) * 1.0

    load_labels = ["Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz"]
    dim_load = len(elementary_load)

    # Define the assembly and solver function
    def assemble_and_solve(a_form, L_form, problem, strain_tensor=None):
        """
        This function assembles the system once and solves for each strain tensor.
        """
        # Update the strain tensor if provided (this affects the RHS)
        if strain_tensor is not None:
            Eps.value = strain_tensor
            u.interpolate(
                fem.Expression(
                    ufl.dot(Eps, y), mpc.function_space.element.interpolation_points()
                )
            )
        
        # Solve the problem (the assembly happens once, the RHS changes)
        problem.solve()
        
        return u.x.array  # The displacement array

    # Initialize the global stiffness matrix (assemble once)
    C_hom = np.zeros((dim_load, dim_load))

    # Iterate over all the elementary strain tensors
    for nload in range(dim_load):
        strain_tensor = elementary_load[nload]  # Get the strain tensor
        displacement_array = assemble_and_solve(a_form, L_form, problem, strain_tensor)

        # Now assemble the matrix using the displacement
        for nload_ in range(dim_load):
            strain_tensor_ = elementary_load[nload_]
            Eps_.value = strain_tensor_
            
            # Compute the component of the stiffness matrix (C_hom)
            C_hom[nload, nload_] = (
                fem.assemble_scalar(fem.form(ufl.inner(sigma(v), Eps_) * ufl.dx)) / vol
            )
        print(f"Load case {load_labels[nload]} solved.")
    
    return C_hom






