using FiniteElementContainers
using LinearAlgebra

# physics code we actually need to implement
struct Poisson{F <: Function} <: AbstractPhysics{1, 0, 0}
    func::F
end

@inline function FiniteElementContainers.residual(
    physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    ∇u_q = interpolate_field_gradients(physics, interps, u_el)
    ∇u_q = unpack_field(∇u_q, 1)
    R_q = ∇N_X * ∇u_q - N * physics.func(X_q, 0.0)
    return JxW * R_q[:]
end
  
@inline function FiniteElementContainers.stiffness(
    physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    K_q = ∇N_X * ∇N_X'
    return JxW * K_q
end

# define some helper funcs upfront
f(_, _) = 0.0
one_func(_, _) = 1.0
zero_func(_, _) = 0.0

function simulate()
    # load a mesh
    mesh = UnstructuredMesh("mug.e")
    V = FunctionSpace(mesh, H1Field, Lagrange)
    physics = Poisson(f)
    props = create_properties(physics)
    u = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(u; use_condensed=true)
    dbcs = [
        DirichletBC(:u, one_func; sideset_name = :bottom)
        DirichletBC(:u, zero_func; sideset_name = :top)
    ]
    U = create_field(asm)
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)

    # solver = NewtonSolver(DirectLinearSolver(asm))
    # integrator = QuasiStaticIntegrator(solver)
    # evolve!(integrator, p)

    FiniteElementContainers.update_bc_values!(p)
    residual_int = FiniteElementContainers.VectorCellIntegral(asm, residual)
    stiffness_int = FiniteElementContainers.MatrixCellIntegral(asm, stiffness)

    K = stiffness_int(U, p)
    FiniteElementContainers.remove_fixed_dofs!(stiffness_int)

    for n in 1:2
        R = residual_int(U, p)
        FiniteElementContainers.remove_fixed_dofs!(residual_int)
        ΔU = -K \ R.data
        U.data .+= ΔU
        @show n norm(R) norm(ΔU)
    end

    U = p.h1_field

    pp = PostProcessor(mesh, "output.e", u)
    write_times(pp, 1, 0.0)
    write_field(pp, 1, ("u",), U)
    close(pp)
end

simulate()
