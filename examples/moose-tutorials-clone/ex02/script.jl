using FiniteElementContainers
using LinearAlgebra
using StaticArrays

# physics code we actually need to implement
struct Advection{N} <: AbstractPhysics{1, 0, 0}
    v::SVector{N, Float64}
end

@inline function FiniteElementContainers.residual(
    physics::Advection, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    ∇u_q = interpolate_field_gradients(physics, interps, u_el)
    ∇u_q = unpack_field(∇u_q, 1)
    term = dot(∇u_q, physics.v)
    R_q = ∇N_X * ∇u_q + term * N
    return JxW * R_q[:]
end
  
@inline function FiniteElementContainers.stiffness(
    physics::Advection, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    term = ∇N_X * physics.v
    K_q = ∇N_X * ∇N_X' + N * term'
    return JxW * K_q
end

# define some helper funcs upfront
one_func(_, _) = 1.0
zero_func(_, _) = 0.0

function simulate()
    # load a mesh
    mesh = UnstructuredMesh("mug.e")
    V = FunctionSpace(mesh, H1Field, Lagrange)
    physics = Advection(SVector{3, Float64}(0., 0., 1.))
    props = create_properties(physics)
    u = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(u; use_condensed=true)
    dbcs = [
        DirichletBC(:u, :bottom, one_func)
        DirichletBC(:u, :top, zero_func)
    ]
    U = create_field(asm)
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)

    # solver = NewtonSolver(DirectLinearSolver(asm))
    # integrator = QuasiStaticIntegrator(solver)
    # evolve!(integrator, p)

    FiniteElementContainers.update_bc_values!(p)
    residual_int = VectorIntegral(asm, residual)
    stiffness_int = MatrixIntegral(asm, stiffness)

    K = stiffness_int(U, p)
    remove_fixed_dofs!(stiffness_int)

    for n in 1:2
        R = residual_int(U, p)
        remove_fixed_dofs!(residual_int)
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
