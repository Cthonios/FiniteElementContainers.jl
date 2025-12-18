using FiniteElementContainers
using ForwardDiff
using LinearAlgebra
using StaticArrays

struct CoupledPhysics <: AbstractPhysics{2, 0, 0}
end

@inline function FiniteElementContainers.residual(
    physics::CoupledPhysics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    ∇u_q = interpolate_field_gradients(physics, interps, u_el)
    ∇u = unpack_field(∇u_q, 1)
    ∇v = unpack_field(∇u_q, 2)
    term = dot(∇u, ∇v)
    R_u = ∇N_X * ∇u + term * N
    R_v = ∇N_X * ∇v
    R = vcat(R_u', R_v')
    return JxW * R[:]
end
  
@inline function FiniteElementContainers.stiffness(
    physics::CoupledPhysics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    # interps = map_interpolants(interps, x_el)
    # (; X_q, N, ∇N_X, JxW) = interps
    # ∇u_q = interpolate_field_gradients(physics, interps, u_el)
    # ∇u = unpack_field(∇u_q, 1)
    # ∇v = unpack_field(∇u_q, 2)

    # # term = ∇N_X * physics.v
    # term_1 = ∇N_X * ∇v
    # term_2 = ∇N_X * ∇v
    # K_uu = ∇N_X * ∇N_X' + N * term_1'
    # K_uv = N * term_2'
    # K_vu = zeros(typeof(K_uv))
    # K_vv = ∇N_X * ∇N_X'

    # # K_q = ∇N_X * ∇N_X' + N * term'
    # K = vcat(
    #     hcat(K_uu, K_uv),
    #     hcat(K_vu, K_vv)
    # )
    # return JxW * K
    ForwardDiff.jacobian(
        z -> FiniteElementContainers.residual(
            physics, interps, x_el, t, dt, z, u_el_old, state_old_q, state_new_q, props_el
        ), u_el
    )
end

# define some helper funcs upfront
one_func(_, _) = 1.0
two_func(_, _) = 2.0
zero_func(_, _) = 0.0

function simulate()
    # load a mesh
    mesh = UnstructuredMesh("mug.e")
    V = FunctionSpace(mesh, H1Field, Lagrange)
    physics = CoupledPhysics()
    props = create_properties(physics)
    u = FiniteElementContainers.GeneralFunction(
        ScalarFunction(V, :u),
        ScalarFunction(V, :v)
    )

    asm = SparseMatrixAssembler(u; use_condensed=true)
    dbcs = [
        DirichletBC(:u, :bottom, two_func)
        DirichletBC(:u, :top, zero_func)
        DirichletBC(:v, :bottom, one_func)
        DirichletBC(:v, :top, zero_func)
    ]
    U = create_field(asm)
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)

    solver = NewtonSolver(DirectLinearSolver(asm))
    integrator = QuasiStaticIntegrator(solver)
    evolve!(integrator, p)

    # FiniteElementContainers.update_bc_values!(p)
    # residual_int = VectorIntegral(asm, residual)
    # stiffness_int = MatrixIntegral(asm, stiffness)

    # K = stiffness_int(U, p)
    # remove_fixed_dofs!(stiffness_int)

    # for n in 1:4
    #     R = residual_int(U, p)
    #     remove_fixed_dofs!(residual_int)
    #     ΔU = -K \ R.data
    #     U.data .+= ΔU
    #     @show n norm(R) norm(ΔU)
    # end

    U = p.h1_field

    pp = PostProcessor(mesh, "output.e", u)
    write_times(pp, 1, 0.0)
    write_field(pp, 1, ("u", "v"), U)
    close(pp)
end

simulate()