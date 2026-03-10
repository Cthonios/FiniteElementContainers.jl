using FiniteElementContainers

mesh_file = Base.source_dir() * "/poisson.g"
output_file = Base.source_dir() * "/output.e"

f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(2π * X[2])
bc_func(_, _) = 0.

struct Poisson{F <: Function} <: AbstractPhysics{1, 0, 0}
    func::F
end

@inline function FiniteElementContainers.residual(
    physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
  )
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    ∇u_q = interpolate_field_gradients(physics, interps, u_el)
    R_q = ∇u_q * ∇N_X' - N' * physics.func(X_q, 0.0)
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

mesh = UnstructuredMesh(mesh_file)
V = FunctionSpace(mesh, H1Field, Lagrange) 
physics = Poisson(f)
props = create_properties(physics)
u = ScalarFunction(V, :u)
asm = SparseMatrixAssembler(u; use_condensed=true)

dbcs = DirichletBC[
    DirichletBC(:u, bc_func; nodeset_name = :nset_1),
    DirichletBC(:u, bc_func; nodeset_name = :nset_2),
    DirichletBC(:u, bc_func; nodeset_name = :nset_3),
    DirichletBC(:u, bc_func; nodeset_name = :nset_4),
]

p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)
solver = NewtonSolver(DirectLinearSolver(asm))
integrator = QuasiStaticIntegrator(solver)
evolve!(integrator, p)

U = p.h1_field

pp = PostProcessor(mesh, output_file, u)
write_times(pp, 1, 0.0)
write_field(pp, 1, ("u",), U)
close(pp)
