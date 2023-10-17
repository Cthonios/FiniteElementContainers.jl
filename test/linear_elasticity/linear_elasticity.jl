# remove this when done
# include("../test_helpers.jl")
# remove this when done

λ = 1.0
μ = 0.5
strain(∇u) = 0.5 * (∇u + ∇u')

# currently allocating a bunch
function pack_2x2_in_3x3(∇u)
  PaddedView(0.0, ∇u, (3, 3))
end

function energy(cell, u_el)
  @unpack X, N, ∇N_X, JxW = cell
  ∇u_q = ∇N_X' * u_el'
  ε_q  = strain(pack_2x2_in_3x3(∇u_q))
  ψ_q  = 0.5 * λ * tr(ε_q)^2 + μ * tr(ε_q * ε_q)
  return JxW * ψ_q
end

residual(cell, u_el) = ForwardDiff.gradient(z -> energy(cell, z), u_el)[:]
tangent(cell, u_el)  = ForwardDiff.hessian(z -> energy(cell, z), u_el)

q_degree = 2
mesh = read_mesh("./linear_elasticity/linear_elasticity.g", [1, 2, 3, 4])

bcs = EssentialBC[
  EssentialBC(mesh, 1, 1),
  EssentialBC(mesh, 1, 2, (x, t) -> 1.e-3),
  EssentialBC(mesh, 3, 1),
  EssentialBC(mesh, 3, 2)
]

fspaces, dof, asm = container_setup(mesh, [1], q_degree, 2)
U = simple_solver(mesh, fspaces, dof, asm, bcs, residual, tangent)
simple_post_processor("./linear_elasticity/linear_elasticity.g", U, ["displ_x", "displ_y"])
exodiff("linear_elasticity/linear_elasticity.gold", "linear_elasticity/linear_elasticity.e")
rm("linear_elasticity/linear_elasticity.e")
