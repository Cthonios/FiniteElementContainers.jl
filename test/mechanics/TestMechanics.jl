using Exodus
using FiniteElementContainers
using Krylov
using Parameters
using Tensors

# mesh file
gold_file = "./mechanics/mechanics.gold"
mesh_file = "./mechanics/mechanics.g"
output_file = "./mechanics/mechanics.e"

fixed(_, _) = 0.
displace(_, _) = 1.e-3

struct Mechanics{Form} <: AbstractPhysics{0, 0}
  formulation::Form
end

function FiniteElementContainers.residual(physics::Mechanics, cell, u_el, args...)
  @unpack X_q, N, ∇N_X, JxW = cell
  G_q = discrete_gradient(physics.formulation, ∇N_X)

  # props
  # TODO eventually add in args

  ∇u_q = u_el * ∇N_X
  ∇u_q = Tensor{3, 3, Float64, 9}(∇u_q...)
  ε_q = symmetric(∇u_q)

  σ_q = K * tr(ε_q) * I + 2. * G * dcontract(dev(ε_q))
  f_q = tovoigt!(σ_q)
end
