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

struct Mechanics{Form} <: AbstractPhysics{2, 0, 0}
  formulation::Form
end

function FiniteElementContainers.residual(physics::Mechanics, cell, u_el, args...)
  @unpack X_q, N, ∇N_X, JxW = cell

  # props
  # TODO eventually add in args

  ∇u_q = u_el * ∇N_X
  ∇u_q = Tensor{3, 3, Float64, 9}(∇u_q...)
  F_q = ∇u_q + one(typeof(∇u_q))
  ε_q = symmetric(∇u_q)

  σ_q = K * tr(ε_q) * I + 2. * G * dcontract(dev(ε_q))
  # Piola transformation
  P_q = det(F_q) * σ_q * inv(F_q)'
  # turn into voigt notation
  P_q = extract_stress(physics.formulation, P_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  return G_q * P_q
end
