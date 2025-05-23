using Exodus
using FiniteElementContainers
using Krylov
using Parameters
using Tensors

# mesh file
gold_file = "./test/mechanics_with_state/mechanics_with_state.gold"
mesh_file = "./test/mechanics_with_state/mechanics_with_state.g"
output_file = "./test/mechanics_with_state/mechanics_with_state.e"

fixed(_, _) = 0.
displace(_, _) = 1.e-3

struct Mechanics{Form} <: AbstractPhysics{2, 4, 7}
  formulation::Form
end

function strain_energy(∇u)
  K = 10.e9
  G = 1.e9
  ε = symmetric(∇u)
  ψ = 0.5 * K * tr(ε)^2 + G * dcontract(dev(ε), dev(ε))
end

function FiniteElementContainers.residual(physics::Mechanics, cell, u_el, args...)
  @unpack X_q, N, ∇N_X, JxW = cell

  # kinematics
  ∇u_q = u_el * ∇N_X
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  P_q = gradient(strain_energy, ∇u_q)
  # turn into voigt notation
  P_q = extract_stress(physics.formulation, P_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  f_q = G_q * P_q
  return f_q[:]
  # return f_q
end

function FiniteElementContainers.stiffness(physics::Mechanics, cell, u_el, args...)
  @unpack X_q, N, ∇N_X, JxW = cell

  # kinematics
  ∇u_q = u_el * ∇N_X
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  K_q = hessian(z -> strain_energy(z), ∇u_q)
  # turn into voigt notation
  K_q = extract_stiffness(physics.formulation, K_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  return G_q * K_q * G_q'
end

mesh = UnstructuredMesh(mesh_file)
V1 = FunctionSpace(mesh, H1Field, Lagrange) 
V2 = FunctionSpace(mesh, L2QuadratureField, Lagrange)

physics = Mechanics(PlaneStrain())

u = VectorFunction(V1, :displ)

# this styel will only work for simple problems
# maybe we need an "active blocks" thing
ε_p = SymmetricTensorFunction(V2, :plastic_strain)
eqps = ScalarFunction(V2, :eqps)

dof = DofManager(u, ε_p, eqps)

state = create_field(dof, L2QuadratureField)
# state = StateFunction(V2, :state, 7, 4)
# dof = DofManager(u, state)
asm = SparseMatrixAssembler(dof, H1Field)

dbcs = DirichletBC[
  DirichletBC(asm.dof, :displ_x, :sset_3, fixed),
  DirichletBC(asm.dof, :displ_y, :sset_3, fixed),
  DirichletBC(asm.dof, :displ_x, :sset_1, fixed),
  DirichletBC(asm.dof, :displ_y, :sset_1, displace),
]
update_dofs!(asm, dbcs)

# # pre-setup some scratch arrays
Uu = create_unknowns(asm, H1Field)
Ubc = create_bcs(asm, H1Field)
U = create_field(asm, H1Field)

update_field!(U, asm, Uu, Ubc)
update_field_bcs!(U, asm.dof, dbcs, 0.)

@time assemble!(asm, physics, U, :residual_and_stiffness)
K = stiffness(asm)

for n in 1:5
  Ru = residual(asm)
  ΔUu, stat = cg(-K, Ru)
  update_field_unknowns!(U, asm.dof, ΔUu, +)
  assemble!(asm, physics, U, :residual)

  @show norm(ΔUu) norm(Ru)

  # if norm(ΔUu) < 1e-12 || norm(Ru) < 1e-12
  if norm(Ru) < 1e-12
    break
  end
end

pp = PostProcessor(mesh, output_file, u, ε_p)
write_times(pp, 1, 0.0)
write_field(pp, 1, U)
close(pp)
