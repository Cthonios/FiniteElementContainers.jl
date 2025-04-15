using Exodus
using FiniteElementContainers
using Parameters
using Tensors

mesh_file = "./test/dynamics/dynamics.g"
output_file = "./test/dynamics/dynamics.e"

fixed(_, _) = 0.
displace(_, _) = 1.e-3

struct Dynamics{Form} <: AbstractPhysics{2, 0, 0}
  formulation::Form
end

function strain_energy(∇u)
  K = 10.e9
  G = 1.e9
  ε = symmetric(∇u)
  ψ = 0.5 * K * tr(ε)^2 + G * dcontract(dev(ε), dev(ε))
end

function FiniteElementContainers.mass(physics::Dynamics, cell, u_el, args...)
  @unpack X_q, N, ∇N_X, JxW = cell
  ρ = 1.e3
  N = discrete_values(physics.formulation, N)
  return ρ * N * N'
end

function FiniteElementContainers.residual(physics::Dynamics, cell, u_el, args...)
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
end

function FiniteElementContainers.stiffness(physics::Dynamics, cell, u_el, args...)
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
physics = Dynamics(PlaneStrain())

u = VectorFunction(V1, :displacement)
v = VectorFunction(V1, :velocity)
a = VectorFunction(V1, :acceleration)

dof = DofManager(u) # only need to add u to dofmanager since v, a have the same pattern and are not the primary solution fields
asm = SparseMatrixAssembler(dof, H1Field)

dbcs = DirichletBC[
  DirichletBC(asm.dof, :displacement_x, :sset_3, fixed),
  DirichletBC(asm.dof, :displacement_y, :sset_3, fixed),
  DirichletBC(asm.dof, :displacement_x, :sset_1, fixed),
  DirichletBC(asm.dof, :displacement_y, :sset_1, displace),
]
update_dofs!(asm, dbcs)

# create unknown arrays
Uu = create_unknowns(asm, H1Field)
Vu = create_unknowns(asm, H1Field)
Au = create_unknowns(asm, H1Field)

# create bc arrays
Ubc = create_bcs(asm, H1Field)
Vbc = create_bcs(asm, H1Field)
Abc = create_bcs(asm, H1Field)

# create field arrays
U = create_field(asm, H1Field)
V = create_field(asm.dof, H1Field, (:velocity_x, :velocity_y))
A = create_field(asm.dof, H1Field, (:acceleration_x, :acceleration_y))

update_field!(U, asm, Uu, Ubc)
update_field_bcs!(U, asm.dof, dbcs, 0.)
# TODO add logic for velocity/acceleration bcs

assemble!(asm, physics, U, :mass)
assemble!(asm, physics, U, :residual_and_stiffness)

M = FiniteElementContainers.mass(asm)
K = stiffness(asm)

# # testing postprocessor
pp = PostProcessor(mesh, output_file, u, v, a)
write_times(pp, 1, 0.0)
write_field(pp, 1, U)
write_field(pp, 1, V)
write_field(pp, 1, A)
close(pp)