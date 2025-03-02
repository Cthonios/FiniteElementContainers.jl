import KernelAbstractions as KA
using Adapt
using CUDA
using Exodus
using FiniteElementContainers
using LinearAlgebra
using Parameters
using ReferenceFiniteElements
using SparseArrays

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(4π * X[2])

function residual(cell, u_el)
  @unpack X_q, N, ∇N_X, JxW = cell
  ∇u_q = u_el * ∇N_X
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:]
end

function tangent(cell, u_el)
  @unpack X_q, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

# do all setup on CPU
# the mesh for instance is not gpu compatable
mesh = UnstructuredMesh("./test/poisson/poisson.g")
V = FunctionSpace(mesh, H1, Lagrange)
u = ScalarFunction(V, :u)
dof = NewDofManager(u)
asm = SparseMatrixAssembler(dof)

bc_nodes = sort!(unique!(vcat(values(mesh.nodeset_nodes)...)))

# TODO this one will be tought to do on the GPU
update_dofs!(asm, bc_nodes)

# TODO move some of these to after GPU movement
# to test kernels on GPU

asm_gpu = Adapt.adapt_structure(CuArray, asm)
# bc_nodes_gpu = Adapt.adapt_structure(CuArray, bc_nodes)

Uu = create_unknowns(asm_gpu)
Ubc = KA.zeros(KA.get_backend(Uu), Float64, length(bc_nodes))
U = create_field(asm_gpu, H1)
R = create_field(asm_gpu, H1)

update_field!(U, asm_gpu.dof, Uu, Ubc)
