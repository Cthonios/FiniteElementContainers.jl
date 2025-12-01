f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_func(_, _) = 0.
# TODO this creates a warning during testing due to
# reinclude
include("poisson/TestPoissonCommon.jl") 

function test_sparse_matrix_assembler()
  # create very simple poisson problem
  mesh_file = "./poisson/poisson.g"
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = SVector{0, Float64}()
  u = ScalarFunction(V, :u)
  @show asm = SparseMatrixAssembler(u)
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]
  p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)
  FiniteElementContainers.initialize!(p)
  Uu = create_unknowns(asm)
  Vu = create_unknowns(asm)

  assemble_scalar!(asm, energy, Uu, p)
  assemble_mass!(asm, mass, Uu, p)
  assemble_stiffness!(asm, stiffness, Uu, p)
  assemble_matrix_action!(asm, stiffness, Uu, Vu, p)
  assemble_vector!(asm, residual, Uu, p)

  K = stiffness(asm)
  M = mass(asm)
  R = residual(asm)
  Mv = hvp(asm, Vu)

  asm = SparseMatrixAssembler(u; use_condensed=false)
  p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)
  U = create_unknowns(asm)
  V = create_unknowns(asm)

  assemble_mass!(asm, mass, U, p)
  assemble_stiffness!(asm, stiffness, U, p)
  assemble_matrix_action!(asm, stiffness, U, V, p)
  assemble_vector!(asm, residual, U, p)

  K = stiffness(asm)
  M = mass(asm)
  R = residual(asm)
  Kv = hvp(asm, V)

  # mainly just to test the show method for parameters
  @show p
end

@testset "Sparse matrix assembler" begin
  test_sparse_matrix_assembler()
end
