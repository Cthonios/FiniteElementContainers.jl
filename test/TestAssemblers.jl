f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_func(_, _) = 0.
# TODO this creates a warning during testing due to
# reinclude
include("poisson/TestPoissonCommon.jl") 


@testset ExtendedTestSet "Sparse matrix assembler" begin
  # create very simple poisson problem
  mesh_file = "./poisson/poisson.g"
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = SVector{0, Float64}()
  u = ScalarFunction(V, :u)
  @show asm = SparseMatrixAssembler(H1Field, u)
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]
  p = create_parameters(asm, physics, props; dirichlet_bcs=dbcs)
  Uu = create_unknowns(asm, H1Field)
  Vu = create_unknowns(asm, H1Field)

  assemble_scalar!(asm, energy, Uu, p, H1Field)
  assemble_mass!(asm, mass, Uu, p, H1Field)
  assemble_stiffness!(asm, stiffness, Uu, p, H1Field)
  assemble_matrix_action!(asm, stiffness, Uu, Vu, p, H1Field)
  assemble_vector!(asm, residual, Uu, p, H1Field)

  K = stiffness(asm)
  M = mass(asm)
  R = residual(asm)
end
