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
  assemble!(asm, Uu, p, Val{:energy}(), H1Field)
  assemble!(asm, Uu, p, Val{:gradient}(), H1Field)
  assemble!(asm, Uu, p, Vu, Val{:hvp}(), H1Field)
  assemble!(asm, Uu, p, Val{:mass}(), H1Field)
  assemble!(asm, Uu, p, Val{:residual}(), H1Field)
  assemble!(asm, Uu, p, Val{:stiffness}(), H1Field)
  assemble!(asm, Uu, p, Vu, Val{:stiffness_action}(), H1Field)

  assemble!(asm, Uu, p, :energy, H1Field)
  assemble!(asm, Uu, p, :gradient, H1Field)
  assemble!(asm, Uu, p, Vu, :hvp, H1Field)
  assemble!(asm, Uu, p, :mass, H1Field)
  assemble!(asm, Uu, p, :residual, H1Field)
  assemble!(asm, Uu, p, :stiffness, H1Field)
  assemble!(asm, Uu, p, Vu, :stiffness_action, H1Field)

  K = stiffness(asm)
  M = mass(asm)
  R = residual(asm)
end
