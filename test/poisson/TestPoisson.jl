using Adapt
using AMDGPU
using Exodus
using FiniteElementContainers
using StaticArrays
using Test

# mesh file
gold_file = Base.source_dir() * "/poisson.gold"
mesh_file = Base.source_dir() * "/poisson.g"
output_file = Base.source_dir() * "/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
# f(_, _) = 1.
bc_func(_, _) = 0.
bc_func_neumann(_, _) = SVector{1, Float64}(1.)

include("TestPoissonCommon.jl")

# read mesh and relevant quantities

function test_poisson_dirichlet(
  dev, use_condensed,
  nsolver, lsolver
)
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(u; use_condensed=use_condensed)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]

  # setup the parameters
  p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)

  if dev != cpu
    p = p |> dev
    asm = asm |> dev 
  end

  # setup solver and integrator
  solver = nsolver(lsolver(asm))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  if dev != cpu
    p = p |> cpu
  end

  U = p.h1_field

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, ("u",), U)
  close(pp)

  if !Sys.iswindows()
    @test exodiff(output_file, gold_file)
  end
  rm(output_file; force=true)
  display(solver.timer)
end

function test_poisson_neumann(
  dev, use_condensed,
  nsolver, lsolver
)
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(u; use_condensed=use_condensed)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func)
  ]

  nbcs = NeumannBC[
    NeumannBC(:u, :sset_3, bc_func_neumann),
    NeumannBC(:u, :sset_4, bc_func_neumann)
  ]

  # direct solver test
  # setup the parameters
  p = create_parameters(
    mesh, asm, physics, props; 
    dirichlet_bcs=dbcs,
    neumann_bcs=nbcs
  )

  if dev != cpu
    p = p |> dev
    asm = asm |> dev 
  end

  # setup solver and integrator
  solver = nsolver(lsolver(asm))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  if dev != cpu
    p = p |> cpu
  end

  # TODO make a neumann gold file
  # U = p.h1_field

  # pp = PostProcessor(mesh, output_file, u)
  # write_times(pp, 1, 0.0)
  # write_field(pp, 1, ("u",), U)
  # close(pp)

  # if !Sys.iswindows()
  #   @test exodiff(output_file, gold_file)
  # end
  # rm(output_file; force=true)
  display(solver.timer)
end

function test_poisson_dirichlet_multi_block_quad4_quad4(
  dev, use_condensed,
  nsolver, lsolver
)
  mesh = UnstructuredMesh(Base.source_dir() * "/poisson/multi_block_mesh_quad4_quad4.g")
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(u; use_condensed=use_condensed)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :boundary, bc_func)
  ]

  # setup the parameters
  p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)

  if dev != cpu
    p = p |> dev
    asm = asm |> dev 
  end

  # setup solver and integrator
  solver = nsolver(lsolver(asm))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  if dev != cpu
    p = p |> cpu
  end

  U = p.h1_field

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, ("u",), U)
  close(pp)

  # if !Sys.iswindows()
  #   @test exodiff(output_file, gold_file)
  # end
  rm(output_file; force=true)
  display(solver.timer)
end

function test_poisson_dirichlet_multi_block_quad4_tri3(
  dev, use_condensed,
  nsolver, lsolver
)
  mesh = UnstructuredMesh(Base.source_dir() * "/poisson/multi_block_mesh_quad4_tri3.g")
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(u; use_condensed=use_condensed)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :boundary, bc_func)
  ]

  # setup the parameters
  p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)

  if dev != cpu
    p = p |> dev
    asm = asm |> dev 
  end

  # setup solver and integrator
  solver = nsolver(lsolver(asm))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  if dev != cpu
    p = p |> cpu
  end

  U = p.h1_field

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, ("u",), U)
  close(pp)

  # if !Sys.iswindows()
  #   @test exodiff(output_file, gold_file)
  # end
  rm(output_file; force=true)
  display(solver.timer)
end
