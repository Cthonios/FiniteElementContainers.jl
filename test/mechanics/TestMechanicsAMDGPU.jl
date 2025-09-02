using Adapt
using AMDGPU
using Exodus
using FiniteElementContainers
using StaticArrays
using Tensors

# mesh file
gold_file = Base.source_dir() * "/mechanics.gold"
mesh_file = Base.source_dir() * "/mechanics.g"
output_file = Base.source_dir() * "/mechanics.e"

fixed(_, _) = 0.
displace(_, t) = 1.e-3 * t

function mechanics_test()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Mechanics(PlaneStrain())
  props = create_properties(physics)

  u = VectorFunction(V, :displ)
  dof = DofManager(u)
  asm = SparseMatrixAssembler(u)

  dbcs = DirichletBC[
    DirichletBC(:displ_x, :sset_3, fixed),
    DirichletBC(:displ_y, :sset_3, fixed),
    DirichletBC(:displ_x, :sset_1, fixed),
    DirichletBC(:displ_y, :sset_1, displace),
  ]

  # pre-setup some scratch arrays
  times = TimeStepper(0., 1., 1)
  p = create_parameters(asm, physics, props; dirichlet_bcs=dbcs, times=times)

  # move to device
  p_gpu = p |> rocm
  asm_gpu = asm |> rocm

  solver = NewtonSolver(IterativeLinearSolver(asm_gpu, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p_gpu)

  p = p_gpu |> cpu

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, ("displ_x", "displ_y"), p.h1_field)
  close(pp)
end

@time mechanics_test()
@time mechanics_test()
