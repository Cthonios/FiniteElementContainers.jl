using Exodus
using FiniteElementContainers
using StaticArrays
using Tensors

# mesh file
gold_file = "./mechanics/mechanics.gold"
mesh_file = "./mechanics/mechanics.g"
output_file = "./mechanics/mechanics.e"

fixed(_, _) = 0.
displace(_, t) = 1.e-3 * t

include("TestMechanicsCommon.jl")

function mechanics_test()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Mechanics(PlaneStrain())

  u = VectorFunction(V, :displ)
  asm = SparseMatrixAssembler(H1Field, u)

  dbcs = DirichletBC[
    DirichletBC(:displ_x, :sset_3, fixed),
    DirichletBC(:displ_y, :sset_3, fixed),
    DirichletBC(:displ_x, :sset_1, fixed),
    DirichletBC(:displ_y, :sset_1, displace),
  ]

  # pre-setup some scratch arrays
  times = TimeStepper(0., 1., 1)
  p = create_parameters(asm, physics; dirichlet_bcs=dbcs, times=times)

  solver = NewtonSolver(IterativeLinearSolver(asm, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  # write_field(pp, 1, U)
  close(pp)
end

mechanics_test()
