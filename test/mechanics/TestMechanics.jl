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

function test_mechanics_dirichlet_only(
  dev, use_condensed,
  nsolver, lsolver
)
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Mechanics(PlaneStrain())
  props = create_properties(physics)

  u = VectorFunction(V, :displ)
  asm = SparseMatrixAssembler(u; use_condensed=use_condensed)

  dbcs = DirichletBC[
    DirichletBC(:displ_x, fixed; sideset_name = :sset_3),
    DirichletBC(:displ_y, fixed; sideset_name = :sset_3),
    DirichletBC(:displ_x, fixed; sideset_name = :sset_1),
    DirichletBC(:displ_y, displace; sideset_name = :sset_1),
  ]

  # pre-setup some scratch arrays
  times = TimeStepper(0., 1., 1)
  p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, times=times)

  if dev != cpu
    p = p |> dev
    asm = asm |> dev 
  end

  solver = nsolver(lsolver(asm))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  if dev != cpu
    p = p |> cpu
  end

  U = p.h1_field

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, ("displ_x", "displ_y"), U)
  # write_field(pp, 1, U)
  close(pp)
end
