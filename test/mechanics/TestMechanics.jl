@testitem "Regression test - mechanics" begin
  if "--test-amdgpu" in ARGS @eval using AMDGPU end
  if "--test-cuda" in ARGS @eval using CUDA end
  using StaticArrays
  using Tensors
  include("TestMechanicsCommon.jl")
  include("../TestUtils.jl")
  backends = _get_backends()

  # mesh file
  gold_file = Base.source_dir() * "/mechanics.gold"
  # mesh_file = Base.source_dir() * "/mechanics.g"
  mesh_file = Base.source_dir() * "/mechanics_coarse.g"
  output_file = Base.source_dir() * "/mechanics.e"

  fixed(_, _) = 0.
  displace(_, t) = 1.e-3 * t

# function test_mechanics_dirichlet_only(
#   dev, nsolver, lsolver; kwargs...
# )
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Mechanics(PlaneStrain())
  props = create_properties(physics)
  u = VectorFunction(V, "displ")
  times = TimeStepper(0., 1., 1)
  lsolver = x -> IterativeLinearSolver(x, :cg)
  for dev in backends
    for sp_type in [:csc, :csr]
      for use_condensed in [false, true]
        for use_inplace in [false, true]
          @info "Mechanics test $dev $sp_type $use_condensed $use_inplace"
          asm = SparseMatrixAssembler(
            u; 
            sparse_matrix_type = sp_type,
            use_condensed = use_condensed,
            use_inplace_methods = use_inplace
          )

          dbcs = DirichletBC[
            # DirichletBC("displ_x", fixed; sideset_name = "sset_3"),
            # DirichletBC("displ_y", fixed; sideset_name = "sset_3"),
            # DirichletBC("displ_x", fixed; sideset_name = "sset_1"),
            # DirichletBC("displ_y", displace; sideset_name = "sset_1"),
            DirichletBC("displ_x", fixed; nodeset_name = "nset_3"),
            DirichletBC("displ_y", fixed; nodeset_name = "nset_3"),
            DirichletBC("displ_x", fixed; nodeset_name = "nset_1"),
            DirichletBC("displ_y", displace; nodeset_name = "nset_1"),
          ]

          # pre-setup some scratch arrays
          p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, times=times)

          if dev != cpu
            p = p |> dev
            asm = asm |> dev 
          end

          solver = NewtonSolver(lsolver(asm))
          integrator = QuasiStaticIntegrator(solver)
          evolve!(integrator, p)

          if dev != cpu
            p = p |> cpu
          end

          U = p.field

          pp = PostProcessor(mesh, output_file, u)
          write_times(pp, 1, 0.0)
          write_field(pp, 1, ("displ_x", "displ_y"), U)
          # write_field(pp, 1, U)
          close(pp)
        end
      end
    end
  end
end
