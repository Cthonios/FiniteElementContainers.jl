@testsnippet LaplaceRegressionHelper begin
  if "--test-amdgpu" in ARGS @eval using AMDGPU end
  if "--test-cuda" in ARGS @eval using CUDA end
  using Exodus
  using StaticArrays
  include("TestLaplaceCommon.jl")
  include("../TestUtils.jl")
  backends = _get_backends()
  use_condenseds = [false, true]
  gold_file = Base.source_dir() * "/laplace.gold"
  mesh_file = Base.source_dir() * "/laplace.g"
  f_lapace(X, _) = SVector{1, Float64}(2. * π^2 * sin(π * X[1]) * sin(π * X[2]))
  bc_func_laplace(_, _) = 0.
  bc_func_neumann_lapace(_, _) = SVector{1, Float64}(1.)
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Laplace()
  props = create_properties(physics)
  u = ScalarFunction(V, "u")
  lsolver = x -> IterativeLinearSolver(x, :cg)
  nlsolver = NewtonSolver
end

@testitem "Regression test - Laplace with sources" setup=[LaplaceRegressionHelper] begin
  output_file = "laplace_test_1.e"
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )
      # setup and update bcs
      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "sset_1"),
        DirichletBC("u", bc_func_laplace; sideset_name = "sset_2"),
        DirichletBC("u", bc_func_laplace; sideset_name = "sset_3"),
        DirichletBC("u", bc_func_laplace; sideset_name = "sset_4"),
      ]
      sources = Source[
        Source("u", f_lapace, "block_1")
      ]

      # setup the parameters
      p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, sources=sources)

      if dev != cpu
        p = p |> dev
        asm = asm |> dev 
      end

      # setup solver and integrator
      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      U = p.field

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
  end
end

@testitem "Regression test - test_laplace_dirichlet_with_nodesets" setup=[LaplaceRegressionHelper] begin
  output_file = "laplace_test_2.e"
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )
      # setup and update bcs
      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; nodeset_name = "nset_1"),
        DirichletBC("u", bc_func_laplace; nodeset_name = "nset_2"),
        DirichletBC("u", bc_func_laplace; nodeset_name = "nset_3"),
        DirichletBC("u", bc_func_laplace; nodeset_name = "nset_4"),
      ]
      sources = Source[
        Source("u", f_lapace, "block_1")
      ]

      # setup the parameters
      p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, sources=sources)

      if dev != cpu
        p = p |> dev
        asm = asm |> dev 
      end

      # setup solver and integrator
      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      U = p.field

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
  end
end

@testitem "Regression test - test_laplace_neumann" setup=[LaplaceRegressionHelper] begin
  output_file = "laplace_test_3.e"
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )
      # setup and update bcs
      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "sset_1"),
        DirichletBC("u", bc_func_laplace; sideset_name = "sset_2")
      ]
      nbcs = NeumannBC[
        NeumannBC("u", bc_func_neumann_lapace, "sset_3"),
        NeumannBC("u", bc_func_neumann_lapace, "sset_4")
      ]
      sources = Source[
        Source("u", f_lapace, "block_1")
      ]

      # direct solver test
      # setup the parameters
      p = create_parameters(
        mesh, asm, physics, props; 
        dirichlet_bcs=dbcs,
        neumann_bcs=nbcs,
        sources=sources
      )

      if dev != cpu
        p = p |> dev
        asm = asm |> dev 
      end

      # setup solver and integrator
      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      # TODO make a neumann gold file
      # U = p.field

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
  end
end

@testitem "Regression test - test_laplace_dirichlet_multi_block_quad4_quad4" setup=[LaplaceRegressionHelper] begin
  output_file = "laplace_test_4.e"
  mesh = UnstructuredMesh(Base.source_dir() * "/multi_block_mesh_quad4_quad4.g")
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Laplace()
  props = create_properties(physics)
  u = ScalarFunction(V, "u")
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )

      # setup and update bcs
      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "boundary")
      ]
      sources = Source[
        Source("u", f_lapace, "block_1")
      ]

      # setup the parameters
      p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, sources=sources)

      if dev != cpu
        p = p |> dev
        asm = asm |> dev 
      end

      # setup solver and integrator
      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      U = p.field

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
  end
end

@testitem "Regression test - test_laplace_dirichlet_multi_block_quad4_tri3" setup=[LaplaceRegressionHelper] begin
  output_file = "laplace_test_5.e"
  mesh = UnstructuredMesh(Base.source_dir() * "/multi_block_mesh_quad4_tri3.g")
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Laplace()
  props = create_properties(physics)
  u = ScalarFunction(V, "u")
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )

      # setup and update bcs
      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "boundary")
      ]
      sources = Source[
        Source("u", f_lapace, "block_1")
      ]

      # setup the parameters
      p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, sources=sources)

      if dev != cpu
        p = p |> dev
        asm = asm |> dev 
      end

      # setup solver and integrator
      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      U = p.field

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
  end
end

@testitem "Regression test - test_laplace_dirichlet_structured_mesh_quad4" setup=[LaplaceRegressionHelper] begin
  output_file = "laplace_test_6.e"
  mesh = StructuredMesh("quad", (0., 0.), (1., 1.), (11, 11))
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Laplace()
  props = create_properties(physics)
  u = ScalarFunction(V, "u")
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )

      # setup and update bcs
      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "bottom"),
        DirichletBC("u", bc_func_laplace; sideset_name = "right"),
        DirichletBC("u", bc_func_laplace; sideset_name = "top"),
        DirichletBC("u", bc_func_laplace; sideset_name = "left"),
      ]
      sources = Source[
        Source("u", f_lapace, "block_1")
      ]

      # setup the parameters
      p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, sources=sources)

      if dev != cpu
        p = p |> dev
        asm = asm |> dev 
      end

      # setup solver and integrator
      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      U = p.field

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
  end
end

@testitem "Regression test - test_laplace_dirichlet_structured_mesh_tri3" setup=[LaplaceRegressionHelper] begin
  output_file = "laplace_test_7.e"
  mesh = StructuredMesh("quad", (0., 0.), (1., 1.), (11, 11))
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Laplace()
  props = create_properties(physics)
  u = ScalarFunction(V, "u")
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )

      # setup and update bcs
      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "bottom"),
        DirichletBC("u", bc_func_laplace; sideset_name = "right"),
        DirichletBC("u", bc_func_laplace; sideset_name = "top"),
        DirichletBC("u", bc_func_laplace; sideset_name = "left"),
      ]
      sources = Source[
        Source("u", f_lapace, "block_1")
      ]

      # setup the parameters
      p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, sources=sources)

      if dev != cpu
        p = p |> dev
        asm = asm |> dev 
      end

      # setup solver and integrator
      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      U = p.field

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
  end
end

@testitem "Regression test - test_laplace_neumann_structured_mesh_quad4" setup=[LaplaceRegressionHelper] begin
  # Laplace equation (-∇²u = 0) on [0,1]² with mixed BCs:
  #   u = 0            at x=0  (Dirichlet, :left)
  #   ∂u/∂n = 1        at x=1  (Neumann,   :right;  outward normal = +x̂)
  #   ∂u/∂n = 0        at y=0,1 (natural — not prescribed)
  #
  # Exact solution: u(x,y) = x
  # QUAD4 bilinear elements represent linear functions exactly, so FEM error = 0.
  #
  # FEC sign convention: the NBC assembler adds +∫g·N dΓ to the residual, so
  #   g = −(∂u/∂n).  For ∂u/∂n = +1 we must pass g = −1.
  nbc_minus = (_, _) -> SVector{1, Float64}(-1.)

  mesh = StructuredMesh("quad", (0., 0.), (1., 1.), (11, 11))
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Laplace()
  props = create_properties(physics)
  u = ScalarFunction(V, "u")
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )

      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "left"),
      ]

      nbcs = NeumannBC[
        NeumannBC("u", nbc_minus, "right"),
      ]

      p = create_parameters(
        mesh, asm, physics, props;
        dirichlet_bcs=dbcs,
        neumann_bcs=nbcs
      )

      if dev != cpu
        p = p |> dev
        asm = asm |> dev
      end

      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      # u(x,y) = x  →  max = 1.0  (x=1),  min = 0.0  (x=0, Dirichlet)
      @test maximum(p.field.data) ≈ 1.0 atol=1e-6
      @test minimum(p.field.data) ≈ 0.0 atol=1e-6
    end
  end
end

@testitem "Regression test - test_laplace_neumann_structured_mesh_tri3" setup=[LaplaceRegressionHelper] begin
  # Same problem as the QUAD4 variant above, meshed with TRI3 elements.
  # TRI3 linear triangles represent linear functions exactly, so FEM error = 0.
  #
  # FEC sign convention: the NBC assembler adds +∫g·N dΓ to the residual, so
  #   g = −(∂u/∂n).  For ∂u/∂n = +1 we must pass g = −1.
  nbc_minus = (_, _) -> SVector{1, Float64}(-1.)

  mesh = StructuredMesh("tri", (0., 0.), (1., 1.), (11, 11))
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Laplace()
  props = create_properties(physics)
  u = ScalarFunction(V, "u")
  for dev in backends
    for use_condensed in use_condenseds
      asm = SparseMatrixAssembler(
        u; 
        sparse_matrix_type = :csc,
        use_condensed = use_condensed,
        use_inplace_methods = false
      )

      dbcs = DirichletBC[
        DirichletBC("u", bc_func_laplace; sideset_name = "left"),
      ]

      nbcs = NeumannBC[
        NeumannBC("u", nbc_minus, "right"),
      ]

      p = create_parameters(
        mesh, asm, physics, props;
        dirichlet_bcs=dbcs,
        neumann_bcs=nbcs
      )

      if dev != cpu
        p = p |> dev
        asm = asm |> dev
      end

      solver = nlsolver(lsolver(asm))
      integrator = QuasiStaticIntegrator(solver)
      evolve!(integrator, p)

      if dev != cpu
        p = p |> cpu
      end

      @test maximum(p.field.data) ≈ 1.0 atol=1e-6
      @test minimum(p.field.data) ≈ 0.0 atol=1e-6
    end
  end
end
