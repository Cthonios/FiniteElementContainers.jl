@testitem "Assemblers - test_sparse_matrix_gpu_trace" tags=[:gpu] begin
  using Adapt
  if "--test-amdgpu" in ARGS @eval using AMDGPU end
  if "--test-cuda" in ARGS @eval using CUDA end
  using LinearAlgebra
  using SparseArrays
  using SparseMatricesCSR
  include("TestUtils.jl")
  A = SparseMatrixCSC(rand(10, 10))
  trA_check = tr(A)

  if _check_functional_backend(:AMDGPU)
    A_gpu = adapt(ROCArray, A)
  elseif _check_functional_backend(:CUDA)
    A_gpu = adapt(CuArray, A)
  else
    return nothing
  end

  trA = FiniteElementContainers._sptrace(A_gpu)
  @test trA ≈ trA_check

  temp = rand(10, 10)
  A = SparseMatrixCSR(temp)
  trA_check = tr(A)

  if _check_functional_backend(:AMDGPU)
    A_gpu = AMDGPU.rocSPARSE.ROCSparseMatrixCSR(adapt(ROCArray, temp))
  elseif _check_functional_backend(:CUDA)
    A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(adapt(CuArray, temp))
  else
    return nothing
  end

  trA = FiniteElementContainers._sptrace(A_gpu)
  @test trA ≈ trA_check
end

@testsnippet AssemblerHelperPoisson begin
  if "--test-amdgpu" in ARGS @eval using AMDGPU end
  if "--test-cuda" in ARGS @eval using CUDA end
  using LinearAlgebra
  using StaticArrays
  include("poisson/TestPoissonCommon.jl")
  mesh_file = "./poisson/multi_block_mesh_quad4_tri3.g"
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
  bc_func(_, _) = 0.
  physics = Poisson(f)
  props = SVector{0, Float64}()
  u = ScalarFunction(V, "u")
  dbcs = DirichletBC[
    DirichletBC("u", bc_func; sideset_name = "boundary")
  ]
end

@testsnippet AssemblerHelperMechanics begin
  if "--test-amdgpu" in ARGS @eval using AMDGPU end
  if "--test-cuda" in ARGS @eval using CUDA end
  using StaticArrays
  using Tensors
  include("mechanics/TestMechanicsCommon.jl")
  mesh_file = "./poisson/multi_block_mesh_quad4_tri3.g"
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Mechanics(1.0, PlaneStrain())
  props = create_properties(physics)
  u = VectorFunction(V, "displ")
  fixed(_, _) = 0.
  dbcs = DirichletBC[
    DirichletBC("displ_x", fixed; sideset_name = "boundary")
    DirichletBC("displ_y", fixed; sideset_name = "boundary")
  ]
  times = TimeStepper(0., 1., 1)
end

@testitem "Assembler - assembler_consistency_poisson" setup=[AssemblerHelperPoisson] begin
  include("TestUtils.jl")
  backends = _get_backends()
  dbcs = DirichletBC[
    DirichletBC("u", bc_func; sideset_name = "boundary")
  ]
  for dev in backends
    for sp_type in [:csc, :csr]
      for use_condensed in [false, true]
        asm_1 = SparseMatrixAssembler(
          u;
          sparse_matrix_type = sp_type,
          use_condensed = use_condensed,
          use_inplace_methods = false
        )
        asm_2 = SparseMatrixAssembler(
          u;
          sparse_matrix_type = sp_type,
          use_condensed = use_condensed,
          use_inplace_methods = true
        )

        p_1 = create_parameters(mesh, asm_1, physics, props; dirichlet_bcs=dbcs)
        p_2 = create_parameters(mesh, asm_2, physics, props; dirichlet_bcs=dbcs)
        FiniteElementContainers.initialize!(p_1)
        FiniteElementContainers.initialize!(p_2)

        if dev != cpu
          asm_1 = asm_1 |> dev
          asm_2 = asm_2 |> dev
          p_1 = p_1 |> dev
          p_2 = p_2 |> dev
        end

        U_1 = create_unknowns(asm_1)
        V_1 = create_unknowns(asm_1)
        U_2 = create_unknowns(asm_2)
        V_2 = create_unknowns(asm_2)
        temp = rand(length(V_2)) |> dev
        V_1 .= temp
        V_2 .= temp

        # test vector consistency
        assemble_vector!(asm_1, residual, U_1, p_1)
        assemble_vector!(asm_2, residual!, U_2, p_2)
        R_1 = residual(asm_1) |> copy
        R_2 = residual(asm_2) |> copy
        if dev != cpu
          R_1 = R_1 |> cpu
          R_2 = R_2 |> cpu
        end
        @test all(R_1 .≈ R_2)

        # test stiffness consistency
        assemble_stiffness!(asm_1, stiffness, U_1, p_1)
        assemble_stiffness!(asm_2, stiffness!, U_2, p_2)
        # @test all(stiffness(asm_1) .≈ stiffness(asm_2))
        K_1 = stiffness(asm_1) |> copy
        K_2 = stiffness(asm_2) |> copy
        if dev != cpu
          K_1 = K_1 |> cpu
          K_2 = K_2 |> cpu
        end
        # @test all(K_1 .≈ K_2)
        @test all(K_1 .≈ K_2)

        # test stiffness action consistency
        assemble_matrix_action!(asm_1, stiffness, U_1, V_1, p_1)
        assemble_matrix_action!(asm_2, stiffness_action!, U_2, V_2, p_2)
        Kv_1 = hvp(asm_1, V_1) |> copy
        Kv_2 = hvp(asm_2, V_2) |> copy
        if dev != cpu
          Kv_1 = Kv_1 |> cpu
          Kv_2 = Kv_2 |> cpu
        end
        @test all(Kv_1 .≈ Kv_2)

        # test mass consistency
        assemble_mass!(asm_1, mass, U_1, p_1)
        assemble_mass!(asm_2, mass!, U_2, p_2)
        M_1 = mass(asm_1) |> copy
        M_2 = mass(asm_2) |> copy
        if dev != cpu
          M_1 = M_1 |> cpu
          M_2 = M_2 |> cpu
        end
        # @test all(M_1 .≈ M_2)
        @test all(M_1 .≈ M_2)
        
        # mass action consistency
        assemble_matrix_action!(asm_1, mass, U_1, V_1, p_1)
        assemble_matrix_action!(asm_2, mass_action!, U_2, V_2, p_2)
        Mv_1 = hvp(asm_1, V_1) |> copy
        Mv_2 = hvp(asm_2, V_2) |> copy
        if dev != cpu
          Mv_1 = Mv_1 |> cpu
          Mv_2 = Mv_2 |> cpu
        end
        @test all(Mv_1 .≈ Mv_2)
      end
    end
  end
end

@testitem "Assembler - assembler_consistency_mechanics" setup=[AssemblerHelperMechanics] begin
  include("TestUtils.jl")
  backends = _get_backends()
  for dev in backends
    for sp_type in [:csc, :csr]
      for use_condensed in [false, true]
        asm_1 = SparseMatrixAssembler(
          u;
          sparse_matrix_type = sp_type,
          use_condensed = use_condensed,
          use_inplace_methods = false
        )
        asm_2 = SparseMatrixAssembler(
          u;
          sparse_matrix_type = sp_type,
          use_condensed = use_condensed,
          use_inplace_methods = true
        )
    
        p_1 = create_parameters(mesh, asm_1, physics, props; dirichlet_bcs = dbcs, times = times)
        p_2 = create_parameters(mesh, asm_2, physics, props; dirichlet_bcs = dbcs, times = times)
    
        if dev != cpu
          asm_1 = asm_1 |> dev
          asm_2 = asm_2 |> dev
          p_1 = p_1 |> dev
          p_2 = p_2 |> dev
        end
    
        FiniteElementContainers.initialize!(p_1)
        FiniteElementContainers.initialize!(p_2)
        U_1 = create_unknowns(asm_1)
        V_1 = create_unknowns(asm_1)
        U_2 = create_unknowns(asm_2)
        V_2 = create_unknowns(asm_2)
    
        # test vector consistency
        assemble_vector!(asm_1, residual, U_1, p_1)
        assemble_vector!(asm_2, residual!, U_2, p_2)
        R_1 = residual(asm_1) |> copy
        R_2 = residual(asm_2) |> copy
        if dev != cpu
          R_1 = R_1 |> cpu
          R_2 = R_2 |> cpu
        end
        @test all(R_1 .≈ R_2)
    
        # test stiffness consistency
        assemble_stiffness!(asm_1, stiffness, U_1, p_1)
        assemble_stiffness!(asm_2, stiffness!, U_2, p_2)
        K_1 = stiffness(asm_1) |> copy
        K_2 = stiffness(asm_2) |> copy
        if dev != cpu
          K_1 = K_1 |> cpu
          K_2 = K_2 |> cpu
        end
        @test all(isapprox.(K_1, K_2, atol = 1e-5, rtol = 1e-5)) # need large tol for this one for some reason maybe AD with symmetric stuff?
    
        # test stiffness action consistency
        assemble_matrix_action!(asm_1, stiffness, U_1, V_1, p_1)
        assemble_matrix_action!(asm_2, stiffness_action!, U_2, V_2, p_2)
        Kv_1 = hvp(asm_1, V_1) |> copy
        Kv_2 = hvp(asm_2, V_2) |> copy
        if dev != cpu
          Kv_1 = Kv_1 |> cpu
          Kv_2 = Kv_2 |> cpu
        end
        @test all(Kv_1 .≈ Kv_2)
    
        # test mass consistency
        assemble_mass!(asm_1, mass, U_1, p_1)
        assemble_mass!(asm_2, mass!, U_2, p_2)
        M_1 = mass(asm_1) |> copy
        M_2 = mass(asm_2) |> copy
        if dev != cpu
          M_1 = M_1 |> cpu
          M_2 = M_2 |> cpu
        end
        @test all(M_1 .≈ M_2)
        
        # mass action consistency
        assemble_matrix_action!(asm_1, mass, U_1, V_1, p_1)
        assemble_matrix_action!(asm_2, mass_action!, U_2, V_2, p_2)
        Mv_1 = hvp(asm_1, V_1) |> copy
        Mv_2 = hvp(asm_2, V_2) |> copy
        if dev != cpu
          Mv_1 = Mv_1 |> cpu
          Mv_2 = Mv_2 |> cpu
        end
        @test all(Mv_1 .≈ Mv_2)
      end
    end
  end
end

@testitem "Assemblers - test_matrix_free_action_poisson" setup=[AssemblerHelperPoisson] begin
  include("TestUtils.jl")
  backends = _get_backends()
  for dev in backends
    asm = SparseMatrixAssembler(u)
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)
    FiniteElementContainers.initialize!(p)
    Uu = create_unknowns(asm)
    Vu = create_unknowns(asm)
    fill!(Vu, 1.0)

    if dev != cpu
      p   = p   |> dev
      asm = asm |> dev
      Uu  = Uu  |> dev
      Vu  = Vu  |> dev
    end

    # stiffness: K·v via matrix path vs matrix-free path must agree
    assemble_matrix_action!(asm, stiffness, Uu, Vu, p)
    Kv_ref = copy(Array(hvp(asm, Vu)))
    assemble_matrix_free_action!(asm, stiffness_action, Uu, Vu, p)
    Kv_mf = Array(hvp(asm, Vu))
    @test Kv_mf ≈ Kv_ref

    # mass: M·v via matrix path vs matrix-free path must agree
    assemble_matrix_action!(asm, mass, Uu, Vu, p)
    Mv_ref = copy(Array(hvp(asm, Vu)))
    assemble_matrix_free_action!(asm, mass_action, Uu, Vu, p)
    Mv_mf = Array(hvp(asm, Vu))
    @test Mv_mf ≈ Mv_ref
  end
end

# function test_matrix_free_action_mechanics(dev)
@testitem "Assemblers - test_matrix_free_action_mechanics" setup=[AssemblerHelperMechanics] begin
  # Mechanics: vector problem, tests stiffness_action
  include("TestUtils.jl")
  backends = _get_backends()
  for dev in backends
    asm = SparseMatrixAssembler(u)
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs, times = times)
    FiniteElementContainers.initialize!(p)
    Uu = create_unknowns(asm)
    Vu = create_unknowns(asm)
    fill!(Vu, 1.0)

    if dev != cpu
      p   = p   |> dev
      asm = asm |> dev
      Uu  = Uu  |> dev
      Vu  = Vu  |> dev
    end

    assemble_matrix_action!(asm, stiffness, Uu, Vu, p)
    Kv_ref = copy(Array(hvp(asm, Vu)))
    assemble_matrix_free_action!(asm, stiffness_action, Uu, Vu, p)
    Kv_mf = Array(hvp(asm, Vu))
    @test Kv_mf ≈ Kv_ref
  end
end

@testitem "Assemblers - test_enzyme_safe_consistency" setup=[AssemblerHelperPoisson] begin
  asm = SparseMatrixAssembler(u)
  p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)
  FiniteElementContainers.initialize!(p)
  Uu = create_unknowns(asm)
  R1 = create_field(asm)
  R2 = create_field(asm)
  assemble_vector!(R1, asm.matrix_pattern, asm.dof, residual, Uu, p)
  FiniteElementContainers.assemble_vector_enzyme_safe!(R2, asm.matrix_pattern, asm.dof, residual, Uu, p)
  @test all(R1 .≈ R2)
end