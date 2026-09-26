# Element-level assembly: a physics with assembles_by_element(physics) == true
# has its kernels called once per element.  The wrapper below sums the
# per-quadrature-point kernels of the Poisson test physics inside one element
# call, so every assembly must reproduce the per-quadrature-point result
# exactly.

@testsnippet ElementAssemblyHelper begin
  using LinearAlgebra
  using ReferenceFiniteElements
  using SparseArrays
  using StaticArrays
  include("poisson/TestPoissonCommon.jl")
  import FiniteElementContainers: _cell_interpolants, state_variables

  struct ByElement{P <: AbstractPhysics} <: AbstractPhysics{1, 0, 0}
    inner::P
  end
  FiniteElementContainers.assembles_by_element(::ByElement) = true
  FiniteElementContainers.create_properties(p::ByElement) = create_properties(p.inner)

  # sum of a per-quadrature-point kernel over the element
  function sum_qps(kernel, physics::ByElement, ref_fe, x_el, t, dt, u_el, u_el_old, states, props_el)
    return mapreduce(+, 1:num_cell_quadrature_points(ref_fe)) do q
      so, sn = state_variables(states, q)
      kernel(physics.inner, _cell_interpolants(ref_fe, q), x_el, t, dt, u_el, u_el_old, so, sn, props_el)
    end
  end

  for kernel in (:residual, :stiffness, :mass)
    @eval @inline function FiniteElementContainers.$kernel(
      physics::ByElement, ref_fe::ReferenceFE, x_el, t, dt, u_el, u_el_old, states::ElementState, props_el
    )
      return sum_qps(FiniteElementContainers.$kernel, physics, ref_fe, x_el, t, dt, u_el, u_el_old, states, props_el)
    end
  end

  # the energy is stored per quadrature point: one entry each
  @inline function FiniteElementContainers.energy(
    physics::ByElement, ref_fe::ReferenceFE, x_el, t, dt, u_el, u_el_old, states::ElementState, props_el
  )
    return Tuple(begin
      so, sn = state_variables(states, q)
      FiniteElementContainers.energy(physics.inner, _cell_interpolants(ref_fe, q), x_el, t, dt, u_el, u_el_old, so, sn, props_el)
    end for q in 1:num_cell_quadrature_points(ref_fe))
  end

  @inline function FiniteElementContainers.stiffness_action(
    physics::ByElement, ref_fe::ReferenceFE, x_el, t, dt, u_el, u_el_old, v_el, states::ElementState, props_el
  )
    return mapreduce(+, 1:num_cell_quadrature_points(ref_fe)) do q
      so, sn = state_variables(states, q)
      FiniteElementContainers.stiffness_action(physics.inner, _cell_interpolants(ref_fe, q), x_el, t, dt, u_el, u_el_old, v_el, so, sn, props_el)
    end
  end

  f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
  bc_func(_, _) = 0.
  mesh = UnstructuredMesh(Base.source_dir() * "/poisson/poisson.g")
  V = FunctionSpace(mesh, H1Field, Lagrange)
  u = ScalarFunction(V, "u")
  dbcs = DirichletBC[
    DirichletBC("u", bc_func; sideset_name = "sset_1"),
    DirichletBC("u", bc_func; sideset_name = "sset_2"),
  ]

  function setup(physics)
    asm = SparseMatrixAssembler(u; sparse_matrix_type = :csc, use_condensed = false,
                                use_inplace_methods = false)
    p = create_parameters(mesh, asm, physics, create_properties(physics); dirichlet_bcs = dbcs)
    return asm, p
  end
end

@testitem "Element assembly - every assembly matches the per-quadrature-point path" setup=[ElementAssemblyHelper] begin
  import FiniteElementContainers: residual, stiffness, mass, diagonal, energy, stiffness_action
  asm_q, p_q = setup(Poisson(f))
  asm_e, p_e = setup(ByElement(Poisson(f)))

  Uu = create_unknowns(asm_q)
  Uu .= sin.(1.0:length(Uu))
  Vu = cos.(1.0:length(Uu))

  assemble_vector!(asm_q, residual, Uu, p_q)
  assemble_vector!(asm_e, residual, Uu, p_e)
  @test residual(asm_e) ≈ residual(asm_q)
  @test maximum(abs, residual(asm_q)) > 0

  assemble_stiffness!(asm_q, stiffness, Uu, p_q)
  assemble_stiffness!(asm_e, stiffness, Uu, p_e)
  @test stiffness(asm_e) ≈ stiffness(asm_q)

  assemble_mass!(asm_q, mass, Uu, p_q)
  assemble_mass!(asm_e, mass, Uu, p_e)
  @test mass(asm_e) ≈ mass(asm_q)

  # the diagonal from the element matrix
  assemble_diagonal!(asm_q, stiffness, Uu, p_q)
  assemble_diagonal!(asm_e, stiffness, Uu, p_e)
  @test diagonal(asm_e) ≈ diagonal(asm_q)

  assemble_scalar!(asm_q, energy, Uu, p_q)
  assemble_scalar!(asm_e, energy, Uu, p_e)
  @test asm_e.scalar_quadrature_storage ≈ asm_q.scalar_quadrature_storage
  @test abs(sum(asm_q.scalar_quadrature_storage)) > 0

  assemble_matrix_free_action!(asm_q, stiffness_action, Uu, Vu, p_q)
  assemble_matrix_free_action!(asm_e, stiffness_action, Uu, Vu, p_e)
  @test asm_e.stiffness_action_storage ≈ asm_q.stiffness_action_storage
  @test maximum(abs, asm_q.stiffness_action_storage) > 0

  assemble_matrix_action!(asm_q, stiffness, Uu, Vu, p_q)
  assemble_matrix_action!(asm_e, stiffness, Uu, Vu, p_e)
  @test asm_e.stiffness_action_storage ≈ asm_q.stiffness_action_storage
end

@testitem "Element assembly - the in-place kernels refuse an element physics" setup=[ElementAssemblyHelper] begin
  import FiniteElementContainers: residual
  physics = ByElement(Poisson(f))
  asm = SparseMatrixAssembler(u; sparse_matrix_type = :csc, use_condensed = false,
                              use_inplace_methods = true)
  p = create_parameters(mesh, asm, physics, create_properties(physics); dirichlet_bcs = dbcs)
  Uu = create_unknowns(asm)
  @test_throws ErrorException assemble_vector!(asm, residual, Uu, p)
end
