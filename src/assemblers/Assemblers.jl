abstract type AbstractAssembler{Dof <: DofManager} end

"""
$(TYPEDSIGNATURES)
Assembly method for a scalar field stored as a size 1 vector
"""
function _assemble_element!(global_val::T, local_val, conn, e, b) where T <: AbstractArray{<:Number, 1}
  global_val[1] += local_val
  return nothing
end

function _assemble_element!(global_val::H1Field, local_val, conn, e, b)
  n_dofs = size(global_val, 1)
  for i in axes(conn, 1)
    for d in 1:n_dofs
      # n = 2 * i + d
      global_id = n_dofs * (conn[i] - 1) + d
      local_id = n_dofs * (i - 1) + d
      global_val[global_id] += local_val[local_id]
    end
  end
  return nothing
end

function _assemble_block!(assembler, physics, ::Val{:residual}, ref_fe, U, X, conns, block_id, ::KA.CPU)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_coordinates(X, ref_fe, conns, e)
    u_el = _element_level_fields(U, ref_fe, conns, e)
    R_el = zeros(SVector{NxNDof, Float64})

    for q in 1:num_quadrature_points(ref_fe)
      interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
      R_q = residual(physics, interps, u_el)
      R_el = R_el + R_q
    end
    
    @views _assemble_element!(assembler.residual_storage, R_el, conns[:, e], e, block_id)
  end
end

KA.@kernel function _assemble_block_residual_kernel!(assembler, physics, ref_fe, U, X, conns, block_id)
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_coordinates(X, ref_fe, conns, E)
  u_el = _element_level_fields(U, ref_fe, conns, E)
  R_el = zeros(SVector{NxNDof, Float64})

  for q in 1:num_quadrature_points(ref_fe)
    interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
    R_q = residual(physics, interps, u_el)
    R_el = R_el + R_q
  end

  # now assemble atomically
  n_dofs = size(assembler.residual_storage, 1)
  for i in 1:size(conns, 1)
    for d in 1:n_dofs
      global_id = n_dofs * (conns[i, E] - 1) + d
      local_id = n_dofs * (i - 1) + d
      Atomix.@atomic assembler.residual_storage.vals[global_id] += R_el[local_id]
    end
  end
end

function _assemble_block!(assembler, physics, ::Val{:residual}, ref_fe, U, X, conns, block_id, backend::KA.Backend)
  kernel! = _assemble_block_residual_kernel!(backend)
  kernel!(assembler, physics, ref_fe, U, X, conns, block_id, ndrange=size(conns, 2))
  return nothing
end

function _assemble_block!(assembler, physics, ::Val{:residual_and_stiffness}, ref_fe, U, X, conns, block_id, ::KA.CPU)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_coordinates(X, ref_fe, conns, e)
    u_el = _element_level_fields(U, ref_fe, conns, e)
    R_el = zeros(SVector{NxNDof, Float64})
    K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

    for q in 1:num_quadrature_points(ref_fe)
      interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
      R_q = residual(physics, interps, u_el)
      K_q = stiffness(physics, interps, u_el)
      R_el = R_el + R_q
      K_el = K_el + K_q
    end
    
    @views _assemble_element!(assembler.residual_storage, R_el, conns[:, e], e, block_id)
    @views _assemble_element!(assembler, K_el, conns[:, e], e, block_id)
  end
  return nothing
end

function _assemble_block!(assembler, physics, ::Val{:stiffness}, ref_fe, U, X, conns, block_id, ::KA.CPU)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_coordinates(X, ref_fe, conns, e)
    u_el = _element_level_fields(U, ref_fe, conns, e)
    K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

    for q in 1:num_quadrature_points(ref_fe)
      interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
      K_q = stiffness(physics, interps, u_el)
      K_el = K_el + K_q
    end
    
    @views _assemble_element!(assembler, K_el, conns[:, e], e, block_id)
  end
  return nothing
end

KA.@kernel function _assemble_block_stiffness_kernel!(assembler, physics, ref_fe, U, X, conns, block_id)
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_coordinates(X, ref_fe, conns, E)
  u_el = _element_level_fields(U, ref_fe, conns, E)
  K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

  for q in 1:num_quadrature_points(ref_fe)
    interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
    K_q = stiffness(physics, interps, u_el)
    K_el = K_el + K_q
  end

  # start_id = (block_id - 1) * assembler.pattern.block_sizes[block_id] + 
  #            (el_id - 1) * assembler.pattern.block_offsets[block_id] + 1
  # end_id = start_id + assembler.pattern.block_offsets[block_id] - 1
  # TODO patch this up for multi-block later
  # hardcoded for single block quad4
  start_id = (E - 1) * 16 + 1
  end_id = start_id + 16 - 1
  ids = start_id:end_id
  for (i, id) in enumerate(ids)
    Atomix.@atomic assembler.stiffness_storage[id] += K_el.data[i]
  end
end

function _assemble_block!(assembler, physics, ::Val{:stiffness}, ref_fe, U, X, conns, block_id, backend::KA.Backend)
  kernel! = _assemble_block_stiffness_kernel!(backend)
  kernel!(assembler, physics, ref_fe, U, X, conns, block_id, ndrange=size(conns, 2))
  return nothing
end

function _assemble_block!(assembler, physics, sym, ref_fe, U, X, conns, block_id)
  backend = KA.get_backend(assembler)
  # TODO add get_backend method of ref_fe
  @assert backend == KA.get_backend(U)
  @assert backend == KA.get_backend(X)
  @assert backend == KA.get_backend(conns)
  _assemble_block!(assembler, physics, Val{sym}(), ref_fe, U, X, conns, block_id, backend)
end

"""
$(TYPEDSIGNATURES)
Top level assembly method
TODO make more general
"""
function assemble!(assembler, physics, U::H1Field, sym::Symbol)
  # TODO need to generalize to different field types
  # fill!(assembler.residual_storage, zero(eltype(assembler.residual_storage)))
  _zero_storage(assembler, Val{sym}())
  fspace = assembler.dof.H1_vars[1].fspace
  for (b, conns) in enumerate(values(fspace.elem_conns))
    ref_fe = values(fspace.ref_fes)[b]
    _assemble_block!(assembler, physics, sym, ref_fe, U, fspace.coords, conns, b)
  end
  return nothing
end


create_bcs(asm::AbstractAssembler, type) = create_bcs(asm.dof, type)
create_field(asm::AbstractAssembler, type) = create_field(asm.dof, type)
create_unknowns(asm::AbstractAssembler) = create_unknowns(asm.dof)

function _element_level_coordinates(X::H1Field, ref_fe, conns, e)
  NDim = size(X, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  x_el = @views SMatrix{NDim, NNPE, Float64, NDim * NNPE}(X[:, conns[:, e]])
  return x_el
end

function _element_level_fields(U::H1Field, ref_fe, conns, e)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  u_el = @views SMatrix{ND, NNPE, Float64, NxNDof}(U[:, conns[:, e]])
  return u_el
end

KA.@kernel function _extract_residual_unknowns!(Ru, unknown_dofs, R)
  N = KA.@index(Global)
  Ru[N] = R[unknown_dofs[N]]
end

function _residual(asm::AbstractAssembler, backend::KA.Backend)
  kernel! = _extract_residual_unknowns!(backend)
  kernel!(asm.residual_unknowns, 
          asm.dof.H1_unknown_dofs, 
          asm.residual_storage, 
          ndrange=length(asm.dof.H1_unknown_dofs))
  return asm.residual_unknowns
end 

# TODO hardcoded to H1 fields right now.
function _residual(asm::AbstractAssembler, ::KA.CPU)
  # for n in axes(asm.residual_unknowns, 1)
  #   asm.residual_unknowns[n] = asm.residual_storage[asm.dof.H1_unknown_dofs[n]]
  # end
  @views asm.residual_unknowns .= asm.residual_storage[asm.dof.H1_unknown_dofs]
  return asm.residual_unknowns
end

function residual(asm::AbstractAssembler)
  return _residual(asm, KA.get_backend(asm))
end

function update_field!(U, asm::AbstractAssembler, Uu, Ubc)
  update_field!(U, asm.dof, Uu, Ubc)
  return nothing
end

function _zero_storage(asm::AbstractAssembler, ::Val{:residual})
  fill!(asm.residual_storage.vals, zero(eltype(asm.residual_storage.vals)))
end

# some utilities
include("SparsityPattern.jl")

# implementations
include("SparseMatrixAssembler.jl")
