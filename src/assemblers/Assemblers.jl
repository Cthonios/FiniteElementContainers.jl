abstract type AbstractAssembler{Dof <: DofManager} end
KA.get_backend(asm::AbstractAssembler) = KA.get_backend(asm.dof)

"""
$(TYPEDSIGNATURES)
Assembly method for a scalar field stored as a size 1 vector
Called on a single element e for a given block b where local_val
has already been constructed from quadrature contributions.
"""
function _assemble_element!(global_val::T, local_val, conn, e, b) where T <: AbstractArray{<:Number, 1}
  global_val[1] += local_val
  return nothing
end

"""
$(TYPEDSIGNATURES)
Assembly method for an H1Field, e.g. internal force
Called on a single element e for a given block b where local_val
has already been constructed from quadrature contributions.
"""
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

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id.

This is a top level method that simply dispatches to CPU/GPU specializations
based on the backend of assembler and also checks for consistent backends.

This method is common acroos all value types, e.g. energy, residual, stiffness, etc.
to maintain a common method call syntax. The difference in dispatch is handled by the 
sym argument. 
"""
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
Top level assembly method for ```H1Field``` that loops over blocks and dispatches
to appropriate kernels based on sym.
"""
function assemble!(assembler, physics, U::H1Field, sym::Symbol)
  _zero_storage(assembler, Val{sym}())
  fspace = assembler.dof.H1_vars[1].fspace
  X = fspace.coords
  for (b, conns) in enumerate(values(fspace.elem_conns))
    ref_fe = values(fspace.ref_fes)[b]
    _assemble_block!(assembler, physics, sym, ref_fe, U, X, conns, b)
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
create_bcs(asm::AbstractAssembler, type) = create_bcs(asm.dof, type)
"""
$(TYPEDSIGNATURES)
"""
create_field(asm::AbstractAssembler, type) = create_field(asm.dof, type)
"""
$(TYPEDSIGNATURES)
"""
create_unknowns(asm::AbstractAssembler) = create_unknowns(asm.dof)

"""
$(TYPEDSIGNATURES)
"""
function _element_level_coordinates(X::H1Field, ref_fe, conns, e)
  NDim = size(X, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  x_el = @views SMatrix{NDim, NNPE, Float64, NDim * NNPE}(X[:, conns[:, e]])
  return x_el
end

"""
$(TYPEDSIGNATURES)
"""
function _element_level_fields(U::H1Field, ref_fe, conns, e)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  u_el = @views SMatrix{ND, NNPE, Float64, NxNDof}(U[:, conns[:, e]])
  return u_el
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

# different backend implementations of abstract methods
include("CPUGeneral.jl")
include("GPUGeneral.jl")

# some utilities
include("SparsityPattern.jl")

# implementations
include("SparseMatrixAssembler.jl")
