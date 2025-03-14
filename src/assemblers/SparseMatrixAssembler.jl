struct SparseMatrixAssembler{
  Dof <: DofManager, 
  Pattern <: SparsityPattern, 
  Storage1 <: AbstractArray{<:Number},
  Storage2 <: AbstractField,
  Storage3 <: AbstractArray{<:Number, 1}
} <: AbstractAssembler{Dof}
  dof::Dof
  pattern::Pattern
  constraint_storage::Storage1
  residual_storage::Storage2
  residual_unknowns::Storage3
  stiffness_storage::Storage1
end

# TODO this will not work for other than single H1 spaces
function SparseMatrixAssembler(dof::DofManager, type::Type{<:AbstractField})
  pattern = SparsityPattern(dof, type)
  constraint_storage = zeros(length(dof))
  # constraint_storage = zeros(_dof_manager_vars(dof, type))
  constraint_storage[dof.H1_bc_dofs] .= 1.
  # fill!(constraint_storage, )
  # residual_storage = zeros(length(dof))
  residual_storage = create_field(dof, H1Field)
  residual_unknowns = create_unknowns(dof)
  stiffness_storage = zeros(num_entries(pattern))
  return SparseMatrixAssembler(
    dof, pattern, 
    constraint_storage, 
    residual_storage, residual_unknowns,
    stiffness_storage
  )
end

# TODO how best to do this?
# Should probably be a block array maybe?
# function SparseMatrixAssembler(vars...)
#   dof = DofManager(vars...)
#   return SparseMatrixAssembler(dof)
# end 

function Base.show(io::IO, asm::SparseMatrixAssembler)
  println(io, "SparseMatrixAssembler")
  println(io, "  ", asm.dof)
end

function _assemble_element!(asm::SparseMatrixAssembler, K_el::SMatrix, conn, el_id::Int, block_id::Int)
  block_size = values(asm.pattern.block_sizes)[block_id]
  block_offset = values(asm.pattern.block_offsets)[block_id]
  # start_id = (block_id - 1) * asm.pattern.block_sizes[block_id] + 
  #            (el_id - 1) * asm.pattern.block_offsets[block_id] + 1
  # end_id = start_id + asm.pattern.block_offsets[block_id] - 1
  start_id = (block_id - 1) * block_size + 
             (el_id - 1) * block_offset + 1
  end_id = start_id + block_offset - 1
  ids = start_id:end_id
  @views asm.stiffness_storage[ids] += K_el[:]
  return nothing
end

KA.get_backend(asm::SparseMatrixAssembler) = KA.get_backend(asm.dof)

"""
$(TYPEDSIGNATURES)
"""
function SparseArrays.sparse(assembler::SparseMatrixAssembler)
  # ids = pattern.unknown_dofs
  pattern = assembler.pattern
  storage = assembler.stiffness_storage
  return @views sparse(pattern.Is, pattern.Js, storage)
end

"""
$(TYPEDSIGNATURES)
"""
function SparseArrays.sparse!(assembler::SparseMatrixAssembler)
  # ids = pattern.unknown_dofs
  pattern = assembler.pattern
  storage = assembler.stiffness_storage
  return @views SparseArrays.sparse!(
    pattern.Is, pattern.Js, storage[assembler.pattern.unknown_dofs],
    length(pattern.klasttouch), length(pattern.klasttouch), +, pattern.klasttouch,
    pattern.csrrowptr, pattern.csrcolval, pattern.csrnzval,
    pattern.csccolptr, pattern.cscrowval, pattern.cscnzval
  )
end

function SparseArrays.spdiagm(assembler::SparseMatrixAssembler)
  return SparseArrays.spdiagm(assembler.constraint_storage)
end

function _stiffness(assembler::SparseMatrixAssembler, ::KA.CPU)
  return SparseArrays.sparse!(assembler)
end

function stiffness(assembler::SparseMatrixAssembler)
  return _stiffness(assembler, KA.get_backend(assembler))
end

# TODO Need to specialize below for different field types
function update_dofs!(assembler::SparseMatrixAssembler, dirichlet_bcs::T) where T <: AbstractArray{DirichletBC}
  vars = assembler.dof.H1_vars

  if length(vars) != 1
    @assert false "multiple fspace not supported yet"
  end

  # fspace = vars[1].fspace

  # make this more efficient
  dirichlet_dofs = unique!(sort!(vcat(map(x -> x.bookkeeping.dofs, dirichlet_bcs)...)))
  update_dofs!(assembler, dirichlet_dofs)
  return nothing
end

function update_dofs!(assembler::SparseMatrixAssembler, dirichlet_dofs::T) where T <: AbstractArray{<:Integer, 1} 
  dirichlet_dofs = copy(dirichlet_dofs)
  unique!(sort!(dirichlet_dofs))
  update_dofs!(assembler.dof, dirichlet_dofs)

  # resize the resiual unkowns
  resize!(assembler.residual_unknowns, num_unknowns(assembler.dof))

  n_total_dofs = length(assembler.dof) - length(dirichlet_dofs)

  # TODO change to a good sizehint!
  resize!(assembler.pattern.Is, 0)
  resize!(assembler.pattern.Js, 0)
  resize!(assembler.pattern.unknown_dofs, 0)

  ND, NN = num_dofs_per_node(assembler.dof), num_nodes(assembler.dof)
  ids = reshape(1:length(assembler.dof), ND, NN)

  # TODO
  vars = assembler.dof.H1_vars
  fspace = vars[1].fspace

  n = 1
  # for fspace in fspaces
  for conns in values(fspace.elem_conns)
    dof_conns = @views reshape(ids[:, conns], ND * size(conns, 1), size(conns, 2))

    # for e in 1:num_elements(fspace)
    for e in 1:size(conns, 2)
    # AK.foraxes(conns, 2) do e
      # conn = dof_connectivity(fspace, e)
      conn = @views dof_conns[:, e]
      for temp in Iterators.product(conn, conn)
        if insorted(temp[1], dirichlet_dofs) || insorted(temp[2], dirichlet_dofs)
          # really do nothing here
        else
          push!(assembler.pattern.Is, temp[1] - count(x -> x < temp[1], dirichlet_dofs))
          push!(assembler.pattern.Js, temp[2] - count(x -> x < temp[2], dirichlet_dofs))
          push!(assembler.pattern.unknown_dofs, n)
        end
        n += 1
      end
    end
  end

  resize!(assembler.pattern.klasttouch, n_total_dofs)
  resize!(assembler.pattern.csrrowptr, n_total_dofs + 1)
  resize!(assembler.pattern.csrcolval, length(assembler.pattern.Is))
  resize!(assembler.pattern.csrnzval, length(assembler.pattern.Is))

  return nothing
end

function _zero_storage(asm::SparseMatrixAssembler, ::Val{:stiffness})
  fill!(asm.stiffness_storage, zero(eltype(asm.stiffness_storage)))
end

function _zero_storage(asm::SparseMatrixAssembler, ::Val{:residual_and_stiffness})
  _zero_storage(asm, Val{:residual}())
  fill!(asm.stiffness_storage, zero(eltype(asm.stiffness_storage)))
end