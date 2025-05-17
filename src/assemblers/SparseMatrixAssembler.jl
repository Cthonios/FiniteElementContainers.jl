"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
General sparse matrix assembler that can handle first or second order
problems in time. 
"""
struct SparseMatrixAssembler{
  Dof      <: DofManager, 
  Pattern  <: SparsityPattern, 
  Storage1 <: AbstractArray{<:Number, 1},
  Storage2 <: AbstractField,
  Storage3 <: NamedTuple
} <: AbstractAssembler{Dof}
  dof::Dof
  pattern::Pattern
  constraint_storage::Storage1
  damping_storage::Storage1
  hessian_storage::Storage1
  mass_storage::Storage1
  residual_storage::Storage2
  residual_unknowns::Storage1
  scalar_quadarature_storage::Storage3 # useful for energy like calculations
  stiffness_storage::Storage1
  stiffness_action_storage::Storage2
  stiffness_action_unknowns::Storage1
end

# TODO this will not work for other than single H1 spaces
"""
$(TYPEDSIGNATURES)
Construct a ```SparseMatrixAssembler``` for a specific field type, 
e.g. ```H1Field```.
Can be used to create block arrays for mixed FEM problems.
"""
function SparseMatrixAssembler(dof::DofManager, type::Type{<:H1Field})
  pattern = SparsityPattern(dof, type)
  # constraint_storage = zeros(length(dof))
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  n_total_dofs = ND * NN
  constraint_storage = zeros(n_total_dofs)
  # constraint_storage = zeros(_dof_manager_vars(dof, type))
  constraint_storage[dof.H1_bc_dofs] .= 1.
  # fill!(constraint_storage, )
  # residual_storage = zeros(length(dof))
  damping_storage = zeros(num_entries(pattern))
  hessian_storage = zeros(num_entries(pattern))
  mass_storage = zeros(num_entries(pattern))
  residual_storage = create_field(dof, H1Field)
  residual_unknowns = create_unknowns(dof, H1Field)
  stiffness_storage = zeros(num_entries(pattern))
  stiffness_action_storage = create_field(dof, H1Field)
  stiffness_action_unknowns = create_unknowns(dof, H1Field)

  # setup quadrature scalar storage
  fspace = values(dof.H1_vars)[1].fspace
  scalar_quadarature_storage = Matrix{Float64}[]
  for (key, val) in pairs(fspace.ref_fes)
    NQ = ReferenceFiniteElements.num_quadrature_points(val)
    NE = size(getfield(fspace.elem_conns, key), 2)
    syms = map(x -> Symbol("quadrature_field_$x"), 1:NQ)
    field = L2ElementField(zeros(Float64, NQ, NE), tuple(syms...))
    push!(scalar_quadarature_storage, field)
    # push!(scalar_quadarature_storage, zeros(Float64, NQ, NE))
  end
  scalar_quadarature_storage = NamedTuple{keys(fspace.ref_fes)}(tuple(scalar_quadarature_storage...))

  return SparseMatrixAssembler(
    dof, pattern, 
    constraint_storage, 
    damping_storage, 
    hessian_storage,
    mass_storage,
    residual_storage, residual_unknowns,
    scalar_quadarature_storage,
    stiffness_storage, 
    stiffness_action_storage, stiffness_action_unknowns
  )
end

function SparseMatrixAssembler(::Type{<:H1Field}, vars...)
  dof = DofManager(vars...)
  return SparseMatrixAssembler(dof, H1Field)
end

function Base.show(io::IO, asm::SparseMatrixAssembler)
  println(io, "SparseMatrixAssembler")
  println(io, "  ", asm.dof)
end

"""
$(TYPEDSIGNATURES)
Specialization of of ```_assemble_element!``` for ```SparseMatrixAssembler```.
"""
function _assemble_element!(
  pattern::SparsityPattern, storage, K_el::SMatrix, el_id::Int, block_id::Int
)
  # figure out ids needed to update
  block_size = values(pattern.block_sizes)[block_id]
  block_offset = values(pattern.block_offsets)[block_id]
  # get range of ids
  start_id = (block_id - 1) * block_size + 
             (el_id - 1) * block_offset + 1
  end_id = start_id + block_offset - 1
  ids = start_id:end_id

  # get appropriate storage and update values
  @views storage[ids] += K_el[:]
  return nothing
end

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
TODO add symbol to interface
"""
function SparseArrays.sparse!(assembler::SparseMatrixAssembler, sym)
  # ids = pattern.unknown_dofs
  pattern = assembler.pattern
  # storage = assembler.stiffness_storage
  storage = getproperty(assembler, sym)
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

function constraint_matrix(assembler::SparseMatrixAssembler)
  # TODO specialize to CPU/GPU
  return SparseArrays.spdiagm(assembler)
end

function _mass(assembler::SparseMatrixAssembler, ::KA.CPU)
  return SparseArrays.sparse!(assembler, :mass_storage)
end

function _stiffness(assembler::SparseMatrixAssembler, ::KA.CPU)
  return SparseArrays.sparse!(assembler, :stiffness_storage)
end

# TODO probably only works for H1 fields
# TODO Need to specialize below for different field types
# TODO make keyword use_condensed more clear
# the use case here being to flag how to update the sparsity pattern
# constraint_storage is used to make a diagonal matrix of 1s and 0s to zero out element of
# the residual and stiffness appropriately without having to reshape, Is, Js, etc.
# when we want to change BCs which is slow

function update_dofs!(assembler::SparseMatrixAssembler, dirichlet_bcs; use_condensed=false)
  vars = assembler.dof.H1_vars

  if length(vars) != 1
    @assert false "multiple fspace not supported yet"
  end

  # dirichlet_dofs = dirichlet_bcs.bookkeeping.dofs
  dirichlet_dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, dirichlet_bcs)
  dirichlet_dofs = unique(sort(dirichlet_dofs))

  update_dofs!(assembler.dof, dirichlet_dofs)

  if use_condensed
    _update_dofs_condensed!(assembler)
  else
    _update_dofs!(assembler, dirichlet_dofs)
  end
  return nothing
end
# TODO Need to specialize below for different field types
# TODO make keyword use_condensed more clear
# the use case here being to flag how to update the sparsity pattern
# constraint_storage is used to make a diagonal matrix of 1s and 0s to zero out element of
# the residual and stiffness appropriately without having to reshape, Is, Js, etc.
# when we want to change BCs which is slow

function _update_dofs_condensed!(assembler::SparseMatrixAssembler)
  assembler.constraint_storage[assembler.dof.H1_unknown_dofs] .= 1.
  assembler.constraint_storage[assembler.dof.H1_bc_dofs] .= 0.
  return nothing
end

# TODO part of this method should be moved to SparsityPattern.jl
# TODO specialize on field type
# TODO probably only works on H1 write now
function _update_dofs!(assembler::SparseMatrixAssembler, dirichlet_dofs::T) where T <: AbstractArray{<:Integer, 1}

  # resize the resiual unkowns
  n_total_H1_dofs = num_nodes(assembler.dof) * num_dofs_per_node(assembler.dof)
  resize!(assembler.residual_unknowns, length(assembler.dof.H1_unknown_dofs))

  # n_total_dofs = length(assembler.dof) - length(dirichlet_dofs)
  n_total_dofs = n_total_H1_dofs - length(dirichlet_dofs)

  # TODO change to a good sizehint!
  resize!(assembler.pattern.Is, 0)
  resize!(assembler.pattern.Js, 0)
  resize!(assembler.pattern.unknown_dofs, 0)

  ND, NN = num_dofs_per_node(assembler.dof), num_nodes(assembler.dof)
  # ids = reshape(1:length(assembler.dof), ND, NN)
  ids = reshape(1:n_total_H1_dofs, ND, NN)

  # TODO
  vars = assembler.dof.H1_vars
  fspace = vars[1].fspace

  n = 1
  for conns in values(fspace.elem_conns)
    dof_conns = @views reshape(ids[:, conns], ND * size(conns, 1), size(conns, 2))

    for e in 1:size(conns, 2)
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
