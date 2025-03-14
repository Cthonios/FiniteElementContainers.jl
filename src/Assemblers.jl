struct SparsityPattern{
  I <: AbstractArray{Int, 1},
  R <: AbstractArray{Float64, 1},
}
  Is::I
  Js::I
  unknown_dofs::I
  block_sizes::I
  block_offsets::I
  # cache arrays
  klasttouch::I
  csrrowptr::I
  csrcolval::I
  csrnzval::R
  # additional cache arrays
  csccolptr::I
  cscrowval::I
  cscnzval::R
end

# TODO won't work for H(div) or H(curl) yet
function SparsityPattern(dof)

  # get number of dofs for creating cache arrays
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  n_total_dofs = NN * ND

  n_blocks = length(dof.H1_vars[1].fspace.ref_fes)

  # first get total number of entries in a stupid matter
  n_entries = 0
  block_sizes = Vector{Int64}(undef, n_blocks)
  block_offsets = Vector{Int64}(undef, n_blocks)

  # TODO need to specialize for componentarray backend
  # for (n, block) in enumerate(valkeys(dof.vars[1].fspace.elem_conns))
  #   ids = reshape(1:n_total_dofs, ND, NN)
  #   conn = getproperty(dof.vars[1].fspace.elem_conns, block)
  #   conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))
  for (n, conn) in enumerate(values(dof.H1_vars[1].fspace.elem_conns))
    ids = reshape(1:n_total_dofs, ND, NN)
    # conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))
    # display(block)
    conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))

    # hacky for now
    if eltype(conn) <: SVector
      n_entries += length(conn[1])^2 * length(conn)
      block_sizes[n] = length(conn)
      block_offsets[n] = length(conn[1])^2
    else
      n_entries += size(conn, 1)^2 * size(conn, 2)
      block_sizes[n] = size(conn, 2)
      block_offsets[n] = size(conn, 1)^2
    end
  end

  # setup pre-allocated arrays based on number of entries found above
  Is = Vector{Int64}(undef, n_entries)
  Js = Vector{Int64}(undef, n_entries)
  unknown_dofs = Vector{Int64}(undef, n_entries)

  # now loop over function spaces and elements
  n = 1
  # for fspace in fspaces
  # for block in valkeys(dof.vars[1].fspace.elem_conns)
  for conn in values(dof.H1_vars[1].fspace.elem_conns)
    ids = reshape(1:n_total_dofs, ND, NN)
    # conn = getproperty(dof.vars[1].fspace.elem_conns, block)
    block_conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))

    # for e in 1:num_elements(fspace)
    for e in axes(block_conn, 2)
      # conn = dof_connectivity(fspace, e)
      conn = @views block_conn[:, e]
      for temp in Iterators.product(conn, conn)
        Is[n] = temp[1]
        Js[n] = temp[2]
        unknown_dofs[n] = n
        n += 1
      end
    end
  end

  # residuals = create_fields(dof) #|> vec
  # stiffnesses = zeros(Float64, size(Is))

  # create caches
  klasttouch = zeros(Int64, n_total_dofs)
  csrrowptr  = zeros(Int64, n_total_dofs + 1)
  csrcolval  = zeros(Int64, length(Is))
  csrnzval   = zeros(Float64, length(Is))

  csccolptr = Vector{Int64}(undef, 0)
  cscrowval = Vector{Int64}(undef, 0)
  cscnzval  = Vector{Float64}(undef, 0)

  # return SparsityPattern{
  #   Float64, Int64, 
  #   typeof(Is), typeof(Js), 
  #   typeof(unknown_dofs), 
  #   typeof(block_sizes), typeof(block_offsets),
  #   # cache arrays
  #   typeof(klasttouch), typeof(csrrowptr), typeof(csrcolval), typeof(csrnzval),
  #   # additional cache arrays
  #   typeof(csccolptr), typeof(cscrowval), typeof(cscnzval)
  # }(
  return SparsityPattern(
    Is, Js, 
    unknown_dofs, 
    block_sizes, block_offsets, 
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval
  )
end

num_entries(s::SparsityPattern) = length(s.Is)

abstract type AbstractAssembler{Dof} end

# TODO should we add different field type
# residual storage?
# Also, how should we set up a general mixed field
# sparse assembler... using blockarrays of sparsearrays?
# or just one big complex sparse array?
#
# similar to FunctionSpace, we can likely initialize
# SparseMatrixAssembler based on a provided field type
#
# need to figure out how to handle mixed spaces
#
struct SparseMatrixAssembler{
  Dof, Pattern, 
  Storage1 <: AbstractArray{<:Number},
  Storage2 <: H1Field
} <: AbstractAssembler{Dof}
  dof::Dof
  pattern::Pattern
  constraint_storage::Storage1
  residual_storage::Storage2
  stiffness_storage::Storage1
end

function SparseMatrixAssembler(dof::DofManager)
  pattern = SparsityPattern(dof)
  constraint_storage = zeros(length(dof))
  constraint_storage[dof.H1_bc_dofs] .= 1.
  # residual_storage = zeros(length(dof))
  residual_storage = create_field(dof, H1Field)
  stiffness_storage = zeros(num_entries(pattern))
  return SparseMatrixAssembler(
    dof, pattern, 
    constraint_storage, residual_storage, stiffness_storage
  )
end

function SparseMatrixAssembler(vars...)
  dof = DofManager(vars...)
  return SparseMatrixAssembler(dof)
end 

function Base.show(io::IO, asm::SparseMatrixAssembler)
  println(io, "SparseMatrixAssembler")
  println(io, "  ", asm.dof)
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

"""
$(TYPEDSIGNATURES)
Assembly method for a scalar field stored as a size 1 vector
"""
function assemble!(
  global_val::Vector, 
  # block_num, e, local_val
  local_val, 
  fspace, dof_conn, e, block_num
)
  global_val[1] += local_val
  return nothing
end

"""
$(TYPEDSIGNATURES)
generic assembly method that directly goes into a vector
"""
function assemble!(
  R::V1, R_el::V2, conn::V3
) where {V1 <: AbstractArray{<:Number}, 
         V2 <: AbstractArray{<:Number},
         V3 <: AbstractArray{<:Integer}}

  for i in axes(conn, 1)
    R[conn[i]] += R_el[i]
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
Assembly method for residuals
TODO this will likely break on mixed space problems
"""
function assemble!(
  global_val::H1Field, 
  # block_num, e, local_val
  local_val, 
  fspace, dof_conn, e, block_num
)
  # dof_conn = dof_connectivity(fspace, e)
  # dof_conn = @views dof_connectivity(fspace, e, block_num, size(global_val, 1))
  FiniteElementContainers.assemble!(global_val, local_val, dof_conn)
  return nothing
end

function assemble!(
  asm::SparseMatrixAssembler, K_el::SMatrix, 
  fspace, dof_conn, el_id::Int, block_id::Int
)
  start_id = (block_id - 1) * asm.pattern.block_sizes[block_id] + 
             (el_id - 1) * asm.pattern.block_offsets[block_id] + 1
  end_id = start_id + asm.pattern.block_offsets[block_id] - 1
  ids = start_id:end_id
  @views asm.stiffness_storage[ids] += K_el[:]
  return nothing
end

# wrapper
function assemble!(
  assembler::SparseMatrixAssembler,
  func::F,
  U::T,
  sym::Symbol
) where {F, T}
  assemble!(assembler, func, U, Val{sym}())
end

KA.@kernel function _assemble_residual_kernel!(asm, func, U, X, conns, ref_fe, block_id)
  E = KA.@index(Global)

  # some helpers
  NDim = size(X, 1)
  ND, NN = num_dofs_per_node(asm.dof), num_nodes(asm.dof)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  # NDim = ReferenceFiniteElements.num_dimensions(ref_fe)
  NxNDof = NNPE * ND
  # IDs = reshape(1:length(asm.dof), ND, size(U, 2))

  # extract element level stuff
  u_el = @views SMatrix{ND, NNPE, Float64, NxNDof}(U[:, conns[:, E]])
  x_el = @views SMatrix{NDim, NNPE, Float64, NDim * NNPE}(X[:, conns[:, E]])
  # dof_conns = reshape(IDs[:, conns], ND * length(conns))

  # assemble element level residual
  R_el = zeros(SVector{NxNDof, Float64})
  for q in 1:num_quadrature_points(ref_fe)
    interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)

    # dynamic method invocation isn't working here. 
    # R_q = func(interps, u_el)
    # R_el = R_el + R_q

    X_q, N, ∇N_X, JxW = interps.X_q, interps.N, interps.∇N_X, interps.JxW
    ∇u_q = u_el * ∇N_X
    f = 2. * π^2 * sin(2π * X_q[1]) * sin(4π * X_q[2])
    R_q = ∇u_q * ∇N_X' - N' * f
    R_el = R_el + JxW * R_q[:]
  end

  # now assemble
  # TODO won't work for problems with more than one dof...
  # need to create dof_conns
  for i in axes(conns, 1)
    Atomix.@atomic asm.residual_storage.vals[conns[i]] += R_el[i]
  end
end

KA.@kernel function _assemble_stiffness_kernel!(asm, func, U, X, conns, ref_fe, block_id)
  E = KA.@index(Global)

  # some helpers
  NDim = size(X, 1)
  ND, NN = num_dofs_per_node(asm.dof), num_nodes(asm.dof)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  # NDim = ReferenceFiniteElements.num_dimensions(ref_fe)
  NxNDof = NNPE * ND

  # extract element level stuff
  u_el = @views SMatrix{ND, NNPE, Float64, NxNDof}(U[:, conns[:, E]])
  x_el = @views SMatrix{NDim, NNPE, Float64, NDim * NNPE}(X[:, conns[:, E]])

  # assemble element level residual
  K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})
  for q in 1:num_quadrature_points(ref_fe)
    interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)

    # dynamic method invocation isn't working here. 
    # R_q = func(interps, u_el)
    # R_el = R_el + R_q

    X_q, N, ∇N_X, JxW = interps.X_q, interps.N, interps.∇N_X, interps.JxW
    K_q = ∇N_X * ∇N_X'
    K_el = K_el + JxW * K_q
  end

  # now assemble
  # TODO won't work for problems with more than one dof...
  # need to create dof_conns
  # start_id = (block_id - 1) * asm.pattern.block_sizes[block_id] + 
  #            (el_id - 1) * asm.pattern.block_offsets[block_id] + 1
  # end_id = start_id + asm.pattern.block_offsets[block_id] - 1
  # ids = start_id:end_id
  # for id in ids
  #   Atomix.@atomic asm.stiffness_storage[id] += K_el[:]
  # end
end

# newer method try to dispatch based on field type
function assemble!(
  assembler::SparseMatrixAssembler,
  residual_func::F,
  U::T,
  ::Val{:residual}
) where {F <: Function, T <: H1Field}
  # TODO hack for now, only grabbing first var fspace
  # fspace = getproperty(assembler.dof.H1_vars, :H1_var_1)
  fspace = assembler.dof.H1_vars[1].fspace
  for (block_id, conns) in enumerate(values(fspace.elem_conns))
    ref_fe = values(fspace.ref_fes)[block_id]
    # backend = KA.get_backend(assembler)
    # TODO eventually dispatch on CPU vs. GPU
    kernel! = _assemble_residual_kernel!(KA.get_backend(assembler))
    kernel!(assembler, residual_func, U, fspace.coords, conns, ref_fe, block_id, ndrange = size(conns, 2))
  end
end

function assemble!(
  assembler::SparseMatrixAssembler,
  tangent_func::F,
  U::T,
  ::Val{:stiffness}
) where {F <: Function, T <: H1Field}
  # TODO hack for now, only grabbing first var fspace
  # fspace = getproperty(assembler.dof.H1_vars, :H1_var_1)
  fspace = assembler.dof.H1_vars[1].fspace
  for (block_id, conns) in enumerate(values(fspace.elem_conns))
    ref_fe = values(fspace.ref_fes)[block_id]
    # backend = KA.get_backend(assembler)
    # TODO eventually dispatch on CPU vs. GPU

    # TODO going to need to query the start/end ids for the block assembler
    kernel! = _assemble_stiffness_kernel!(KA.get_backend(assembler))
    kernel!(assembler, tangent_func, U, fspace.coords, conns, ref_fe, block_id, ndrange = size(conns, 2))
  end
end

# TODO hardcoded for H1
function assemble!(
  R,
  assembler::SparseMatrixAssembler,
  residual_func::F,
  U::T
) where {F <: Function, T <: AbstractArray}

  R .= zero(eltype(R))
  # fill(R.vals, zero(eltype(R)))

  vars = assembler.dof.H1_vars

  if length(vars) != 1
    @assert false "multiple fspace not supported yet"
  end

  fspace = vars[1].fspace

  for (block_id, conns) in enumerate(values(fspace.elem_conns))
    ref_fe = values(fspace.ref_fes)[block_id]
    assemble!(R, assembler, residual_func, U, fspace.coords, conns, ref_fe)
  end
end

function assemble!(
  R,
  assembler::SparseMatrixAssembler,
  residual_func,
  U,
  coords,
  conns, 
  # dof_conns,
  ref_fe
)
  NDim = size(coords, 1)
  ND, NN = num_dofs_per_node(assembler.dof), num_nodes(assembler.dof)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  # NDim = ReferenceFiniteElements.num_dimensions(ref_fe)
  NxNDof = NNPE * ND
  ids = reshape(1:length(assembler.dof), ND, NN)
  dof_conns = @views reshape(ids[:, conns], ND * size(conns, 1), size(conns, 2))

  # AK.foraxes(conns, 2) do e
  for e in 1:size(conns, 2)
    u_el = @views SMatrix{ND, NNPE, Float64, NxNDof}(U[dof_conns[:, e]])
    x_el = @views SMatrix{NDim, NNPE, Float64, NDim * NNPE}(coords[:, conns[:, e]])
    # K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})
    # R_el = zeros(SMatrix{ND, NNPE, Float64, NxNDof})
    R_el = zeros(SVector{NxNDof, Float64})

    for q in 1:num_quadrature_points(ref_fe)
      interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
      R_q = residual_func(interps, u_el)
      R_el = R_el + R_q
    end

    # assemble!(assembler, R_el, e, block_id)
    # Note this method is in old stuff TODO
    @views assemble!(R, R_el, dof_conns[:, e])
  end
end

# for sparse matrix
# TODO hardcoded for H1
function assemble!(
  assembler::SparseMatrixAssembler, 
  tangent_func,
  # U::T, vars...
  U::T
) where T <: AbstractArray

  vars = assembler.dof.H1_vars

  if length(vars) != 1
    @assert false "multiple fspace not supported yet"
  end

  fspace = vars[1].fspace

  NDim = size(fspace.coords, 1)
  ND, NN = num_dofs_per_node(assembler.dof), num_nodes(assembler.dof)
  ids = reshape(1:length(assembler.dof), ND, NN)
  # for (block_id, block) in enumerate(valkeys(fspace.elem_conns))
  for (block_id, conns) in enumerate(values(fspace.elem_conns))
    ref_fe = values(fspace.ref_fes)[block_id]
    NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
    NxNDof = NNPE * ND
    # conns = getproperty(fspace.elem_conns, block)
    dof_conns = @views reshape(ids[:, conns], ND * size(conns, 1), size(conns, 2))

    # accelerated kernels drop in here
    for e in 1:size(conns, 2)
      u_el = @views U[dof_conns[:, e]]
      x_el = @views SMatrix{NDim, NNPE, Float64, NDim * NNPE}(fspace.coords[:, conns[:, e]])
      K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

      for q in 1:num_quadrature_points(ref_fe)
        interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
        K_q = tangent_func(interps, u_el)
        K_el = K_el + K_q
      end

      assemble!(assembler, K_el, dof_conns[:, e], fspace, e, block_id)
    end
  end
end

create_bcs(asm::SparseMatrixAssembler, type) = create_bcs(asm.dof, type)
create_field(asm::SparseMatrixAssembler, type) = create_field(asm.dof, type)
create_unknowns(asm::SparseMatrixAssembler) = create_unknowns(asm.dof)

# this thing should probably always be done on the CPU
# and then moved to the GPU
# TODO hardcoded for H1 fields
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

function update_field!(U, asm::SparseMatrixAssembler, Uu, Ubc)
  update_field!(U, asm.dof, Uu, Ubc)
  return nothing
end
