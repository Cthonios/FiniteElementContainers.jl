"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```DirichletBC````.
"""
struct DirichletBC{F} <: AbstractBC{F}
  func::F
  sset_name::Symbol
  var_name::Symbol
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBC(var_name::Symbol, sset_name::Symbol, func::Function)
  return DirichletBC{typeof(func)}(func, sset_name, var_name)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBC(var_name::String, sset_name::String, func::Function)
  return DirichletBC(Symbol(var_name), Symbol(sset_name), func)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct DirichletBCContainer{B, T, V} <: AbstractBCContainer{B, T, 1, V}
  bookkeeping::B
  vals::V
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBCContainer(dof::DofManager, dbc::DirichletBC)
  bk = BCBookKeeping(dof, dbc.var_name, dbc.sset_name)

  # sort nodes and dofs for dirichlet bc
  dof_perm = _unique_sort_perm(bk.dofs)
  dofs = bk.dofs[dof_perm]
  nodes = bk.nodes[dof_perm]
  resize!(bk.dofs, length(dofs))
  resize!(bk.nodes, length(nodes))
  copyto!(bk.dofs, dofs)
  copyto!(bk.nodes, nodes)

  vals = zeros(length(bk.nodes))
  return DirichletBCContainer{typeof(bk), eltype(vals), typeof(vals)}(
    bk, vals
  )
end

function Base.length(bc::DirichletBCContainer)
  return length(bc.bookkeeping.dofs)
end

# function bc_set_ids(bc::DirichletBCContainer)
#   return bc.bookkeeping.nodes
# end

# need checks on if field types are compatable
"""
$(TYPEDSIGNATURES)
CPU implementation for updating stored bc values 
based on the stored function
"""
function _update_bc_values!(bc::DirichletBCContainer, func, X, t, ::KA.CPU)
  ND = num_fields(X)
  for (n, node) in enumerate(bc.bookkeeping.nodes)
    X_temp = @views SVector{ND, eltype(X)}(X[:, node])
    # bc.vals[n] = bc.func(X_temp, t)
    bc.vals[n] = func(X_temp, t)
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
GPU kernel for updating stored bc values based on the stored function
"""
# COV_EXCL_START
KA.@kernel function _update_bc_values_kernel!(bc::DirichletBCContainer, func, X, t)
  I = KA.@index(Global)
  ND = num_fields(X)
  node = bc.bookkeeping.nodes[I]

  # hacky for now, but it works
  # can't do X[:, node] on the GPU, this results in a dynamic
  # function call
  if ND == 1
    X_temp = SVector{ND, eltype(X)}(X[1, node])
  elseif ND == 2
    X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node])
  elseif ND == 3
    X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node], X[3, node])
  end
  bc.vals[I] = func(X_temp, t)
end
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
GPU kernel wrapper for updating bc values based on the stored function
"""
function _update_bc_values!(bc::DirichletBCContainer, func, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(bc, func, X, t, ndrange=length(bc))
  return nothing
end

# TODO change below names to be more specific to dbcs
function _update_bcs!(bc::DirichletBCContainer, U, ::KA.CPU)
  for (dof, val) in zip(bc.bookkeeping.dofs, bc.vals)
    U[dof] = val
  end
  return nothing
end

# COV_EXCL_START
KA.@kernel function _update_bcs_kernel!(bc::DirichletBCContainer, U)
  I = KA.@index(Global)
  dof = bc.bookkeeping.dofs[I]
  val = bc.vals[I]
  U[dof] = val
end
# COV_EXCL_STOP

function _update_bcs!(bc::DirichletBCContainer, U, backend::KA.Backend)
  kernel! = _update_bcs_kernel!(backend)
  kernel!(bc, U, ndrange=length(bc.vals))
  return nothing
end

function create_dirichlet_bcs(dof::DofManager, dirichlet_bcs::Vector{<:DirichletBC})
  if length(dirichlet_bcs) == 1
    return NamedTuple(), NamedTuple()
  end

  syms = map(x -> Symbol("dirichlet_bc_$x"), 1:length(dirichlet_bcs))
  # dbcs = NamedTuple{tuple(syms...)}(tuple(dbcs...))
  # dbcs = DirichletBCContainer(dbcs, size(assembler.dof.H1_vars[1].fspace.coords, 1))
  dirichlet_bc_funcs = NamedTuple{tuple(syms...)}(
    map(x -> x.func, dirichlet_bcs)
  )
  dirichlet_bcs = DirichletBCContainer.((dof,), dirichlet_bcs)

  if length(dirichlet_bcs) > 0
    temp_dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, dirichlet_bcs)
    temp_dofs = unique(sort(temp_dofs))
    update_dofs!(dof, temp_dofs)
  end

  dirichlet_bcs = NamedTuple{tuple(syms...)}(tuple(dirichlet_bcs...))

  return dirichlet_bcs, dirichlet_bc_funcs
end