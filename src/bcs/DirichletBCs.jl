abstract type AbstractDirichletBC{F} <: AbstractBC{F} end
# abstract type AbstractDirichletBCContainer{
#   IT, VT, IV, IM, VV
# } <: AbstractBCContainer{IT, VT, 1, IV, IM, VV} end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```DirichletBC````.
"""
struct DirichletBC{F} <: AbstractDirichletBC{F}
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
  return DirichletBC(func, sset_name, var_name)
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
struct DirichletBCContainer{
  IT <: Integer,
  VT <: Number,
  IV <: AbstractArray{IT, 1},
  VV <: AbstractArray{VT, 1}
} <: AbstractBCContainer
  dofs::IV
  nodes::IV
  vals::VV
  vals_dot::VV
  vals_dot_dot::VV
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBCContainer(mesh, dof::DofManager, dbc::DirichletBC)
  bk = BCBookKeeping(mesh, dof, dbc.var_name, sset_name=dbc.sset_name)

  # sort nodes and dofs for dirichlet bc
  dof_perm = _unique_sort_perm(bk.dofs)
  dofs = bk.dofs[dof_perm]
  nodes = bk.nodes[dof_perm]
  resize!(bk.dofs, length(dofs))
  resize!(bk.nodes, length(nodes))
  copyto!(bk.dofs, dofs)
  copyto!(bk.nodes, nodes)

  vals = zeros(length(bk.nodes))
  vals_dot = zeros(length(bk.nodes))
  vals_dot_dot = zeros(length(bk.nodes))
  return DirichletBCContainer(bk.dofs, bk.nodes, vals, vals_dot, vals_dot_dot)
end

function Base.length(bc::DirichletBCContainer)
  return length(bc.dofs)
end

# need checks on if field types are compatable
"""
$(TYPEDSIGNATURES)
CPU implementation for updating stored bc values 
based on the stored function
"""
function _update_bc_values!(bc::DirichletBCContainer, func, X, t, ::KA.CPU)
  ND = num_fields(X)
  for (n, node) in enumerate(bc.nodes)
    X_temp = @views SVector{ND, eltype(X)}(X[:, node])
    bc.vals[n] = func.func(X_temp, t)
    bc.vals_dot[n] = func.func_dot(X_temp, t)
    bc.vals_dot_dot[n] = func.func_dot_dot(X_temp, t)
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
  node = bc.nodes[I]

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
  bc.vals[I] = func.func(X_temp, t)
  bc.vals_dot[I] = func.func_dot(X_temp, t)
  bc.vals_dot_dot[I] = func.func_dot_dot(X_temp, t)
end
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
GPU kernel wrapper for updating bc values based on the stored function
"""
function _update_bc_values!(bc::DirichletBCContainer, func, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(bc, func, X, t, ndrange = length(bc))
  return nothing
end

# TODO change below names to be more specific to dbcs
function _update_field_dirichlet_bcs!(U, bc::DirichletBCContainer, ::KA.CPU)
  for (dof, val) in zip(bc.dofs, bc.vals)
    U[dof] = val
  end
  return nothing
end

function _update_field_dirichlet_bcs!(U, V, bc::DirichletBCContainer, ::KA.CPU)
  for (dof, val, val_dot) in zip(bc.dofs, bc.vals, bc.vals_dot)
    U[dof] = val
    V[dof] = val_dot
  end
  return nothing
end

function _update_field_dirichlet_bcs!(U, V, A, bc::DirichletBCContainer, ::KA.CPU)
  for (dof, val, val_dot, val_dot_dot) in zip(bc.dofs, bc.vals, bc.vals_dot, bc.vals_dot_dot)
    U[dof] = val
    V[dof] = val_dot
    A[dof] = val_dot_dot
  end
  return nothing
end

# COV_EXCL_START
KA.@kernel function _update_field_dirichlet_bcs_kernel!(U, bc::DirichletBCContainer)
  I = KA.@index(Global)
  dof = bc.dofs[I]
  U[dof] = bc.vals[I]
end
# COV_EXCL_STOP

# COV_EXCL_START
KA.@kernel function _update_field_dirichlet_bcs_kernel!(U, V, A, bc::DirichletBCContainer)
  I = KA.@index(Global)
  dof = bc.dofs[I]
  val = bc.vals[I]
  U[dof] = bc.vals[I]
  V[dof] = bc.vals_dot[I]
  A[dof] = bc.vals_dot_dot[I]
end
# COV_EXCL_STOP

function _update_field_dirichlet_bcs!(U, bc::DirichletBCContainer, backend::KA.Backend)
  kernel! = _update_field_dirichlet_bcs_kernel!(backend)
  kernel!(U, bc, ndrange = length(bc.vals))
  return nothing
end

function _update_field_dirichlet_bcs!(U, V, A, bc::DirichletBCContainer, backend::KA.Backend)
  kernel! = _update_field_dirichlet_bcs_kernel!(backend)
  kernel!(U, V, A, bc, ndrange = length(bc.vals))
  return nothing
end

function update_field_dirichlet_bcs!(U, bcs::NamedTuple)
  for bc in values(bcs)
    _update_field_dirichlet_bcs!(U, bc, KA.get_backend(bc))
  end
  return nothing
end

function update_field_dirichlet_bcs!(U, V, bcs::NamedTuple)
  for bc in values(bcs)
    _update_field_dirichlet_bcs!(U, V, bc, KA.get_backend(bc))
  end
end

function update_field_dirichlet_bcs!(U, V, A, bcs::NamedTuple)
  for bc in values(bcs)
    _update_field_dirichlet_bcs!(U, V, A, bc, KA.get_backend(bc))
  end
end

struct DirichletBCFunction{F1, F2, F3} <: AbstractBCFunction{F1}
  func::F1
  func_dot::F2
  func_dot_dot::F3
end

function DirichletBCFunction(func)
  func_dot = (x, t) -> ForwardDiff.derivative(z -> func(x, z), t)
  func_dot_dot = (x, t) -> ForwardDiff.derivative(z -> func_dot(x, z), t)
  return DirichletBCFunction(func, func_dot, func_dot_dot)
end

function create_dirichlet_bcs(mesh, dof::DofManager, dirichlet_bcs::Vector{<:DirichletBC})
  if length(dirichlet_bcs) == 0
    return NamedTuple(), NamedTuple()
  end

  syms = map(x -> Symbol("dirichlet_bc_$x"), 1:length(dirichlet_bcs))
  dirichlet_bc_funcs = NamedTuple{tuple(syms...)}(
    map(x -> DirichletBCFunction(x.func), dirichlet_bcs)
  )
  dirichlet_bcs = DirichletBCContainer.((mesh,), (dof,), dirichlet_bcs)

  if length(dirichlet_bcs) > 0
    temp_dofs = mapreduce(x -> x.dofs, vcat, dirichlet_bcs)
    temp_dofs = unique(sort(temp_dofs))
    update_dofs!(dof, temp_dofs)
  end

  dirichlet_bcs = NamedTuple{tuple(syms...)}(tuple(dirichlet_bcs...))

  return dirichlet_bcs, dirichlet_bc_funcs
end
