abstract type AbstractDirichletBC{F} <: AbstractBC{F} end

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
  IV <: AbstractArray{<:Integer, 1},
  RV <: AbstractArray{<:Number, 1}
} <: AbstractBCContainer
  dofs::IV
  nodes::IV
  vals::RV
  vals_dot::RV
  vals_dot_dot::RV
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

function Adapt.adapt_structure(to, bc::DirichletBCContainer)
  dofs = adapt(to, bc.dofs)
  nodes = adapt(to, bc.nodes)
  vals = adapt(to, bc.vals)
  vals_dot = adapt(to, bc.vals_dot)
  vals_dot_dot = adapt(to, bc.vals_dot_dot)
  return DirichletBCContainer(dofs, nodes, vals, vals_dot, vals_dot_dot)
end

function Base.length(bc::DirichletBCContainer)
  return length(bc.dofs)
end

function Base.show(io::IO, bc::DirichletBCContainer)
  println(io, "$(typeof(bc).name.name):")
  println(io, "  Number of active dofs     = $(length(bc.dofs))")
  println(io, "  Number of active nodes    = $(length(bc.nodes))")
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

struct DirichletBCs{
  IV      <: AbstractArray{<:Integer, 1},
  RV      <: AbstractArray{<:Number, 1},
  BCFuncs <: NamedTuple
}
  bc_caches::Vector{DirichletBCContainer{IV, RV}}
  bc_funcs::BCFuncs
end

function DirichletBCs(mesh, dof, dirichlet_bcs)

  if length(dirichlet_bcs) == 0
    bc_caches = DirichletBCContainer{Vector{Int}, Vector{Float64}}[]
    bc_funcs = NamedTuple()
    return DirichletBCs(bc_caches, bc_funcs)
  end

  syms = map(x -> Symbol("dirichlet_bc_$x"), 1:length(dirichlet_bcs))
  dirichlet_bc_funcs = NamedTuple{tuple(syms...)}(
    map(x -> DirichletBCFunction(x.func), dirichlet_bcs)
  )
  dirichlet_bcs = DirichletBCContainer.((mesh,), (dof,), dirichlet_bcs)

  temp_dofs = mapreduce(x -> x.dofs, vcat, dirichlet_bcs)
  temp_dofs = unique(sort(temp_dofs))
  update_dofs!(dof, temp_dofs)

  return DirichletBCs(dirichlet_bcs, dirichlet_bc_funcs)
end

function Adapt.adapt_structure(to, bcs::DirichletBCs)
  return DirichletBCs(
    map(x -> adapt(to, x), bcs.bc_caches),
    adapt(to, bcs.bc_funcs)
  )
end

function Base.length(bcs::DirichletBCs)
  return length(bcs.bc_caches)
end

function Base.show(io::IO, bcs::DirichletBCs)
  for (n, (cache, func)) in enumerate(zip(bcs.bc_caches, bcs.bc_funcs))
    show(io, "Dirichlet_BC_$n")
    show(io, cache)
    show(io, func)
  end
end

function update_bc_values!(bcs::DirichletBCs, X, t)
  update_bc_values!(bcs.bc_caches, bcs.bc_funcs, X, t)
  return nothing
end

function update_field_dirichlet_bcs!(U, bcs::DirichletBCs)
  for bc in values(bcs.bc_caches)
    _update_field_dirichlet_bcs!(U, bc, KA.get_backend(bc))
  end
  return nothing
end

function update_field_dirichlet_bcs!(U, V, bcs::DirichletBCs)
  for bc in values(bcs.bc_caches)
    _update_field_dirichlet_bcs!(U, V, bc, KA.get_backend(bc))
  end
end

function update_field_dirichlet_bcs!(U, V, A, bcs::DirichletBCs)
  for bc in values(bcs.bc_caches)
    _update_field_dirichlet_bcs!(U, V, A, bc, KA.get_backend(bc))
  end
end
