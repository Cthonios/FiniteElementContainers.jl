abstract type AbstractDirichletBC{F} <: AbstractBC{F} end

const EntityName = Union{Nothing, Symbol}

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```DirichletBC````.
"""
struct DirichletBC{F} <: AbstractDirichletBC{F}
  func::F
  block_name::EntityName
  nset_name::EntityName
  sset_name::EntityName
  var_name::Symbol
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBC(
  var_name::Symbol, func::Function;
  block_name::EntityName = nothing,
  nodeset_name::EntityName = nothing,
  sideset_name::EntityName = nothing
)
  return DirichletBC(func, block_name, nodeset_name, sideset_name, var_name)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBC(
  var_name::String, func::Function;
  block_name::Union{Nothing, String} = nothing,
  nodeset_name::Union{Nothing, String} = nothing,
  sideset_name::Union{Nothing, String} = nothing
)
  if block_name !== nothing
    block_name = Symbol(block_name)
  end
  
  if nodeset_name !== nothing
    nodeset_name = Symbol(nodeset_name)
  end

  if sideset_name !== nothing
    sideset_name = Symbol(sideset_name)
  end
  return DirichletBC(func, block_name, nodeset_name, sideset_name, Symbol(var_name))
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
} <: AbstractBCContainer{IV, RV}
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
  if dbc.block_name !== nothing
    bk = BCBookKeeping(mesh, dof, dbc.var_name, block_name=dbc.block_name)
  elseif dbc.nset_name !== nothing
    bk = BCBookKeeping(mesh, dof, dbc.var_name, nset_name=dbc.nset_name)
  elseif dbc.sset_name !== nothing
    bk = BCBookKeeping(mesh, dof, dbc.var_name, sset_name=dbc.sset_name)
  else
    @assert false
  end

  # bk = BCBookKeeping(mesh, dof, dbc.var_name, sset_name=dbc.sset_name)

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

"""
$(TYPEDSIGNATURES)
"""
function _update_bc_values!(bc::DirichletBCContainer, func, X, t)
  entity_foreach(bc.nodes) do n
    ND = num_fields(X)
    node = bc.nodes[n]
  
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

    bc.vals[n] = func.func(X_temp, t)
    bc.vals_dot[n] = func.func_dot(X_temp, t)
    bc.vals_dot_dot[n] = func.func_dot_dot(X_temp, t)
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

struct DirichletBCs{
  IV      <: AbstractArray{<:Integer, 1},
  RV      <: AbstractArray{<:Number, 1},
  BCFuncs <: NamedTuple
} <: AbstractBCs{BCFuncs}
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

function dirichlet_dofs(bcs::DirichletBCs)
  return unique(sort(mapreduce(x -> x.dofs, vcat, bcs.bc_caches)))
end

function _update_field_dirichlet_bcs!(U, bc::DirichletBCContainer)
  entity_foreach(bc.dofs) do I
    dof = bc.dofs[I]
    U[dof] = bc.vals[I]
  end
  return nothing
end

function _update_field_dirichlet_bcs!(U, V, A, bc::DirichletBCContainer)
  entity_foreach(bc.dofs) do I
    dof = bc.dofs[I]
    U[dof] = bc.vals[I]
    V[dof] = bc.vals_dot[I]
    A[dof] = bc.vals_dot_dot[I]
  end
  return nothing
end

function update_field_dirichlet_bcs!(U, bcs::DirichletBCs)
  for bc in values(bcs.bc_caches)
    _update_field_dirichlet_bcs!(U, bc)
  end
  return nothing
end

function update_field_dirichlet_bcs!(U, V, A, bcs::DirichletBCs)
  for bc in values(bcs.bc_caches)
    _update_field_dirichlet_bcs!(U, V, A, bc)
  end
end
