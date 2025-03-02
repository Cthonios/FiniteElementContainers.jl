abstract type AbstractBCBookKeeping{S, T <: Integer, V <: AbstractArray{T, 1}} end

struct BCBookKeeping{S, T, V} <: AbstractBCBookKeeping{S, T, V}
  blocks::V
  dofs::V
  elements::V
  nodes::V
  sides::V
end

# TODO hardcoded for H1 spaces right now.

# TODO also need to adapt this to differ on what var_name we look for based on build_dofs_array
# e.g. if it's neumann and a vector look for :u but if it's dirichlet and a vector look for :u_x
function BCBookKeeping(dof::NewDofManager, var_name::Symbol, sset_name::Symbol; build_dofs_array=false)
  # need to extract the var from dof based on teh symbol name
  var_index = 0
  dof_index = 0
  found = false
  for (vi, var) in enumerate(dof.H1_vars)
    for name in names(var)
      dof_index = dof_index + 1
      if var_name == name
        var_index = vi
        found = true
        break
      end
    end
  end

  if build_dofs_array
    @assert found == true "Failed to find variable $var_name"
  end

  @assert dof_index <= length(mapreduce(x -> names(x), +, dof.H1_vars)) "Found invalid dof index"

  fspace = dof.H1_vars[var_index].fspace
  elems = getproperty(fspace.sideset_elems, sset_name)
  nodes = getproperty(fspace.sideset_nodes, sset_name)
  sides = getproperty(fspace.sideset_sides, sset_name)

  blocks = Vector{Int64}(undef, 0)

  # gather the blocks that are present in this sideset
  for (n, val) in enumerate(values(fspace.elem_id_maps))
    # note these are the local elem id to the block, e.g. starting from 1.
    indices_in_sset = indexin(val, elems)
    filter!(x -> x !== nothing, indices_in_sset)
    
    if length(indices_in_sset) > 0
      # add stuff to arrays
      push!(blocks, n)
    end
  end

  # setting up dofs for use in dirichlet bcs
  if build_dofs_array
    all_dofs = reshape(1:length(dof), num_dofs_per_node(dof), num_nodes(dof))
    dofs = all_dofs[dof_index, nodes]
  else
    dofs = Vector{Int64}(undef, 0)
  end

  return BCBookKeeping{sset_name, Int64, typeof(blocks)}(blocks, dofs, elems, nodes, sides)
end

# actual bcs
abstract type AbstractBC{S, B <: AbstractBCBookKeeping, F <: Function, V <: AbstractArray{<:Number, 1}} end
name(::AbstractBC{S, B, F, V}) where {S, B, F, V} = S

# function Base.NamedTuple(bcs::AbstractArray{T, 1}) where T <: AbstractBC
#   syms = map(x -> name(x), bcs)
#   return NamedTuple{syms}(bcs)
# end

struct DirichletBC{S, B, F, V} <: AbstractBC{S, B, F, V}
  bookkeeping::B
  func::F
  vals::V
end

function DirichletBC(dof::NewDofManager, var_name::Symbol, sset_name::Symbol, func::Function)
  bookkeeping = BCBookKeeping(dof, var_name, sset_name; build_dofs_array=true)
  vals = zeros(Float64, length(bookkeeping.nodes))
  sym = Symbol(var_name, :_, sset_name)
  return DirichletBC{sym, typeof(bookkeeping), typeof(func), typeof(vals)}(bookkeeping, func, vals)
end

struct NeumannBC{S, B, F, V} <: AbstractBC{S, B, F, V}
  bookkeeping::B
  func::F
  vals::V
end

# TODO need to hack the var_name thing
function NeumannBC(dof::NewDofManager, var_name::Symbol, sset_name::Symbol, func::Function)
  bookkeeping = BCBookKeeping(dof, var_name, sset_name)
  vals = zeros(Float64, length(bookkeeping.elements))
  sym = Symbol(var_name, :_, sset_name) # TODO maybe add func name?
  return DirichletBC{sym, typeof(bookkeeping), typeof(func), typeof(vals)}(bookkeeping, func, vals)
end
