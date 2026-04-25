# Body forces: volumetric force density b(x,t) integrated over element blocks.
#
# Contributes -∫ Nᵢ · b dΩ to the residual (external force, same sign
# convention as Neumann BCs).
#
# User-facing: Source{F} — specifies block, component, and function.
# Internal:    SourceContainer — pre-evaluated vals at volume QPs.
# Collection:  Sources — holds named containers + original funcs.

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User-facing API to define a body force on an element block.
"""
struct Source{F} <: AbstractBC{F}
  func::F
  block_name::String
  var_name::String

  function Source(var_name::String, func, block_name::String)
    new{typeof(func)}(func, block_name, var_name)
  end
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal container for body forces — stores pre-evaluated values at
volume quadrature points for one block.
  TODO remove element_conns and ref_fes
  and have assemblers just ping the correct block
"""
struct SourceContainer{
  RM <: AbstractMatrix{<:SVector}
}
  vals::RM                    # size (NQ_cell, nelem) of SVector{ND}
  is_constant::Bool           # skip re-evaluation after first call
  initialized::Base.RefValue{Bool}

  function SourceContainer(vals::RM, is_constant, initialized) where RM
    new{RM}(vals, is_constant, initialized)
  end
end

_vals_type(::Vector{SourceContainer{RM}}) where RM = RM

function Adapt.adapt_structure(to, source::SourceContainer)
  SourceContainer(
    adapt(to, source.vals),
    source.is_constant,
    source.initialized,
  )
end

function _update_source_values!(vals, func, ref_fe, conns, coffset, X, t)
  fec_foraxes(vals, 2) do e
    conn = connectivity(ref_fe, conns, e, coffset)
    X_el = _element_level_fields_flat(X, ref_fe, conn)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      interps = map_interpolants(interps, X_el)
      vals[q, e] = func(interps.X_q, t)
    end
  end
end

struct SourceFunction{F} <: AbstractBCFunction{F}
  func::F
end

function (f::SourceFunction)(x, t)
  return f.func(x, t)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Collection of body force containers, one per specified body force entry.
"""
struct Sources{
  SourceFuncs,
  RM           <: AbstractMatrix{<:SVector},
} <: AbstractBCs{SourceFuncs}
  source_block_ids::Vector{Int} # note this is the id order in FEC not the id in exodus
  source_block_names::Vector{String}
  source_caches::Vector{SourceContainer{RM}}
  source_funcs::SourceFuncs

  function Sources{SourceFuncs, RM}(
    source_block_ids, source_block_names, source_caches, source_funcs
  ) where {SourceFuncs, RM}
    new{SourceFuncs, RM}(source_block_ids, source_block_names, source_caches, source_funcs)
  end

  # TODO should we really allow for scalar funcs for this in the case of scalar variables?
  function Sources(mesh, dof::DofManager, sources::Vector{Source}, ::Type{RT} = Float64) where RT <: Number
    ND = size(dof, 1)
    caches = SourceContainer{Matrix{SVector{ND, RT}}}[]
    funcs = SourceFunction[]
    if length(sources) == 0
      return Sources{typeof(funcs), Matrix{SVector{ND, RT}}}(Int[], String[], SourceContainer{Matrix{SVector{ND, RT}}}[], funcs)
    end

    fspace = function_space(dof)
    ND = length(dof.var)
    source_block_ids = Int[]
    source_block_names = String[]
    for source in sources
      block_name = source.block_name
      block_id = findfirst(x -> x == block_name, mesh.element_block_names)
      push!(source_block_ids, block_id)
      push!(source_block_names, block_name)
      NQ, NE = block_quadrature_size(fspace, block_id)
      # conns = Connectivity([conns_full])
      # if ND == 1
      #   vals = zeros(Float64, NQ, nelem)
      # else
      vals  = zeros(SVector{ND, Float64}, NQ, NE)
      # end

      # Detect constant: the is_constant flag is set by the caller (Carina)
      # via the Source struct. For now, default to false (re-evaluate every step).
      is_constant = false

      push!(caches, SourceContainer(vals, is_constant, Ref(false)))
      push!(funcs, SourceFunction(source.func))
    end

    # return Sources(source_block_ids, source_block_names, caches, funcs)
    return Sources{typeof(funcs), Matrix{SVector{ND, RT}}}(
      source_block_ids, source_block_names, caches, funcs
    )
  end
end

function Adapt.adapt(to, sources::Sources{SourceFuncs, RM}) where {SourceFuncs, RM}
  if length(sources.source_caches) > 0
    caches = map(x -> adapt(to, x), sources.source_caches)
  else
    temp = adapt(to, _vals_type(sources.source_caches)(undef, 0, 0))
    caches = SourceContainer{typeof(temp)}[]
  end

  return Sources{SourceFuncs, _vals_type(caches)}(
    sources.source_block_ids,
    sources.source_block_names,
    caches,
    sources.source_funcs
  )
end

function Base.show(io::IO, sources::Sources)
  type = typeof(sources).name.name
  for (n, (cache, func)) in enumerate(zip(sources.source_caches, sources.source_funcs))
    show(io, "$(type)_$n")
    # show(io, cache)
    show(io, func)
    show(io, "\n")
  end
end

function update_source_values!(sources::Sources, assembler, X, t)
  fspace = function_space(assembler)
  for n in axes(sources.source_block_ids, 1)
    block_id = sources.source_block_ids[n]
    cache = sources.source_caches[n]
    if cache.is_constant && cache.initialized[]
      continue
    end

    func = sources.source_funcs[n]
    ref_fe = block_reference_element(fspace, block_id)
    conns_data = fspace.elem_conns.data
    coffset = fspace.elem_conns.offsets[block_id]
    _update_source_values!(cache.vals, func, ref_fe, conns_data, coffset, X, t)
    cache.initialized[] = true
  end
  return nothing
end
