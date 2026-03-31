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
  block_name::Symbol
  var_name::Symbol

  function Source(var_name::String, func, block_name::String)
    new{typeof(func)}(func, Symbol(block_name), Symbol(var_name))
  end

  function Source(var_name::Symbol, func, block_name::Symbol)
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
  IT <: Integer,
  IV <: AbstractArray{IT, 1},
  RV <: AbstractArray{<:Union{<:Number, <:SVector}, 2},
  RE <: ReferenceFE
}
  element_conns::Connectivity{IT, IV}
  ref_fe::RE
  vals::RV                    # size (NQ_cell, nelem) of SVector{ND}
  is_constant::Bool           # skip re-evaluation after first call
  initialized::Base.RefValue{Bool}
end

function Adapt.adapt_structure(to, source::SourceContainer)
  SourceContainer(
    adapt(to, source.element_conns),
    adapt(to, source.ref_fe),
    adapt(to, source.vals),
    source.is_constant,
    source.initialized,
  )
end

Base.length(bc::SourceContainer) = size(bc.vals, 2)

function _update_source_values!(vals, func, ref_fe, conns, X, t)
  fec_axes(vals, 2) do e
    conn = connectivity(ref_fe, conns, e, 1)
    X_el = _element_level_fields_flat(X, ref_fe, conn)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      interps = map_interpolants(interps, X_el)
      vals[q, e] = func(interps.X_q, t)
    end
  end
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Collection of body force containers, one per specified body force entry.
"""
struct Sources{
  SourceCaches <: NamedTuple,
  SourceFuncs  <: NamedTuple
} <: AbstractBCs{SourceFuncs}
  source_block_ids::Vector{Int} # note this is the id order in FEC not the id in exodus
  source_block_names::Vector{Symbol}
  source_caches::SourceCaches
  source_funcs::SourceFuncs
end

function Sources(mesh, dof::DofManager, sources::Vector{Source})
  if length(sources) == 0
    return Sources(Int[], Symbol[], NamedTuple(), NamedTuple())
  end

  fspace = function_space(dof)
  ND = length(dof.var)
  caches = []
  funcs  = []
  source_block_ids = Int[]
  source_block_names = Symbol[]
  for (i, bf) in enumerate(sources)
    block_name = bf.block_name
    block_id = indexin([block_name], collect(keys(fspace.ref_fes)))
    @assert length(block_id) == 1
    block_id = block_id[1]
    push!(source_block_ids, block_id)
    push!(source_block_names, block_name)
    ref_fe     = getfield(fspace.ref_fes, block_name)
    conns_full = mesh.element_conns[block_name]
    nelem      = size(conns_full, 2)
    NQ         = num_cell_quadrature_points(ref_fe)

    conns = Connectivity([conns_full])
    # if ND == 1
    #   vals = zeros(Float64, NQ, nelem)
    # else
    vals  = zeros(SVector{ND, Float64}, NQ, nelem)
    # end

    # Detect constant: the is_constant flag is set by the caller (Carina)
    # via the Source struct. For now, default to false (re-evaluate every step).
    is_constant = false

    push!(caches, SourceContainer(conns, ref_fe, vals, is_constant, Ref(false)))
    push!(funcs, bf.func)
  end

  syms = tuple(map(x -> Symbol("source_$x"), 1:length(caches))...)
  return Sources(
    source_block_ids, source_block_names,
    NamedTuple{syms}(tuple(caches...)),
    NamedTuple{syms}(tuple(funcs...)),
  )
end

function Adapt.adapt(to, sources::Sources)
  return Sources(
    sources.source_block_ids,
    sources.source_block_names,
    adapt(to, sources.source_caches),
    adapt(to, sources.source_funcs)
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

function update_source_values!(sources::Sources, X, t)
  for (source, func) in zip(sources.source_caches, sources.source_funcs)
    if source.is_constant && source.initialized[]
      continue
    end
    _update_source_values!(source.vals, func, source.ref_fe, source.element_conns.data, X, t)
    source.initialized[] = true
  end
  return nothing
end
