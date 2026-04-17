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
  RV <: AbstractArray{<:Union{<:Number, <:SVector}, 2}
}
  vals::RV                    # size (NQ_cell, nelem) of SVector{ND}
  is_constant::Bool           # skip re-evaluation after first call
  initialized::Base.RefValue{Bool}
end

function Adapt.adapt_structure(to, source::SourceContainer)
  SourceContainer(
    adapt(to, source.vals),
    source.is_constant,
    source.initialized,
  )
end

Base.length(bc::SourceContainer) = size(bc.vals, 2)

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

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Collection of body force containers, one per specified body force entry.
"""
struct Sources{
  RV           <: AbstractArray{<:Union{<:Number, <:SVector}, 2},
  SourceFuncs  <: NamedTuple
} <: AbstractBCs{SourceFuncs}
  source_block_ids::Vector{Int} # note this is the id order in FEC not the id in exodus
  source_block_names::Vector{String}
  source_caches::Vector{SourceContainer{RV}}
  source_funcs::SourceFuncs
end

function Sources(mesh, dof::DofManager, sources::Vector{Source})
  if length(sources) == 0
    return Sources(Int[], String[], SourceContainer{Matrix{Float64}}[], NamedTuple())
  end

  fspace = function_space(dof)
  ND = length(dof.var)
  caches = []
  funcs  = Function[]
  source_block_ids = Int[]
  source_block_names = String[]
  for source in sources
    block_name = source.block_name
    # block_id = indexin([block_name], collect(keys(fspace.ref_fes)))
    # @assert length(block_id) == 1
    # block_id = block_id[1]
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

    # push!(caches, SourceContainer(conns, ref_fe, vals, is_constant, Ref(false)))
    push!(caches, SourceContainer(vals, is_constant, Ref(false)))
    push!(funcs, source.func)
  end

  # what happens if they're not all the same type?
  caches = convert(Vector{typeof(caches[1])}, caches)

  syms = tuple(map(x -> Symbol("source_$x"), 1:length(caches))...)
  return Sources(
    source_block_ids, source_block_names, caches,
    NamedTuple{syms}(tuple(funcs...)),
  )
end

function Adapt.adapt(to, sources::Sources)
  # needed due to failures on 1.10 and 1.11
  if length(sources.source_caches) == 0
    caches = Vector{SourceContainer{Matrix{Float64}}}(undef, 0)
  else
    caches = map(x -> adapt(to, x), sources.source_caches)
  end

  return Sources(
    sources.source_block_ids,
    sources.source_block_names,
    caches,
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

function update_source_values!(sources::Sources, assembler, X, t)
  fspace = function_space(assembler)
  for (block_id, source, func) in zip(sources.source_block_ids, sources.source_caches, sources.source_funcs)
    if source.is_constant && source.initialized[]
      continue
    end
    ref_fe = fspace.ref_fes[block_id]
    conns_data = fspace.elem_conns.data
    coffset = fspace.elem_conns.offsets[block_id]
    _update_source_values!(source.vals, func, ref_fe, conns_data, coffset, X, t)
    source.initialized[] = true
  end
  return nothing
end
