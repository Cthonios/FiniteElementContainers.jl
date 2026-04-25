"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```NeumannBC````.
"""
struct NeumannBC{F} <: AbstractBC{F}
  func::F
  sset_name::String
  var_name::String

  """
  $(TYPEDEF)
  $(TYPEDSIGNATURES)
  $(TYPEDFIELDS)
  """
  function NeumannBC(var_name::String, func, sset_name::String)
    new{typeof(func)}(func, sset_name, var_name)
  end
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct NeumannBCContainer{
  IT <: Integer,
  IV <: AbstractVector{IT},
  RM <: AbstractMatrix{<:SVector}
} <: AbstractWeaklyEnforcedBCContainer{IT, IV, RM}
  element_conns::Connectivity{IT, IV}
  elements::IV
  sides::IV
  vals::RM

  function NeumannBCContainer(element_conns::Connectivity{IT, IV}, elements::IV, sides::IV, vals::RM) where {IT, IV, RM}
    new{IT, IV, RM}(element_conns, elements, sides, vals)
  end

  function NeumannBCContainer{IT, IV, RM}() where {IT, IV, RM}
    new{IT, IV, RM}(Connectivity{IT, IV}(), IV(undef, 0), IV(undef, 0), RM(undef, 0))
  end
end

_indices_type(::Vector{NeumannBCContainer{IT, IV, RM}}) where {IT, IV, RM} = IV
_vals_type(::Vector{NeumannBCContainer{IT, IV, RM}}) where {IT, IV, RM} = RM

function Adapt.adapt_structure(to, bc::NeumannBCContainer)
  el_conns = adapt(to, bc.element_conns)
  elements = adapt(to, bc.elements)
  sides = adapt(to, bc.sides)
  vals = adapt(to, bc.vals)
  return NeumannBCContainer(el_conns, elements, sides, vals)
end

function _update_bc_values!(vals, func, ref_fe, conns, sides, X, t)
  fec_foraxes(vals, 2) do e
    conn = connectivity(ref_fe, conns, e, 1)
    X_el = _element_level_fields(X, ref_fe, conn)
  
    for q in 1:num_surface_quadrature_points(ref_fe)
      side = sides[e]
      interps = MappedH1OrL2SurfaceInterpolants(ref_fe, X_el, q, side)
      vals[q, e] = func(interps.X_q, t)
    end
  end
end

struct NeumannBCFunction{F} <: AbstractBCFunction{F}
  func::F
end

function (f::NeumannBCFunction)(x, t)
  return f.func(x, t)
end

struct NeumannBCs{
  BCFuncs,
  IT      <: Integer,
  IV      <: AbstractVector{IT},
  RM      <: AbstractMatrix{<:SVector},
} <: AbstractBCs{BCFuncs}
  bc_caches::Vector{NeumannBCContainer{IT, IV, RM}}
  bc_funcs::BCFuncs
  block_ids::Vector{IT}
  block_names::Vector{String}

  function NeumannBCs(mesh, dof::DofManager, neumann_bcs::Vector{NeumannBC})
    ND = size(dof, 1)
    bc_caches = NeumannBCContainer{Int, Vector{Int}, Matrix{SVector{ND, Float64}}}[]
    if length(neumann_bcs) == 0
      return NeumannBCs(bc_caches, NeumannBCFunction[], Int[], String[])
    end

    bc_funcs = map(bc -> NeumannBCFunction(bc.func), neumann_bcs)
    block_ids = Int[]
    block_names = String[]
    for bc in neumann_bcs
      block_id, block_name, conns, elements, sides, vals = _setup_sideset(mesh, dof, bc)
      push!(bc_caches, NeumannBCContainer(conns, elements, sides, vals))
      push!(block_ids, block_id)
      push!(block_names, block_name)
    end
    return NeumannBCs(bc_caches, bc_funcs, block_ids, block_names)
  end

  function NeumannBCs{F}(mesh, dof::DofManager, neumann_bcs) where F
    ND = size(dof, 1)
    bc_caches = NeumannBCContainer{Int, Vector{Int}, Matrix{SVector{ND, Float64}}}[]
    if length(neumann_bcs) == 0
      return NeumannBCs(bc_caches, NeumannBCFunction{F}[], Int[], String[])
    end

    bc_funcs = map(bc -> NeumannBCFunction{F}(bc.func), neumann_bcs)
    block_ids = Int[]
    block_names = String[]
    for bc in neumann_bcs
      block_id, block_name, conns, elements, sides, vals = _setup_sideset_juliac_safe(mesh, dof, bc)
      push!(bc_caches, NeumannBCContainer(conns, elements, sides, vals))
      push!(block_ids, block_id)
      push!(block_names, block_name)
    end
    return NeumannBCs(bc_caches, bc_funcs, block_ids, block_names)
  end

  function NeumannBCs(bc_caches, bc_funcs, block_ids, block_names)
    new{typeof(bc_funcs), eltype(block_ids), _indices_type(bc_caches), _vals_type(bc_caches)}(
      bc_caches, bc_funcs, block_ids, block_names
    )
  end
end

function Adapt.adapt_structure(to, bcs::NeumannBCs)
  # NOTE
  # below logic is needed due to improper
  # adapt mapping for an empty array in julia 1.10/1.11
  # where Vector{T}(undef, 0) gets mappend to Vector{Any}
  if length(bcs.bc_caches) > 0
    bc_caches = map(x -> adapt(to, x), bcs.bc_caches)
  else
    temp_int = adapt(to, zeros(Int, 0))
    temp_vals = adapt(to, _vals_type(bcs.bc_caches)(undef, 0, 0))
    bc_caches = NeumannBCContainer{Int, typeof(temp_int), typeof(temp_vals)}[]
  end
  return NeumannBCs(
    bc_caches,
    bcs.bc_funcs,
    bcs.block_ids,
    bcs.block_names
  )
end

function update_bc_values!(bcs::NeumannBCs, assembler, X, t)
  fspace = function_space(assembler.dof)
  for n in axes(bcs.block_ids, 1)
    bc = bcs.bc_caches[n]
    block_id = bcs.block_ids[n]
    func = bcs.bc_funcs[n]
    ref_fe = block_reference_element(fspace, block_id)
    _update_bc_values!(bc.vals, func, ref_fe, bc.element_conns.data, bc.sides, X, t)
  end
  return nothing
end
