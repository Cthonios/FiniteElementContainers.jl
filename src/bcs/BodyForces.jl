# Body forces: volumetric force density b(x,t) integrated over element blocks.
#
# Contributes -∫ Nᵢ · b dΩ to the residual (external force, same sign
# convention as Neumann BCs).
#
# User-facing: BodyForce{F} — specifies block, component, and function.
# Internal:    BodyForceContainer — pre-evaluated vals at volume QPs.
# Collection:  BodyForces — holds named containers + original funcs.

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User-facing API to define a body force on an element block.
"""
struct BodyForce{F} <: AbstractBC{F}
  func::F
  block_name::Symbol
  var_name::Symbol

  function BodyForce(var_name::Symbol, func, block_name::Symbol)
    new{typeof(func)}(func, block_name, var_name)
  end
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal container for body forces — stores pre-evaluated values at
volume quadrature points for one block.
"""
struct BodyForceContainer{
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

function Adapt.adapt_structure(to, bc::BodyForceContainer)
  BodyForceContainer(
    adapt(to, bc.element_conns),
    bc.ref_fe,
    adapt(to, bc.vals),
    bc.is_constant,
    bc.initialized,
  )
end

Base.length(bc::BodyForceContainer) = size(bc.vals, 2)

function _update_body_force_values!(vals, func, ref_fe, conns, X, t)
  for e in axes(vals, 2)
    conn = connectivity(ref_fe, conns, e, 1)
    X_el = _element_level_fields(X, ref_fe, conn)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = MappedH1OrL2Interpolants(ref_fe, X_el, q)
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
struct BodyForces{
  BCCaches <: NamedTuple,
  BCFuncs  <: NamedTuple
} <: AbstractBCs{BCFuncs}
  bc_caches::BCCaches
  bc_funcs::BCFuncs
end

function BodyForces(mesh, dof::DofManager, body_forces::Vector{BodyForce})
  if length(body_forces) == 0
    return BodyForces(NamedTuple(), NamedTuple())
  end

  fspace = function_space(dof)
  ND = length(dof.var)
  caches = []
  funcs  = []

  for (i, bf) in enumerate(body_forces)
    block_name = bf.block_name
    ref_fe     = getfield(fspace.ref_fes, block_name)
    conns_full = mesh.element_conns[block_name]
    nelem      = size(conns_full, 2)
    NQ         = num_cell_quadrature_points(ref_fe)

    conns = Connectivity([conns_full])
    vals  = zeros(SVector{ND, Float64}, NQ, nelem)

    # Detect constant: the is_constant flag is set by the caller (Carina)
    # via the BodyForce struct. For now, default to false (re-evaluate every step).
    is_constant = false

    push!(caches, BodyForceContainer(conns, ref_fe, vals, is_constant, Ref(false)))
    push!(funcs, bf.func)
  end

  syms = tuple(map(x -> Symbol("body_force_$x"), 1:length(caches))...)
  return BodyForces(
    NamedTuple{syms}(tuple(caches...)),
    NamedTuple{syms}(tuple(funcs...)),
  )
end

function update_bc_values!(bcs::BodyForces, X, t)
  for (bc, func) in zip(bcs.bc_caches, bcs.bc_funcs)
    if bc.is_constant && bc.initialized[]
      continue
    end
    _update_body_force_values!(bc.vals, func, bc.ref_fe, bc.element_conns.data, X, t)
    bc.initialized[] = true
  end
  return nothing
end
