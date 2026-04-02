"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```NeumannBC````.
"""
struct NeumannBC{F} <: AbstractBC{F}
  func::F
  sset_name::Symbol
  var_name::Symbol

  """
  $(TYPEDEF)
  $(TYPEDSIGNATURES)
  $(TYPEDFIELDS)
  """
  function NeumannBC(var_name::Symbol, func, sset_name::Symbol)
    new{typeof(func)}(func, sset_name, var_name)
  end
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function NeumannBC(var_name::String, func::Function, sset_name::String)
  return NeumannBC(Symbol(var_name), func, Symbol(sset_name))
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct NeumannBCContainer{
  IT <: Integer,
  IV <: AbstractArray{IT, 1},
  RV <: AbstractArray{<:Union{<:Number, <:SVector}, 2},
  RE <: ReferenceFE
} <: AbstractWeaklyEnforcedBCContainer{IT, IV, RV, RE}
  element_conns::Connectivity{1, IT, IV}
  elements::IV
  sides::IV
  ref_fe::RE
  vals::RV
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

struct NeumannBCs{
  BCCaches <: NamedTuple, 
  BCFuncs  <: NamedTuple
} <: AbstractBCs{BCFuncs}
  bc_caches::BCCaches
  bc_funcs::BCFuncs
end

# note this method has the potential to make 
# bookkeeping.dofs and bookkeeping.nodes nonsensical
# since we're splitting things off but not properly updating
# these to match the current nodes and sides
# TODO modify method to actually properly update
# nodes and dofs

# TODO below method also currently likely doesn't
# handle blocks correclty 
function NeumannBCs(mesh, dof::DofManager, neumann_bcs::Vector{NeumannBC})
  if length(neumann_bcs) == 0
    return NeumannBCs(NamedTuple(), NamedTuple())
  end

  new_bcs, new_funcs = _setup_weakly_enforced_bc_container(mesh, dof, neumann_bcs, NeumannBCContainer)

  syms = tuple(map(x -> Symbol("neumann_bc_$x"), 1:length(new_bcs))...)
  new_bcs = NamedTuple{syms}(tuple(new_bcs...))
  new_funcs = NamedTuple{syms}(tuple(new_funcs...))
  return NeumannBCs(new_bcs, new_funcs)
end

function update_bc_values!(bcs::NeumannBCs, X, t)
  for (bc, func) in zip(bcs.bc_caches, bcs.bc_funcs)
    _update_bc_values!(bc.vals, func, bc.ref_fe, bc.element_conns.data, bc.sides, X, t)
  end
  return nothing
end
