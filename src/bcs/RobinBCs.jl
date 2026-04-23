"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```RobinBC````.
"""
struct RobinBC{F} <: AbstractBC{F}
  func::F
  sset_name::String
  var_name::String

  """
  $(TYPEDEF)
  $(TYPEDSIGNATURES)
  $(TYPEDFIELDS)
  """
  function RobinBC(var_name::String, func, sset_name::String)
    new{typeof(func)}(func, sset_name, var_name)
  end
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct RobinBCContainer{
  IT  <: Integer,
  IV  <: AbstractArray{IT, 1},
  RV  <: AbstractArray{<:Union{<:Number, <:SVector}, 2},
  dRV <: AbstractArray{<:Union{<:Number, <:SMatrix}, 2}
} <: AbstractWeaklyEnforcedBCContainer{IT, IV, RV}
  element_conns::Connectivity{IT, IV}
  elements::IV
  sides::IV
  vals::RV
  dvalsdu::dRV
end

function Adapt.adapt_structure(to, bc::RobinBCContainer)
  el_conns = adapt(to, bc.element_conns)
  elements = adapt(to, bc.elements)
  sides = adapt(to, bc.sides)
  vals = adapt(to, bc.vals)
  dvalsdu = adapt(to, bc.dvalsdu)
  return RobinBCContainer(el_conns, elements, sides, vals, dvalsdu)
end

struct RobinBCFunction{F1, F2} <: AbstractBCFunction{F1}
  func::F1
  dfuncdu::F2
end

function RobinBCFunction(func)
  dfuncdu = (x, t, u) -> ForwardDiff.jacobian(z -> func(x, t, z), u)
  return RobinBCFunction(func, dfuncdu)
end

function _update_bc_values!(vals, dvalsdu, func, ref_fe, conns, sides, X, t, U)
  fec_axes(vals, 2) do e
    conn = connectivity(ref_fe, conns, e, 1)
    X_el = _element_level_fields(X, ref_fe, conn)
    u_el = _element_level_fields(U, ref_fe, conn)
  
    for q in 1:num_surface_quadrature_points(ref_fe)
      side = sides[e]
      interps = MappedH1OrL2SurfaceInterpolants(ref_fe, X_el, q, side)
      u_q = u_el * interps.N
      vals[q, e] = func.func(interps.X_q, t, u_q)
      dvalsdu[q, e] = func.dfuncdu(interps.X_q, t, u_q)
    end
  end
end

struct RobinBCs{
    BCCaches <: NamedTuple, 
    BCFuncs  <: NamedTuple
  } <: AbstractBCs{BCFuncs}
    bc_caches::BCCaches
    bc_funcs::BCFuncs
    block_ids::Vector{Int}
    block_names::Vector{String}
end

function RobinBCs(mesh, dof::DofManager, robin_bcs::Vector{RobinBC})
    if length(robin_bcs) == 0
        return RobinBCs(NamedTuple(), NamedTuple(), Int[], String[])
    end

    new_bcs, new_funcs, block_ids, block_names = _setup_weakly_enforced_bc_container(mesh, dof, robin_bcs, RobinBCContainer)
    new_funcs = RobinBCFunction.(new_funcs)
    syms = tuple(map(x -> Symbol("robin_bc_$x"), 1:length(new_bcs))...)
    new_bcs = NamedTuple{syms}(tuple(new_bcs...))
    new_funcs = NamedTuple{syms}(tuple(new_funcs...))
    return RobinBCs(new_bcs, new_funcs, block_ids, block_names)
end

function Adapt.adapt_structure(to, bcs::RobinBCs)
  return RobinBCs(
    map(x -> adapt(to, x), bcs.bc_caches),
    bcs.bc_funcs,
    bcs.block_ids,
    bcs.block_names
  )
end

function update_bc_values!(bcs::RobinBCs, assembler, X, t, U)
  # for (bc, func) in zip(bcs.bc_caches, bcs.bc_funcs)
  #   _update_bc_values!(bc.vals, bc.dvalsdu, func, bc.ref_fe, bc.element_conns.data, bc.sides, X, t, U)
  # end
  fspace = function_space(assembler.dof)
  for n in axes(bcs.block_ids, 1)
    bc = values(bcs.bc_caches)[n]
    block_id = bcs.block_ids[n]
    func = values(bcs.bc_funcs)[n]
    ref_fe = block_reference_element(fspace, block_id)
    _update_bc_values!(bc.vals, bc.dvalsdu, func, ref_fe, bc.element_conns.data, bc.sides, X, t, U)
  end
  return nothing
end
