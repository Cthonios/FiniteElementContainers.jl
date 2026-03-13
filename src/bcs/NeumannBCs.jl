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
} <: AbstractBCContainer{IV, RV}
  element_conns::Connectivity{IT, IV}
  elements::IV
  sides::IV
  ref_fe::RE
  vals::RV
end

function Adapt.adapt_structure(to, bc::NeumannBCContainer)
  el_conns = adapt(to, bc.element_conns)
  elements = adapt(to, bc.elements)
  sides = adapt(to, bc.sides)
  ref_fe = adapt(to, bc.ref_fe)
  vals = adapt(to, bc.vals)
  return NeumannBCContainer(el_conns, elements, sides, ref_fe, vals)
end

Base.length(bc::NeumannBCContainer) = size(bc.vals, 2)

function Base.show(io::IO, bc::NeumannBCContainer)
  println(io, "$(typeof(bc).name.name):")
  # println(io, "Blocks                    = $(unique(bk.blocks))")
  println(io, "  Number of active elements = $(length(bc.elements))")
  println(io, "  Number of active nodes    = $(length(bc.side_nodes))")
  println(io, "  Number of active sides    = $(length(bc.sides))")
end

function _update_bc_values!(bc::NeumannBCContainer, func, X, t, ::KA.CPU)
  for n in axes(bc.elements, 1)
    conn = connectivity(bc.ref_fe, bc.element_conns.data, n, 1)
    X_el = _element_level_fields(X, bc.ref_fe, conn)

    for q in 1:num_surface_quadrature_points(bc.ref_fe)
      side = bc.sides[n]
      interps = MappedH1OrL2SurfaceInterpolants(bc.ref_fe, X_el, q, side)
      bc.vals[q, n] = Base.invokelatest(func, interps.X_q, t)
    end
  end
end

KA.@kernel function _update_bc_values_kernel!(
  # bc::NeumannBCContainer, func, X, t
  vals, func, X, t,
  ref_fe, element_conns, sides
)
  # Q, E = KA.@index(Global, NTuple)E
  E = KA.@index(Global)
  conn = connectivity(ref_fe, element_conns, E, 1)
  X_el = _element_level_fields(X, ref_fe, conn)

  for q in 1:num_surface_quadrature_points(ref_fe)
    side = sides[E]
    interps = MappedH1OrL2SurfaceInterpolants(ref_fe, X_el, q, side)
    vals[q, E] = func(interps.X_q, t)
  end
end

function _update_bc_values!(bc::NeumannBCContainer, func, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(
    bc.vals, func, X, t, 
    bc.ref_fe, bc.element_conns.data, bc.sides,
    # ndrange = size(bc.vals)
    ndrange = size(bc.vals, 2)
  )
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

  sets = map(x -> x.sset_name, neumann_bcs)
  vars = map(x -> x.var_name, neumann_bcs)
  funcs = map(x -> x.func, neumann_bcs)

  # NOTE neumann bcs must be present on a sideset
  # so that is the only mesh entity that will be
  # supported for this BC type
  bks = map((v, s) -> BCBookKeeping(mesh, dof, v; sset_name=s), vars, sets)
  fspace = function_space(dof)
  new_bcs = NeumannBCContainer[]
  new_funcs = Function[]

  for (bk, func) in zip(bks, funcs)
    blocks = sort(unique(bk.blocks))

    # TODO fix this
    if length(blocks) > 1
      @error "Neumann BCs present on multiple blocks will likely fail"
    end

    for block in blocks
      block_name = mesh.element_block_names[block]
      ids = findall(x -> x == block, bk.blocks)
      new_blocks = bk.blocks[ids]
      new_elements = bk.elements[ids]
      new_sides = bk.sides[ids]
      new_side_nodes = bk.side_nodes[:, ids]

      # TODO update nodes and dofs
      new_bk = BCBookKeeping(new_blocks, bk.dofs, new_elements, bk.nodes, new_sides, new_side_nodes)
      ref_fe = getproperty(fspace.ref_fes, block_name)
      NQ = num_surface_quadrature_points(ref_fe)
      ND = length(dof.var)

      # TODO below isn't correct
      # we need to map bk.elements using the block element id map
      # NOTE I think that is currently handled in BCBookKeeping
      # by setting bk.elements to the indices where the side set
      # elements live in the block element id map
      conns = mesh.element_conns[block_name][:, bk.elements]

      conns = Connectivity([conns])
      vals = zeros(SVector{ND, Float64}, NQ, length(bk.sides))
      new_bc = NeumannBCContainer(
        conns, new_bk.elements, new_bk.sides, ref_fe, vals
      )
      push!(new_bcs, new_bc)
      push!(new_funcs, func)
    end
  end

  syms = tuple(map(x -> Symbol("neumann_bc_$x"), 1:length(new_bcs))...)
  new_bcs = NamedTuple{syms}(tuple(new_bcs...))
  new_funcs = NamedTuple{syms}(tuple(new_funcs...))
  return NeumannBCs(new_bcs, new_funcs)
end
