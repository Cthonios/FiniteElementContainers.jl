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
  IM <: AbstractArray{IT, 2},
  VV <: AbstractArray{<:Union{<:Number, <:SVector}, 2},
  RE <: ReferenceFE
} <: AbstractBCContainer
  element_conns::Connectivity{IT, IV}
  elements::IV
  side_nodes::IM
  sides::IV
  surface_conns::Connectivity{IT, IV}
  ref_fe::RE
  vals::VV
end

function Adapt.adapt_structure(to, bc::NeumannBCContainer)
  el_conns = adapt(to, bc.element_conns)
  elements = adapt(to, bc.elements)
  side_nodes = adapt(to, bc.side_nodes)
  sides = adapt(to, bc.sides)
  surf_conns = adapt(to, bc.surface_conns)
  ref_fe = adapt(to, bc.ref_fe)
  vals = adapt(to, bc.vals)
  return NeumannBCContainer(el_conns, elements, side_nodes, sides, surf_conns, ref_fe, vals)
end

function Base.show(io::IO, bc::NeumannBCContainer)
  println(io, "$(typeof(bc).name.name):")
  # println(io, "Blocks                    = $(unique(bk.blocks))")
  println(io, "  Number of active elements = $(length(bc.elements))")
  println(io, "  Number of active nodes    = $(length(bc.side_nodes))")
  println(io, "  Number of active sides    = $(length(bc.sides))")
end

function _update_bc_values!(bc::NeumannBCContainer, func, X, t, ::KA.CPU)
  ND = size(X, 1)
  NN = num_vertices(bc.ref_fe)
  NNPS = num_vertices(surface_element(bc.ref_fe.element))
  for (n, e) in enumerate(bc.elements)
    # conn = @views bc.element_conns[:, n]
    conn = connectivity(bc.ref_fe, bc.element_conns.data, n, 1)
    X_el = SVector{ND * NN, eltype(X)}(@views X[:, conn])
    X_el = SMatrix{length(X_el) ÷ ND, ND, eltype(X_el), length(X_el)}(X_el...)

    for q in 1:num_quadrature_points(surface_element(bc.ref_fe.element))
      side = bc.sides[n]
      interps = MappedSurfaceInterpolants(bc.ref_fe, X_el, q, side)
      X_q = interps.X_q
      bc.vals[q, n] = func(X_q, t)
    end
  end
end

KA.@kernel function _update_bc_values_kernel!(
  # bc::NeumannBCContainer, func, X, t
  vals, func, X, t,
  ref_fe, elements, element_conns, sides
)
  ND = size(X, 1)
  NN = num_vertices(ref_fe)
  NNPS = num_vertices(surface_element(ref_fe.element))

  Q, E = KA.@index(Global, NTuple)
  # E = KA.@index(Global)
  el_id = elements[E]

  # conn = @views element_conns[:, E]
  conn = connectivity(ref_fe, element_conns, E, 1)
  X_el = SVector{ND * NN, eltype(X)}(@views X[:, conn])
  X_el = SMatrix{length(X_el) ÷ ND, ND, eltype(X_el), length(X_el)}(X_el...)

  # for q in 1:num_quadrature_points(bc.ref_fe.surface_element)
  side = sides[E]
  interps = MappedSurfaceInterpolants(ref_fe, X_el, Q, side)
  X_q = interps.X_q
  vals[Q, E] = func(X_q, t)
  # end
end

function _update_bc_values!(bc::NeumannBCContainer, func, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(
    bc.vals, func, X, t, 
    bc.ref_fe, bc.elements, bc.element_conns.data, bc.sides,
    ndrange = size(bc.vals)
  )
end

struct NeumannBCs{
  BCCaches <: NamedTuple, 
  BCFuncs  <: NamedTuple
}
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

  for (bk, func, var) in zip(bks, funcs, vars)
    blocks = sort(unique(bk.blocks))

    # TODO fix this
    if length(blocks) > 1
      @error "Neumann BCs present on multiple blocks will likely fail"
    end

    for block in blocks
      ids = findall(x -> x == block, bk.blocks)
      new_blocks = bk.blocks[ids]
      new_elements = bk.elements[ids]
      new_sides = bk.sides[ids]
      new_side_nodes = bk.side_nodes[:, ids]
      # TODO update nodes and dofs
      # push!(new_bks, BCBookKeeping(new_blocks, bk.dofs, new_elements, bk.nodes, new_sides))
      new_bk = BCBookKeeping(new_blocks, bk.dofs, new_elements, bk.nodes, new_sides, new_side_nodes)
      ref_fe = values(fspace.ref_fes)[block]
      NQ = num_quadrature_points(surface_element(ref_fe.element))
      # ND = length(getfield(dof.H1_vars, var))
      ND = length(dof.var)
      NN = num_vertices(ref_fe)
      NNPS = num_vertices(surface_element(ref_fe.element))

      # conns = values(fspace.elem_conns)[block][:, new_elements]

      # need to set up "surface connectivity"
      # these are the nodes associated with the sides
      # TODO we need to be careful for higher order elements
      # conns = Vector{eltype(new_elements)}(undef, 0)
      # TODO fix this to get blocks correctly
      # conns = values(fspace.elem_conns)[block][:, bk.elements]
      # conns = connectivity(fspace, block)
      conns = values(mesh.element_conns)[block][:, bk.elements]
      # @show size(conns)
      # display(conns)
      # TODO get field set up properly
      # conns = Connectivity{size(conns, 1), size(conns, 2)}(vec(conns))
      # display(conns)
      surface_conns = Vector{eltype(new_elements)}(undef, 0)
      # conn_syms = map(x -> Symbol("node_$x"), 1:NN)
      # for element in new_elements
      for e in axes(new_elements, 1)
        # for i in 1:bc.num_nodes_per_side[e]
        # append!(conns, values(fspace.elem_conns)[blocks[1]][:, e])
        for i in 1:NNPS
          # TODO is this 2 correct?
          # where tf did it come from
          # k = 2 * (e - 1) + i
          k = NNPS * (e - 1) + i
          # push!(conns, bc.side_nodes[k])
          # push!(conns, bk.nodes[k])
          append!(surface_conns, bk.side_nodes[:, k])
        end
      end

      # conns = Connectivity(conns)
      # # surface_conns = Connectivity{NNPS, length(new_bk.elements)}(surface_conns)
      # surface_conns = Connectivity{eltype(surface_conns), typeof(surface_conns), NNPS}(surface_conns)

      conns = Connectivity([conns])
      surface_conns = reshape(surface_conns, NNPS, length(surface_conns) ÷ NNPS)
      surface_conns = Connectivity([surface_conns])

      vals = zeros(SVector{ND, Float64}, NQ, length(bk.sides))
      # new_bc = NeumannBCContainer{
      #   typeof(new_bk), typeof(conns), typeof(surface_conns), typeof(ref_fe), eltype(vals), typeof(vals)
      # }(
      new_bc = NeumannBCContainer(
        # new_bk, conns, surface_conns, ref_fe, vals
        conns, new_bk.elements, new_bk.side_nodes, new_bk.sides, surface_conns, ref_fe, vals
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

function Adapt.adapt_structure(to, bcs::NeumannBCs)
  return NeumannBCs(
    adapt(to, bcs.bc_caches),
    adapt(to, bcs.bc_funcs)
  )
end

function update_bc_values!(bcs::NeumannBCs, X, t)
  update_bc_values!(bcs.bc_caches, bcs.bc_funcs, X, t)
  return nothing
end
