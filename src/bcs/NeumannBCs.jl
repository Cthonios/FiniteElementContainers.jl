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
end

# TODO need to hack the var_name thing
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function NeumannBC(var_name::Symbol, sset_name::Symbol, func::Function)
  return NeumannBC{typeof(func)}(func, sset_name, var_name)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function NeumannBC(var_name::String, sset_name::Symbol, func::Function)
  return NeumannBC{typeof(func)}(Symbol(var_name), Symbol(sset_name), func)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct NeumannBCContainer{B, C1, C2, R, T, V} <: AbstractBCContainer{B, T, 2, V}
  bookkeeping::B
  element_conns::C1  
  surface_conns::C2
  ref_fe::R
  vals::V
end

function _update_bc_values!(bc::NeumannBCContainer, func, X, t, ::KA.CPU)
  ND = size(X, 1)
  NN = num_vertices(bc.ref_fe)
  NNPS = num_vertices(bc.ref_fe.surface_element)
  for (n, e) in enumerate(bc.bookkeeping.elements)
    conn = @views bc.element_conns[:, n]
    X_el = SVector{ND * NN, eltype(X)}(@views X[:, conn])
    X_el = SMatrix{length(X_el) ÷ ND, ND, eltype(X_el), length(X_el)}(X_el...)

    for q in 1:num_quadrature_points(bc.ref_fe.surface_element)
      side = bc.bookkeeping.sides[n]
      interps = MappedSurfaceInterpolants(bc.ref_fe, X_el, q, side)
      X_q = interps.X_q
      bc.vals[q, n] = func(X_q, t)
    end
  end
end

KA.@kernel function _update_bc_values_kernel!(bc::NeumannBCContainer, func, X, t)
  ND = size(X, 1)
  NN = num_vertices(bc.ref_fe)
  NNPS = num_vertices(bc.ref_fe.surface_element)

  Q, E = KA.@index(Global, NTuple)
  # E = KA.@index(Global)
  el_id = bc.bookkeeping.elements[E]

  conn = @views bc.element_conns[:, E]
  X_el = SVector{ND * NN, eltype(X)}(@views X[:, conn])
  X_el = SMatrix{length(X_el) ÷ ND, ND, eltype(X_el), length(X_el)}(X_el...)

  # for q in 1:num_quadrature_points(bc.ref_fe.surface_element)
  side = bc.bookkeeping.sides[E]
  interps = MappedSurfaceInterpolants(bc.ref_fe, X_el, Q, side)
  X_q = interps.X_q
  bc.vals[Q, E] = func(X_q, t)
  # end
end

function _update_bc_values!(bc::NeumannBCContainer, func, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(bc, func, X, t, ndrange=size(bc.vals))
end

# note this method has the potential to make 
# bookkeeping.dofs and bookkeeping.nodes nonsensical
# since we're splitting things off but not properly updating
# these to match the current nodes and sides
# TODO modify method to actually properly update
# nodes and dofs

# TODO below method also currently likely doesn't
# handle blocks correclty 
function create_neumann_bcs(dof::DofManager, neumann_bcs::Vector{NeumannBC})
  sets = map(x -> x.sset_name, neumann_bcs)
  vars = map(x -> x.var_name, neumann_bcs)
  funcs = map(x -> x.func, neumann_bcs)
  bks = BCBookKeeping.((dof,), vars, sets)
  # bks = _split_bookkeeping_by_block(bks)
  fspace = function_space(dof, H1Field)
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
      NQ = num_quadrature_points(ref_fe.surface_element)
      ND = length(getfield(dof.H1_vars, var))
      NN = num_vertices(ref_fe)
      NNPS = num_vertices(ref_fe.surface_element)

      # conns = values(fspace.elem_conns)[block][:, new_elements]

      # need to set up "surface connectivity"
      # these are the nodes associated with the sides
      # TODO we need to be careful for higher order elements
      # conns = Vector{eltype(new_elements)}(undef, 0)
      # TODO fix this to get blocks correctly
      conns = values(fspace.elem_conns)[block][:, bk.elements]
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

      conns = Connectivity(conns)
      # surface_conns = Connectivity{NNPS, length(new_bk.elements)}(surface_conns)
      surface_conns = Connectivity{eltype(surface_conns), typeof(surface_conns), NNPS}(surface_conns)

      vals = zeros(SVector{ND, Float64}, NQ, length(bk.sides))
      new_bc = NeumannBCContainer{
        typeof(new_bk), typeof(conns), typeof(surface_conns), typeof(ref_fe), eltype(vals), typeof(vals)
      }(
        new_bk, conns, surface_conns, ref_fe, vals
      )
      push!(new_bcs, new_bc)
      push!(new_funcs, func)
    end
  end

  syms = tuple(map(x -> Symbol("neumann_bc_$x"), 1:length(new_bcs))...)
  new_bcs = NamedTuple{syms}(tuple(new_bcs...))
  new_funcs = NamedTuple{syms}(tuple(new_funcs...))
  return new_bcs, new_funcs
end
