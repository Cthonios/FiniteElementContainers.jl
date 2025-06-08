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

# """
# $(TYPEDEF)
# $(TYPEDSIGNATURES)
# $(TYPEDFIELDS)
# Method assumes there's only one block present in nbc
# """
# function NeumannBCContainer(dof::DofManager, nbc::NeumannBC)
#   fspace = function_space(dof, H1Field)
#   ref_fe = values(fspace.ref_fes)[bk.blocks[1]]
#   ND = length(getfield(dof.H1_vars, nbc.var_name))
#   NN = num_vertices(ref_fe.surface_element)

#   bk = BCBookKeeping(dof, nbc.var_name, nbc.sset_name)

#   # sort blocks, elements, sides, and side_nodes
#   el_perm = _unique_sort_perm(bk.elements)
#   blocks = bk.blocks[el_perm]
#   elements = bk.elements[el_perm]
#   sides = bk.elements[el_perm]
#   side_nodes = reshape(bk.side_nodes, NN, length(sides))[:, el_perm]

#   resize!(bk.blocks, length(blocks))
#   resize!(bk.elements, length(elements))
#   resize!(bk.sides, length(sides))
#   # TODO how to resize a matrix

#   copyto!(bk.blocks, blocks)
#   copyto!(bk.elements, elements)
#   copyto!(bk.sides, sides)
#   copyto!(bk.side_nodes, side_nodes)

#   vals = zeros(SVector{ND, Float64}, length(bk.sides))
#   return NeumannBCContainer{typeof(bk), eltype(vals), typeof(vals)}(
#     bk, vals
#   )
# end

function _update_bc_values!(bc::NeumannBCContainer, func, X, t, ::KA.CPU)
  ND = size(X, 1)
  NN = num_vertices(bc.ref_fe)
  NNPS = num_vertices(bc.ref_fe.surface_element)
  for (n, e) in enumerate(bc.bookkeeping.elements)
    # block_id = bc.bookkeeping.blocks[n]
    # ref_fe = values(fspace.ref_fes)[block_id]
    # X_el = surface_element_coordinates(sec, X, e)
    conn = @views bc.element_conns[:, n]
    # display(conn)
    X_el = SVector{ND * NN, eltype(X)}(@views X[:, conn])
    # X_el = SMatrix{ND, length(X_el) ÷ ND, eltype(X_el), length(X_el)}(X_el...)
    X_el = SMatrix{length(X_el) ÷ ND, ND, eltype(X_el), length(X_el)}(X_el...)

    for q in 1:num_quadrature_points(bc.ref_fe)
      side = bc.bookkeeping.sides[n]
      interps = MappedSurfaceInterpolants(bc.ref_fe, X_el, q, side)
      X_q = interps.X_q
      bc.vals[q, n] = func(X_q, t)
    end
  end
end

# note this method has the potential to make 
# bookkeeping.dofs and bookkeeping.nodes nonsensical
# since we're splitting things off but not properly updating
# these to match the current nodes and sides
# TODO modify method to actually properly update
# nodes and dofs
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
    blocks = sort!(unique(bk.blocks))

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
      NQ = num_quadrature_points(ref_fe)
      ND = length(getfield(dof.H1_vars, var))
      NN = num_vertices(ref_fe)
      NNPS = num_vertices(ref_fe.surface_element)

      # conns = values(fspace.elem_conns)[block][:, new_elements]

      # need to set up "surface connectivity"
      # these are the nodes associated with the sides
      # TODO we need to be careful for higher order elements
      conns = Vector{eltype(new_elements)}(undef, 0)
      surface_conns = Vector{eltype(new_elements)}(undef, 0)
      # conn_syms = map(x -> Symbol("node_$x"), 1:NN)
      # for element in new_elements
      for e in axes(new_elements, 1)
        # for i in 1:bc.num_nodes_per_side[e]
        append!(conns, values(fspace.elem_conns)[blocks[1]][:, e])
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

      conns = Connectivity{NN, length(new_bk.elements)}(conns)
      surface_conns = Connectivity{NNPS, length(new_bk.elements)}(surface_conns)

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

  return new_bcs, new_funcs
end
