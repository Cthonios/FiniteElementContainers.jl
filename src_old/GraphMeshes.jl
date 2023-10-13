
# function GraphMesh2D(exo::ExodusDatabase{M, I, B, F}) where {M, I, B, F}

# 	# grab the blocks
# 	blocks       = read_sets(exo, Block)

# 	# setup node to edge graph
# 	node_to_edge = SimpleDiGraph(exo.init.num_nodes |> Int64)	

# 	for block in blocks
# 		conn = block.conn
# 		for e in axes(conn, 2)
# 			for n in axes(conn, 1)
# 				if n == size(conn, 1)
# 					add_edge!(node_to_edge, conn[n, e], conn[1, e])
# 				else
# 					add_edge!(node_to_edge, conn[n, e], conn[n + 1, e])
# 				end
# 			end
# 		end
# 	end

# 	# setup edge to cell graph
# 	# edge_to_cell = SimpleGraph(ne(node_to_edge))
	
# 	# for block in blocks
# 	# 	conn = block.conn
# 	# 	for e in axes(conn, 2)
# 	# 		# get edge
# 	# 		for n in axes(conn, 1)

# 	# 		end
# 	# 	end
# 	# end
# 	node_to_edge
# end

abstract type AbstractFace{T} end
abstract type AbstractFaceIter end
abstract type AbstractSimpleFace{T} <: AbstractFace{T} end

mutable struct SimpleFace{T <: Integer} <: AbstractSimpleFace{T}
	edges::Vector{T}
end

mutable struct Simple2DMeshGraph{T <: Integer} <: Graphs.AbstractSimpleGraph{T}
	n_edges::Int
	n_elements::Int
	edge_adj::Vector{Vector{T}}
	face_adj::Vector{Vector{T}}
end

edge_adj(m::Simple2DMeshGraph) = m.edge_adj
face_adj(m::Simple2DMeshGraph) = m.face_adj

Graphs.edges(m::Simple2DMeshGraph) = Graphs.SimpleGraphs.SimpleEdgeIter(m)
Graphs.is_directed(m::Simple2DMeshGraph) = false
Graphs.ne(m::Simple2DMeshGraph) = m.n_edges
Graphs.nv(m::Simple2DMeshGraph{T}) where T = T(length(edge_adj(m)))
Graphs.edgetype(::Simple2DMeshGraph{T}) where {T<:Integer} = SimpleGraphEdge{T}
Graphs.vertices(m::Simple2DMeshGraph) = Base.OneTo(nv(m))


function has_edge(g::Simple2DMeshGraph{T}, s, d) where {T}
	verts = vertices(g)
	(s in verts && d in verts) || return false  # edge out of bounds
	@inbounds list_s = g.edge_adj[s]
	@inbounds list_d = g.edge_adj[d]
	if length(list_s) > length(list_d)
			d = s
			list_s = list_d
	end
	return insorted(d, list_s)
end

function has_edge(g::Simple2DMeshGraph{T}, e::Graphs.SimpleGraphEdge{T}) where {T}
	s, d = T.(Tuple(e))
	return has_edge(g, s, d)
end

function add_edge!(g::Simple2DMeshGraph{T}, e::Graphs.SimpleGraphEdge{T}) where {T}
	s, d = T.(Tuple(e))
	verts = vertices(g)
	(s in verts && d in verts) || return false  # edge out of bounds
	@inbounds list = g.edge_adj[s]
	index = searchsortedfirst(list, d)
	@inbounds (index <= length(list) && list[index] == d) && return false  # edge already in graph
	insert!(list, index, d)

	g.n_edges += 1
	s == d && return true  # selfloop

	@inbounds list = g.edge_adj[d]
	index = searchsortedfirst(list, s)
	insert!(list, index, s)
	return true  # edge successfully added
end
add_edge!(g::Simple2DMeshGraph{T}, a::Integer, b::Integer) where T = add_edge!(g, Graphs.SimpleEdge{T}(a, b))

function Simple2DMeshGraph(n_nodes::Integer, n_elements::Integer, T::Type{<:Integer})
	edge_adj = [Vector{T}() for _ in 1:n_nodes]
	face_adj = [Vector{T}() for _ in 1:n_elements]
	return Simple2DMeshGraph{T}(0, 0, edge_adj, face_adj)
end

function Simple2DMeshGraph(exo::ExodusDatabase{M, I, B, F}) where {M, I, B, F}
	mesh = Simple2DMeshGraph(exo.init.num_nodes, exo.init.num_elems, B)
	blocks = read_sets(exo, Block)
	
	edge_to_edge_id = Dict{Graphs.SimpleGraphEdge{B}, B}()
	edge_cnt::B = 1

	element_edges = Vector{B}[]

	for block in blocks
		conn = block.conn

		for e in axes(conn, 2)
			element_edges_temp = B[]
			for n in axes(conn, 1)

				if n == size(conn, 1)
					src_index = n
					dst_index = 1
				else
					src_index = n
					dst_index = n + 1
				end

				edge = Graphs.SimpleEdge(conn[src_index, e], conn[dst_index, e])
				if !has_edge(mesh, conn[src_index, e], conn[dst_index, e])
					add_edge!(mesh, edge)
					edge_cnt += 1
					edge_to_edge_id[edge] = edge_cnt
					push!(element_edges_temp, edge_cnt)
				else
					if !(edge in keys(edge_to_edge_id))
						other_edge = Graphs.SimpleEdge(conn[dst_index, e], conn[src_index, e])
						push!(element_edges_temp, edge_to_edge_id[other_edge])
					else
						push!(element_edges_temp, edge_to_edge_id[edge])
					end
				end

			end

			push!(element_edges, element_edges_temp)
		end
	end

	display(element_edges)
	return mesh
end

