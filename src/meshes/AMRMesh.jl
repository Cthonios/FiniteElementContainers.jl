struct AMRMesh{
    ND,
    RT <: Number,
    IT <: Integer,
    EConns
} <: AbstractMesh
    nodal_coords::H1Field{RT, Vector{RT}, ND}
    element_block_names::Vector{Symbol}
    element_types::Vector{Symbol}
    element_conns::EConns
    element_id_maps::Dict{Symbol, Vector{IT}}
    node_id_map::Vector{IT}
    nodeset_nodes::Dict{Symbol, Vector{IT}}
    sideset_elems::Dict{Symbol, Vector{IT}}
    sideset_nodes::Dict{Symbol, Vector{IT}}
    sideset_sides::Dict{Symbol, Vector{IT}}
    sideset_side_nodes::Dict{Symbol, Matrix{IT}}
    #
    edges::Vector{NTuple{2, IT}}
    elem2edge::Dict{Symbol, Matrix{IT}}
    edge2adjelem::Dict{Symbol, Vector{NTuple{2, IT}}}
    ref_edges::Dict{Symbol, Vector{Int}}
end

function AMRMesh(mesh::AbstractMesh)
    edges, elem2edge, edge2adjelem = _create_edges(mesh.element_conns)
    ref_edges = Dict{Symbol, Vector{Int}}()
    for (key, conns) in pairs(mesh.element_conns)
        ref_edges[key] = _init_ref_edges(mesh.nodal_coords, conns)
    end

    return AMRMesh(
        mesh.nodal_coords,
        mesh.element_block_names,
        mesh.element_types,
        mesh.element_conns,
        mesh.element_id_maps,
        mesh.node_id_map,
        mesh.nodeset_nodes,
        mesh.sideset_elems,
        mesh.sideset_nodes,
        mesh.sideset_sides,
        mesh.sideset_side_nodes,
        edges, elem2edge, edge2adjelem,
        ref_edges
    )
end

# this uses a longest edge initialization
# slightly smarter than simply initalizing as 1
# but not as great as structured based
function _init_ref_edges(coords, conns)
    ref_edge = Int[]
    for e in axes(conns, 2)
        n1, n2, n3 = conns[:, e]

        l1 = norm(coords[:, n2] - coords[:, n3])  # edge 1
        l2 = norm(coords[:, n3] - coords[:, n1])  # edge 2
        l3 = norm(coords[:, n1] - coords[:, n2])  # edge 3

        if l1 ≥ l2 && l1 ≥ l3
            push!(ref_edge, 1)
        elseif l2 ≥ l3
            push!(ref_edge, 2)
        else
            push!(ref_edge, 3)
        end
    end
    return ref_edge
end

@inline function _local_edge_nodes(conn::NTuple{3, Int}, le::Int)
    if le == 1
        return conn[2], conn[3]
    elseif le == 2
        return conn[3], conn[1]
    elseif le == 3
        return conn[1], conn[2]
    else
        error("Invalid local edge id")
    end
end

function _assign_ref_edge(conn, m)
    for le = 1:3
        a, b = _local_edge_nodes(conn, le)
        if a != m && b != m
            return le
        end
    end
    error("No valid reference edge found")
end

# this method invalidates elem2edge and edge2adjelem
function _refine_element!(
    coords, conns, ref_edge, elem2edge, edge2adjelem, nodeset_nodes,
    edge_midpoint, e
)
    conn = (conns[1, e], conns[2, e], conns[3, e])
    le   = ref_edge[e]
    i, j = _local_edge_nodes(conn, le)
    k    = only(setdiff(conn, (i, j)))
    gid = elem2edge[le, e]

    if haskey(edge_midpoint, gid)
        m = edge_midpoint[gid]
    else
        m = size(coords, 2) + 1
        new_coord = 0.5 .* (coords[:, i] + coords[:, j])
        append!(coords.data, new_coord)
        edge_midpoint[gid] = m

        el, er = edge2adjelem[gid]
        if er == -1
            _update_nodesets_on_edge!(nodeset_nodes, i, j, m)
        end
    end

    # children
    conn1 = (m, j, k)
    conn2 = (i, m, k)

    ref1 = _assign_ref_edge(conn1, m)
    ref2 = _assign_ref_edge(conn2, m)

    conns[:, e] .= conn1
    ref_edge[e]  = ref1

    append!(conns.data, collect(conn2))
    push!(ref_edge, ref2)
end

function _closure(initial_marked, ref_edge, elem2edge, edge2adjelem)
    to_refine = Set(initial_marked)
    changed = true

    while changed
        changed = false

        for e in collect(to_refine)
            # global edge being bisected by element e
            le  = ref_edge[e]
            gid = elem2edge[le, e]

            el, er = edge2adjelem[gid]

            for nbr in (el, er)
                if nbr != -1 && !(nbr in to_refine)
                    push!(to_refine, nbr)
                    changed = true
                end
            end
        end
    end

    # Important: reverse order so children don't invalidate parents
    return sort!(collect(to_refine), rev = true)
end

# TODO fix to work on multi-block
function _refine!(amr::AMRMesh, marked)
    for key in keys(amr.elem2edge)
        closure = _closure(
            marked, amr.ref_edges[key], 
            amr.elem2edge[key], amr.edge2adjelem[key]
        )
        edge_midpoint = Dict{Int, Int}()
        for e in closure
            _refine_element!(
                amr.nodal_coords, 
                amr.element_conns[key], amr.ref_edges[key], 
                amr.elem2edge[key], amr.edge2adjelem[key],
                amr.nodeset_nodes,
                edge_midpoint, e
            )
        end
    end

    resize!(amr.node_id_map, size(amr.nodal_coords, 2))
    amr.node_id_map .= 1:length(amr.node_id_map)

    # now we need to reset the edge connectivity and assert conformity

end

# currently a dumb implementation
# make this more efficient
function _update_nodesets_on_edge!(
    nodeset_nodes::Dict{Symbol, Vector{Int}},
    i::Int, j::Int, m::Int
)
    for (_, nodes) in pairs(nodeset_nodes)
        # Check if this nodeset contains the entire edge
        if (i in nodes) && (j in nodes)
            # Insert midpoint if not already present
            if !(m in nodes)
                push!(nodes, m)
            end
        end
    end
end
