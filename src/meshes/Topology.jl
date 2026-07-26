# doesn't work on 1.10
# function _canonical_facet(nodes::NTuple{N, I}) where {N, I <: Integer}
#     return sort(nodes)
# end

function _canonical_facet(nodes::NTuple{N,I}) where {N,I<:Integer}
    sorted = sort!(collect(nodes))
    return ntuple(i -> sorted[i], Val(N))
end

function _get_local_facets(el_type, conn, e)
    # eltype = block_element_type(block)

    if el_type in ("HEX","HEX8")
        n = conn[:, e]
        return (
            (n[1], n[2], n[3], n[4]),
            (n[5], n[6], n[7], n[8]),
            (n[1], n[2], n[6], n[5]),
            (n[2], n[3], n[7], n[6]),
            (n[3], n[4], n[8], n[7]),
            (n[4], n[1], n[5], n[8]),
        )
    elseif el_type in ("QUAD", "QUAD4")
        n1, n2, n3, n4 = conn[:, e]
        return (
            (n1, n2),
            (n2, n3),
            (n3, n4),
            (n4, n1),
        )
    elseif el_type in ("TET","TETRA","TET4")
        n1, n2, n3, n4 = conn[:, e]
        return (
            (n1, n2, n3),
            (n1, n2, n4),
            (n2, n3, n4),
            (n1, n3, n4),
        )
    elseif el_type in ("TRI", "TRI3")
        n1, n2, n3 = conn[:, e]
        return (
            (n1, n2),
            (n2, n3),
            (n3, n1),
        )
    else
        error("unsupported 2D element")
    end
end

# minimum working version for likely 2d only
function _is_oriented_positive(nodes)
    return nodes[1] == minimum(nodes)
end

struct MeshToplogy{I <: Integer}
    elem_to_facets::Dict{String, Matrix{I}}
    facets::Vector{Vector{I}}
    facet_map::Dict{Vector{I}, I}
    facet_orientation::Dict{String, Matrix{I}}
    facet_to_elem::Vector{Vector{I}}
end

function MeshToplogy(mesh::AbstractMesh)
    facets = Vector{Vector{Int}}()
    facet_map = Dict{Vector{Int}, Int}()
    facet_to_elem = Vector{Vector{Int}}()
    elem_to_facets = Dict{String, Matrix{Int}}()
    facet_orientation = Dict{Tuple{String, Int, Int}, Int}()

    adj = Dict{Vector{Int}, Vector{Tuple{String, Int, Int}}}()

    # for (block, conn) in block_conns(mesh)
    # for (block, conn) in zip(block_names(mesh), block_conns(mesh))
    for (block_name, conn) in zip(block_names(mesh), block_conns(mesh))
        ne = size(conn, 2)
        el_type = mesh.element_types[block_name]
        nfacets = length(_get_local_facets(el_type, conn, 1))
        elem_to_facets[block_name] = Matrix{Int}(undef, nfacets, ne)

        for e in 1:ne
            local_facets = _get_local_facets(el_type, conn, e)

            for (lf, nodes_tuple) in enumerate(local_facets)

                key = _canonical_facet(nodes_tuple) |> collect
                fid = get!(facet_map, key) do
                    push!(facets, collect(key))
                    push!(facet_to_elem, Int[])
                    length(facets)
                end

                elem_to_facets[block_name][lf, e] = fid

                # orientation (simple but extensible)
                facet_orientation[(block_name, e, lf)] =
                    _is_oriented_positive(nodes_tuple) ? 1 : -1

                push!(get!(adj, collect(key), Tuple{String, Int, Int}[]),
                      (block_name, e, lf))
            end
        end
    end

    for (fkey, uses) in adj
        fid = facet_map[fkey]
        facet_to_elem[fid] = [u[2] for u in uses]
    end

    # better pack orientations
    facet_orientation_new = Dict{String, Matrix{Int}}()
    # for (block, conn) in block_conns(mesh)
    for (block_name, conn) in zip(block_names(mesh), block_conns(mesh))
        temp = Matrix{Int}(undef, size(conn)...)
        for e in axes(conn, 2)
            for n in axes(conn, 1)
                temp[n, e] = facet_orientation[(block_name, e, n)]
            end
        end
        facet_orientation_new[block_name] = temp
    end

    return MeshToplogy(
        elem_to_facets,
        facets,
        facet_map,
        facet_orientation_new,
        facet_to_elem
    )
end
