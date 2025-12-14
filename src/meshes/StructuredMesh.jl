struct StructuredMesh{
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
end

function StructuredMesh(element_type, mins, maxs, counts)
    @assert length(mins) == length(maxs)
    @assert length(maxs) == length(counts)
    for (n, (m1, m2)) in enumerate(zip(mins, maxs))
        if m1 >= m2
            @error "Dimension $n has negative or zero length"
            throw(BoundsError())
        end
    end
    element_type = uppercase(element_type)
    if occursin(element_type, "HEX")
        element_type = :HEX8
        num_dims = 3
        nodal_coords, element_conns, nodeset_nodes,
        sideset_elems, sideset_nodes, sideset_sides, sideset_side_nodes = 
            _hex8_structured_mesh_data(mins, maxs, counts)
    elseif occursin(element_type, "QUAD")
        element_type = :QUAD4
        num_dims = 2
        nodal_coords, element_conns, nodeset_nodes,
        sideset_elems, sideset_nodes, sideset_sides, sideset_side_nodes = 
            _quad4_structured_mesh_data(mins, maxs, counts)
    elseif occursin(element_type, "TET")
        @assert false "Implement tet case"
    elseif occursin(element_type, "TRI")
        element_type = :TRI3
        num_dims = 2
        nodal_coords, element_conns, nodeset_nodes,
        sideset_elems, sideset_nodes, sideset_sides, sideset_side_nodes = 
            _tri3_structured_mesh_data(mins, maxs, counts)
    else
        throw(ArgumentError("Unsupported element type $element_type"))
    end
    # coords, conns
    nodal_coords = H1Field(nodal_coords)
    element_block_names = Symbol[:block_1]
    element_types = Symbol[element_type]
    element_id_maps = Dict(:block_1 => 1:size(element_conns, 2) |> collect)
    element_conns = Dict(:block_1 => L2ElementField(element_conns))
    node_id_map = 1:size(nodal_coords, 2) |> collect

    element_conns = NamedTuple(element_conns)

    return StructuredMesh(
        nodal_coords,
        element_block_names, element_types,
        element_conns, element_id_maps,
        node_id_map,
        nodeset_nodes,
        sideset_elems, sideset_nodes,
        sideset_sides, sideset_side_nodes
    )
end

function Base.show(io::IO, mesh::StructuredMesh)
    println(io, "StructuredMesh:")
    println(io, "  Number of dimensions = $(size(mesh.nodal_coords, 1))")
    println(io, "  Number of nodes      = $(size(mesh.nodal_coords, 2))")
    println(io, "  Number of elements   = $(size(mesh.element_conns[:block_1], 2))")
    println(io, "  Element type         = $(mesh.element_types[1])")

end

function _hex8_structured_mesh_data(mins, maxs, counts)
    N_x, N_y, N_z = counts
    E_x = N_x - 1
    E_y = N_y - 1
    E_z = N_z - 1
  
    coords = Matrix{Float64}(undef, 3, N_x * N_y * N_z)
    _three_dimensional_coords!(coords, mins, maxs, counts)
  
    node(i, j, k) = i + N_x * (j - 1) + N_x * N_y * (k - 1)

    conns = Matrix{Int64}(undef, 8, E_x * E_y * E_z)
    n = 1
    # for ez in 1:E_z
    #     for ey in 1:E_y
    #         for ex in 1:E_x
    #             conns[1, n] = node(ex, ey, ez)
    #             conns[2, n] = node(ex+1, ey, ez)
    #             conns[3, n] = node(ex+1, ey+1, ez)
    #             conns[4, n] = node(ex, ey+1, ez)
    #             conns[5, n] = node(ex, ey, ez+1)
    #             conns[6, n] = node(ex+1, ey, ez+1)
    #             conns[7, n] = node(ex+1, ey+1, ez+1)
    #             conns[8, n] = node(ex, ey+1, ez+1)
    #             n += 1
    #         end
    #     end
    # end
    for ex in 1:E_x
        for ey in 1:E_y
            for ez in 1:E_z
                conns[1, n] = node(ex, ey, ez)
                conns[2, n] = node(ex + 1, ey, ez)
                conns[3, n] = node(ex + 1, ey + 1, ez)
                conns[4, n] = node(ex, ey + 1, ez)
                conns[5, n] = node(ex, ey, ez + 1)
                conns[6, n] = node(ex + 1, ey, ez + 1)
                conns[7, n] = node(ex + 1, ey + 1, ez + 1)
                conns[8, n] = node(ex, ey + 1, ez + 1)
                n = n + 1
            end
        end
    end
    nodeset_nodes = _three_dimensional_nsets(N_x, N_y, N_z)

    return coords, conns, nodeset_nodes, _hex8_ssets(conns, N_x, N_y, N_z)...
end

function _hex8_ssets(conns, N_x, N_y, N_z)
    E_x = N_x - 1
    E_y = N_y - 1
    E_z = N_z - 1
    elem(i, j, k) = (i - 1) * E_y * E_z + (j - 1) * E_z + k

    sideset_elems = Dict{Symbol, Vector{Int}}(
        :bottom => Int[],
        :right => Int[],
        :front => Int[],
        :top => Int[],
        :left => Int[],
        :back => Int[]
    )
    sideset_side_nodes = Dict{Symbol, Matrix{Int}}(
        :bottom => Matrix{Int}(undef, 4, E_x),
        :right => Matrix{Int}(undef, 4, E_y),
        :front => Matrix{Int}(undef, 4, E_z),
        :top => Matrix{Int}(undef, 4, E_x),
        :left => Matrix{Int}(undef, 4, E_y),
        :back => Matrix{Int}(undef, 4, E_z)
    )
    sideset_sides = Dict{Symbol, Vector{Int}}(
        :bottom => Int[],
        :right => Int[],
        :front => Int[],
        :top => Int[],
        :left => Int[],
        :back => Int[]
    )

    # bottom
    j = 1
    for k in 1:E_z, i in 1:E_x
        e = elem(i, j, k)
        n1, n2, n3, n4 = conns[1,e], conns[2,e], conns[3,e], conns[4,e]
        push!(sideset_elems[:bottom], e)
        push!(sideset_sides[:bottom], 5)
        sideset_side_nodes[:bottom][:, i] .= [n1, n2, n3, n4]
    end

    # right
    i = E_x
    for k in 1:E_z, j in 1:E_y
        e = elem(i, j, k)
        n2, n3, n7, n6 = conns[2, e], conns[3, e], conns[7, e], conns[6, e]
        push!(sideset_elems[:right], e)
        push!(sideset_sides[:right], 2) 
        sideset_side_nodes[:right][:, j] .= [n2, n3, n7, n6]   
    end

    # front
    k = E_z
    for j in 1:E_y, i in 1:E_x
        e = elem(i, j, k)
        n1, n2, n6, n5 = conns[1, e], conns[2, e], conns[6, e], conns[5, e]
        push!(sideset_elems[:front], e)
        push!(sideset_sides[:front], 1) 
        sideset_side_nodes[:front][:, j] .= [n1, n2, n6, n5]   
    end

    # top
    j = E_y
    for k in 1:E_z, i in 1:E_x
        e = elem(i, j, k)
        n5, n6, n7, n8 = conns[5, e], conns[6, e], conns[7, e], conns[8, e]
        push!(sideset_elems[:top], e)
        push!(sideset_sides[:top], 6) 
        sideset_side_nodes[:top][:, i] .= [n5, n6, n7, n8]
    end

    # left
    i = 1
    for j in 1:E_y
        e = elem(i, j, k)
        n4, n1, n5, n8 = conns[4, e], conns[1, e], conns[5, e], conns[8, e]
        push!(sideset_elems[:left], e)
        push!(sideset_sides[:left], 4)
        sideset_side_nodes[:left][:, j] .= [n4, n1, n5, n8]
    end

    # back
    k = 1
    for j in 1:E_y, i in 1:E_x
        e = elem(i, j, k)
        n3, n4, n8, n7 = conns[3, e], conns[4, e], conns[8, e], conns[7, e]
        push!(sideset_elems[:back], e)
        push!(sideset_sides[:back], 3) 
        sideset_side_nodes[:back][:, j] .= [n3, n4, n8, n7]   
    end

    sideset_nodes = Dict{Symbol, Vector{Int}}()
    for (k, v) in sideset_side_nodes
        sideset_nodes[k] = unique(v)
    end

    return sideset_elems, sideset_nodes, sideset_sides, sideset_side_nodes
end

function _quad4_structured_mesh_data(mins, maxs, counts)
    N_x, N_y = counts
    E_x = N_x - 1
    E_y = N_y - 1
  
    coords = Matrix{Float64}(undef, 2, N_x * N_y)
    _two_dimensional_coords!(coords, mins, maxs, counts)
  
    node(i, j) = i + (j - 1) * N_x

    conns = Matrix{Int64}(undef, 4, E_x * E_y)
    n = 1
    for ex in 1:E_x
        for ey in 1:E_y
            conns[1, n] = node(ex, ey)
            conns[2, n] = node(ex + 1, ey)
            conns[3, n] = node(ex + 1, ey + 1)
            conns[4, n] = node(ex, ey + 1)
            n = n + 1
        end
    end
    nodeset_nodes = _two_dimensional_nsets(N_x, N_y)
    return coords, conns, nodeset_nodes, _quad4_ssets(conns, N_x, N_y)...
end

function _quad4_ssets(conns, N_x, N_y)
    E_x = N_x - 1
    E_y = N_y - 1
    elem(i, j) = (i - 1) * E_y + j

    sideset_elems = Dict{Symbol, Vector{Int}}(
        :bottom => Int[],
        :right => Int[],
        :top => Int[],
        :left => Int[]
    )
    sideset_side_nodes = Dict{Symbol, Matrix{Int}}(
        :bottom => Matrix{Int}(undef, 2, E_x),
        :right => Matrix{Int}(undef, 2, E_y),
        :top => Matrix{Int}(undef, 2, E_x),
        :left => Matrix{Int}(undef, 2, E_y)
    )
    sideset_sides = Dict{Symbol, Vector{Int}}(
        :bottom => Int[],
        :right => Int[],
        :top => Int[],
        :left => Int[]
    )

    # bottom
    j = 1
    for i in 1:E_x
        e = elem(i, j)
        n1, n2 = conns[1, e], conns[2, e]
        push!(sideset_elems[:bottom], e)
        push!(sideset_sides[:bottom], 1)
        sideset_side_nodes[:bottom][:, i] .= [n1, n2]
    end

    # right
    i = E_x
    for j in 1:E_y
        e = elem(i, j)
        n2, n3 = conns[2, e], conns[3, e]
        push!(sideset_elems[:right], e)
        push!(sideset_sides[:right], 2) 
        sideset_side_nodes[:right][:, j] .= [n2, n3]   
    end

    # top
    j = E_y
    for i in 1:E_x
        e = elem(i, j)
        n3, n4 = conns[3, e], conns[4, e]
        push!(sideset_elems[:top], e)
        push!(sideset_sides[:top], 3) 
        sideset_side_nodes[:top][:, i] .= [n3, n4]
    end

    # left
    i = 1
    for j in 1:E_y
        e = elem(i, j)
        n4, n1 = conns[4, e], conns[1, e]
        push!(sideset_elems[:left], e)
        push!(sideset_sides[:left], 4)
        sideset_side_nodes[:left][:, j] .= [n4, n1]
    end

    sideset_nodes = Dict{Symbol, Vector{Int}}()
    for (k, v) in sideset_side_nodes
        sideset_nodes[k] = unique(v)
    end

    return sideset_elems, sideset_nodes, sideset_sides, sideset_side_nodes
end

function _tri3_structured_mesh_data(mins, maxs, counts)
    N_x, N_y = counts
    E_x = N_x - 1
    E_y = N_y - 1
  
    coords = Matrix{Float64}(undef, 2, N_x * N_y)
    _two_dimensional_coords!(coords, mins, maxs, counts)
  
    conns = Matrix{Int64}(undef, 3, 2 * E_x * E_y)
    n = 1
    for ex in 1:E_x
        for ey in 1:E_y
            conns[1, n] = (ex - 1) + N_x * (ey - 1) + 1
            conns[2, n] = ex + N_x * (ey - 1) + 1
            conns[3, n] = ex + N_x * ey + 1
            conns[1, n + 1] = (ex - 1) + N_x * (ey - 1) + 1
            conns[2, n + 1] = ex + N_x * ey + 1
            conns[3, n + 1] = (ex - 1) + N_x * ey + 1
            n = n + 2
        end
    end
    nodeset_nodes = _two_dimensional_nsets(N_x, N_y)
    return coords, conns, nodeset_nodes, _tri3_ssets(conns, N_x, N_y)...
end

# TODO finish me
function _tri3_ssets(conns, N_x, N_y)
    E_x = N_x - 1
    E_y = N_y - 1
    quad(i, j) = (i - 1) * E_y + j
    tri_a(q) = 2q - 1
    tri_b(q) = 2q

    sideset_elems = Dict{Symbol, Vector{Int}}(
        :bottom => Int[],
        :right => Int[],
        :top => Int[],
        :left => Int[]
    )
    sideset_side_nodes = Dict{Symbol, Matrix{Int}}(
        :bottom => Matrix{Int}(undef, 2, E_x),
        :right => Matrix{Int}(undef, 2, E_y),
        :top => Matrix{Int}(undef, 2, E_x),
        :left => Matrix{Int}(undef, 2, E_y)
    )
    sideset_sides = Dict{Symbol, Vector{Int}}(
        :bottom => Int[],
        :right => Int[],
        :top => Int[],
        :left => Int[]
    )

        # bottom
        j = 1
        for i in 1:E_x
            q = quad(i, j)
            e = tri_a(q)
            n1, n2 = conns[1, e], conns[2, e]
            push!(sideset_elems[:bottom], e)
            push!(sideset_sides[:bottom], 1)
            sideset_side_nodes[:bottom][:, i] .= [n1, n2]
        end
    
        # right
        i = E_x
        for j in 1:E_y
            q = quad(i, j)
            e = tri_a(q)
            n2, n3 = conns[2, e], conns[3, e]
            push!(sideset_elems[:right], e)
            push!(sideset_sides[:right], 2) 
            sideset_side_nodes[:right][:, j] .= [n2, n3]   
        end
    
        # top
        j = E_y
        for i in 1:E_x
            q = quad(i, j)
            e = tri_b(q)
            n3, n4 = conns[2, e], conns[3, e]
            push!(sideset_elems[:top], e)
            push!(sideset_sides[:top], 2) 
            sideset_side_nodes[:top][:, i] .= [n3, n4]
        end
    
        # left
        i = 1
        for j in 1:E_y
            q = quad(i, j)
            e = tri_b(q)
            n4, n1 = conns[3, e], conns[1, e]
            push!(sideset_elems[:left], e)
            push!(sideset_sides[:left], 3)
            sideset_side_nodes[:left][:, j] .= [n4, n1]
        end

    sideset_nodes = Dict{Symbol, Vector{Int}}()
    for (k, v) in sideset_side_nodes
        sideset_nodes[k] = unique(v)
    end 

    return sideset_elems, sideset_nodes, sideset_sides, sideset_side_nodes
end

function _three_dimensional_coords!(coords, mins, maxs, counts)
    xs = LinRange(mins[1], maxs[1], counts[1])
    ys = LinRange(mins[2], maxs[2], counts[2])
    zs = LinRange(mins[3], maxs[3], counts[3])
    n = 1
    for nz in 1:counts[1]
        for ny in 1:counts[2]
            for nx in 1:counts[3]
                coords[1, n] = xs[nx]
                coords[2, n] = ys[ny]
                coords[3, n] = zs[nz]
                n = n + 1
            end
        end
    end
    return nothing
end

# TODO
function _three_dimensional_nsets(N_x, N_y, N_z)
    node(i, j, k) = i + (j - 1) * N_x + (k - 1) * N_x * N_y

    bottom = [node(i, 1, k) for i in 1:N_x, k in 1:N_z] |> vec
    right  = [node(N_x, j, k) for j in 1:N_y, k in 1:N_z] |> vec
    front  = [node(i, j, N_z) for i in 1:N_x, j in 1:N_y] |> vec
    top    = [node(i, N_y, k) for i in 1:N_x, k in 1:N_z] |> vec
    left   = [node(1, j, k) for j in 1:N_y, k in 1:N_z] |> vec
    back   = [node(i, j, 1) for i in 1:N_x, j in 1:N_y] |> vec

    return Dict{Symbol, Vector{Int}}(
        :bottom => bottom,
        :right => right,
        :front => front,
        :top => top,
        :left => left,
        :back => back
    )
end

function _two_dimensional_coords!(coords, mins, maxs, counts)
    xs = LinRange(mins[1], maxs[1], counts[1])
    ys = LinRange(mins[2], maxs[2], counts[2])
    n = 1
    for ny in 1:counts[1]
        for nx in 1:counts[2]
            coords[1, n] = xs[nx]
            coords[2, n] = ys[ny]
            n = n + 1
        end
    end
    return nothing
end

function _two_dimensional_nsets(N_x, N_y)
    node(i, j) = i + (j - 1) * N_x

    bottom = [node(i, 1) for i in 1:N_x]
    right  = [node(N_x, j) for j in 1:N_y]
    top    = [node(i, N_y) for i in 1:N_x]
    left   = [node(1, j) for j in 1:N_y]

    return Dict{Symbol, Vector{Int}}(
        :bottom => bottom,
        :right => right,
        :top => top,
        :left => left
    )
end
