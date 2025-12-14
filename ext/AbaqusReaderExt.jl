module AbaqusReaderExt

using AbaqusReader
using FiniteElementContainers

function _check_for_command(line)
    if line[1] == '*' && line[2] != '*'
        return true
    else
        return false
    end
end

function _check_for_comment(line)
    try
        if line[1:2] == "**"
            return true
        end
    catch BoundsError
        # do nothing
    end
    return false
end

function _get_command_data(lines, command_line_ids, comment_line_ids, command_id)
    if _check_for_comment(lines[command_id])
        return nothing
    end
    # substr = split(lines[command_id], '*')[2]
    substr = lowercase(lines[command_id])

    if contains(substr, "*element") && 
       !contains(substr, "*element output") &&
       !contains(substr, "*user element")
        element_conns = Dict{Int, Vector{Int}}()

        if contains(substr, "elset")
            elset_name = String(split(split(substr, "elset")[2], '=')[2])
        else
            elset_name = nothing
        end

        if contains(substr, "type")
            el_type = String(split(split(substr, "type")[2], '=')[2])
        else
            @assert false
        end

        n = command_id + 1
        while n ∉ command_line_ids
            if n in comment_line_ids
                n = n + 1
                continue
            end
            parts = split(lines[n], ',')
            el_id = parse(Int, parts[1])
            conn = parse.((Int,), parts[2:end])
            element_conns[el_id] = conn
            n = n + 1
        end
        return "element", Dict(elset_name => element_conns), el_type
    elseif contains(substr, "*heading")
        heading = String[]
        n = command_id + 1
        while n ∉ command_line_ids
            if n in comment_line_ids
                n = n + 1
                continue
            end
            push!(heading, lines[n])
            n = n + 1
        end
        return "heading", heading
    elseif contains(substr, "*material")
        mat_name = String(split(split(substr, ',')[2], '=')[2])
        props = Dict{String, Float64}()
        n = command_id + 1
        if contains(lowercase(lines[n]), "*elastic")
            n = n + 1
            props["elastic"] = parse(Float64, lines[n])
        end
        return "material", Dict(mat_name => props)
    elseif contains(substr, "*node") && !contains(substr, "*node output")
        nodes = Dict{Int, Vector{Float64}}()
        if contains(substr, "nset")
            nset_name = split(split(substr, "nset")[2], '=')[2]
        else
            nset_name = nothing
        end

        n = command_id + 1
        while n ∉ command_line_ids
            if n in comment_line_ids
                n = n + 1
                continue
            end

            parts = split(lines[n], ',')
            node_id = parse(Int, parts[1])
            coords = parse.((Float64,), parts[2:end])
            nodes[node_id] = coords
            n = n + 1
        end
        return "nodes", Dict(nset_name => nodes)
    elseif contains(substr, "*nset")
        nodes = Dict{String, Vector{Int}}()
        n = command_id + 1
        if contains(substr, "generate")
            nset_name = split(split(split(split(substr, "*nset")[2], "generate")[1], ',')[2], '=')[2]
            parts = parse.((Int,), split(lines[n], ','))
            nodes[nset_name] = range(start=parts[1], step=parts[3], stop=parts[2])
        else
            nset_name = split(split(split(substr, "*nset")[2], ',')[2], '=')[2]
            temp_nodes = Vector{Int}(undef, 0)
            while n ∉ command_line_ids
                if n in comment_line_ids
                    n = n + 1
                    continue
                end
                @show lines[n]
                for node in split(lines[n], ',')
                    try
                        push!(temp_nodes, parse(Int, node))
                    catch ArgumentError
                        # do nothing
                        if node != ""
                            @warn "Got a potentially bad line reading a nodeset"
                            @warn "$(lines[n])"
                        end
                    end
                end
                n = n + 1
            end
            nodes[nset_name] = temp_nodes
        end
        return "nset", nodes
    elseif contains(substr, "*parameter")
        parameters = String[]
        n = command_id + 1
        while n ∉ command_line_ids
            if n in comment_line_ids
                n = n + 1
                continue
            end
            push!(parameters, lines[n])
            n = n + 1
        end
        return "parameter", parameters
    else
        @warn "Implement $substr"
        substr, nothing
    end
end

function FiniteElementContainers.UnstructuredMesh(
    ::FiniteElementContainers.AbaqusMesh, 
    file_name::String,
    create_edges::Bool, create_faces::Bool
)
    command_line_ids = Int[]
    comment_line_ids = Int[]
    comments = String[]

    sections = Dict{String, Any}()
    open(file_name, "r") do f
        lines = readlines(f)

        # do a first pass to find where commands stop
        for (n, line) in enumerate(lines)
            if _check_for_command(line)
                push!(command_line_ids, n)
            end

            # skip comments
            if _check_for_comment(line)
                push!(comment_line_ids, n)
            end
        end

        for command_id in command_line_ids
            key, data = _get_command_data(lines, command_line_ids, comment_line_ids, command_id)

            if key === nothing || data === nothing
                @warn "Got nothing key/data for line $(lines[command_id])"
                continue
            end

            if haskey(sections, key)
                sections[key] = merge(sections[key], data)
            else
                sections[key] = data
            end
        end

        for comment_id in comment_line_ids
            push!(comments, lines[comment_id])
        end
    end
    
    sections
    # comments
    # WIP below
    # AbaqusReader.register_element!("CPE4T", 4, "Quad4")
    # AbaqusReader.register_element!("C3D8RT", 8, "Hex8")
    # AbaqusReader.register_element!("C3D8T", 8, "Hex8")

    # # adding a few user element type names
    # # we've encountered in the wild
    # # NOTE this is probably dangerous in general
    # AbaqusReader.register_element!("U1", 8, "Quad4")
    # # AbaqusReader.register_element!("U3", 8, "Hex8")

    # abq_data = abaqus_read_mesh(file_name)

    # # nodal_coords = abq_data["nodes"]
    # # nodal_coords = values(abq_data["nodes"])
    # # node_id_map = keys(abq_data["nodes"])
    # # perm = sortperm(keys(abq_data["nodes"]) |> collect)
    # # node_id_map = collect(keys(abq_data["nodes"]))[perm]
    # # nodal_coords = collect(values(abq_data["nodes"]))[perm]

    # # node stuff
    # # perm = Int[]
    # abq_to_fem = Dict{Int, Int}()
    # for (n, k) in enumerate(keys(abq_data["nodes"]))
    #     # push!(perm, k)
    #     abq_to_fem[k] = n
    # end

    # nodal_coords = Matrix{Float64}(undef, length(abq_data["nodes"][1]), length(abq_data["nodes"]))
    # for (n, v) in enumerate(values(abq_data["nodes"]))
    #     nodal_coords[:, n] = v
    # end
    # node_id_map = 1:size(nodal_coords, 2) |> collect

    # # element stuff
    # element_block_names = keys(abq_data["element_sets"])
    # # NOTE these are mapping el id to type name
    # # we need to sort through this with the 
    # element_types = abq_data["element_types"]
    # # abq_conns = values(abq_data["element_sets"])
    # blocks = abq_data["element_sets"]

    # # @show length(element_types)
    # # @show length(blocks)

    # # collect element type by block
    # for block in values(blocks)
    #     # el_types = Vector{Symbol}(undef, length(block))
    #     el_types = Symbol[]
    #     for (k, v) in element_types
    #         # el_types[k] = v
    #         push!(el_types, v)
    #     end
    #     display(el_types)
    #     # @assert length(unique(el_types)) == 1
    # end

    # # element_conns = L2ElementField[]

    # # for conn in abq_conns
    # #     # block_el_ids = map(x -> )
    # #     # n_elem = length(conn)
    # #     # fem_conns = Matrix{Int}(undef, )
    # #     # @show el_type
    # #     # el_type = unique(el_type)
    # #     # @show el_type
    # #     # @assert length(el_type) == 1
    # #     # el_type = el_type[1]
    # #     # @show el_type

    # #     if el_type == :Hex8
    # #         nnpe = 8
    # #     elseif el_type == :Quad4
    # #         nnpe = 4
    # #     elseif el_type == :Tri3
    # #         nnpe = 3
    # #     elseif el_type == :Tri6
    # #         nnpe = 6
    # #     elseif el_type == :Tet4
    # #         nnpe = 4
    # #     elseif el_type == :Tet10
    # #         nnpe = 10
    # #     else
    # #         # @assert false "Got unnsupported el_type $el_type"
    # #     end

    # #     ne = length(conn) ÷ nnpe
    # #     conn_rs = reshape(conn, nnpe, ne)
    # #     display(conn_rs )
    # # end
    # # # nodesets
    # # nodeset_nodes = Dict{Symbol, Vector{Int}}()
    # # for (k, v) in abq_data["node_sets"]
    # #     nodeset_nodes[Symbol(k)] = v
    # # end

    # # TODO need to set up sidesets

    # # TODO edge/face connectivities
    # # abq_data
    # # nodal_coords
    # # abq_conns
    # abq_data
end

end # module
