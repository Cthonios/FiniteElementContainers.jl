module GmshExt

import FiniteElementContainers.GmshMesh
using FiniteElementContainers
using Gmsh: Gmsh, gmsh

const GmshFile = FileMesh{Nothing, GmshMesh}

# TODO flesh this out more
# should we really do anything beyond linear elements?
const gmsh_to_exo = Dict{String, String}(
    "Triangle 3" => "Tri3"
)

function FiniteElementContainers.FileMesh(::GmshMesh, file_name::String)
    Gmsh.initialize()
    gmsh.open(file_name)

    if splitext(file_name)[2] == ".geo"
        @info "Generating mesh using $file_name"
        dim = Int64(gmsh.model.getDimension())
        gmsh.model.mesh.generate(dim)
    end

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    return FileMesh{Nothing, GmshMesh}(file_name, nothing)
end 

function FiniteElementContainers.copy_mesh(mesh, file_2::String, ::Type{GmshMesh})
    FiniteElementContainers.write_to_file(mesh, file_2)
    return nothing
end

function FiniteElementContainers.element_blocks(mesh::GmshFile)
    dim = num_dimensions(mesh)
    phys_groups = gmsh.model.getPhysicalGroups()
    temp = filter(pg -> pg[1] == dim, phys_groups)
    block_ids = map(x -> x[2], temp)
    names = map(x -> Symbol(gmsh.model.getPhysicalName(dim, x)), block_ids)
    conns = Dict{Symbol, Matrix{Int}}()
    el_types = Dict{Symbol, Symbol}()
    el_id_maps = Dict{Symbol, Vector{Int}}()

    for (block_id, name) in zip(block_ids, names)
        temp_el_types = String[]
        temp_conns = Matrix{Int}[]
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, block_id)
            ent_type, elem_tags, node_tags = gmsh.model.mesh.getElements(dim, ent)
            @assert length(ent_type) == 1 "Not supporting mixed element blocks right now" 
            node_tags = convert(Vector{Int}, node_tags[1])
            el_type, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(ent_type[1]) 
            conn = reshape(node_tags, Int(num_nodes), length(node_tags) ÷ num_nodes)
            push!(temp_conns, conn)
            push!(temp_el_types, el_type)
        end
        @assert length(temp_conns) == 1 && length(temp_el_types) == 1
        conns[name] = temp_conns[1]
        el_types[name] = Symbol(gmsh_to_exo[temp_el_types[1]])

        # TODO fix this up
        el_id_maps[name] = Vector{Int}(undef, 0)
    end
    block_ids = convert(Vector{Int}, block_ids)
    names = Dict(zip(block_ids, names))
    return conns, el_id_maps, names, el_types
end

function FiniteElementContainers.element_ids(mesh::GmshFile)
    dim = num_dimensions(mesh)
    _, elem_tags, _ = gmsh.model.mesh.getElements(dim, -1)
    return convert(Vector{Int}, elem_tags[1])
end

function FiniteElementContainers.finalize(::GmshFile)
    Gmsh.finalize()
end

function FiniteElementContainers.nodal_coordinates_and_ids(mesh::GmshFile)
    ids, coords = gmsh.model.mesh.getNodes()
    dim = num_dimensions(mesh)
    coords = reshape(coords, 3, length(coords) ÷ 3)[1:dim, :]
    coords = H1Field(coords)
    ids = convert(Vector{Int}, ids)
    return coords, ids
end

function FiniteElementContainers.nodesets(mesh::GmshFile)
    dim = num_dimensions(mesh) - 1
    phys_groups = gmsh.model.getPhysicalGroups()
    temp = filter(pg -> pg[1] == dim, phys_groups)
    ids = map(x -> x[2], temp)
    names = map(x -> Symbol(gmsh.model.getPhysicalName(dim, x)), ids)

    # pre-allocate stuff
    nodes = Dict{Symbol, Vector{Int}}()

    for (id, name) in zip(ids, names)
        temp = Vector{Int}(undef, 0)
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, id)
            _, _, node_tags = gmsh.model.mesh.getElements(dim, ent)
            node_tags = convert(Vector{Int}, node_tags[1])
            # sort!(unique!(node_tags))
            append!(temp, node_tags)
        end
        sort!(unique!(temp))
        nodes[name] = temp
    end

    ids = convert(Vector{Int}, ids)
    names = Dict(zip(ids, names))

    return names, nodes
end

function FiniteElementContainers.num_dimensions(::GmshFile)
    return Int64(gmsh.model.getDimension())
end

function FiniteElementContainers.sidesets(mesh::GmshFile)
    @warn "Sidesets are currently not supported with Gmsh meshes"
    dim = num_dimensions(mesh) - 1
    phys_groups = gmsh.model.getPhysicalGroups()
    temp = filter(pg -> pg[1] == dim, phys_groups)
    sset_ids = map(x -> x[2], temp)
    names = map(x -> Symbol(gmsh.model.getPhysicalName(dim, x)), sset_ids)

    # pre-allocate stuff
    elems = Dict{Symbol, Vector{Int}}()
    nodes = Dict{Symbol, Vector{Int}}()
    sides = Dict{Symbol, Vector{Int}}()
    side_nodes = Dict{Symbol, Matrix{Int}}()

    # for (name, sset_id) in zip(names, sset_ids)
    #     for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, sset_id)
    #         ent_type, elem_tags, node_tags = gmsh.model.mesh.getElements(dim, ent)
    #         @assert length(ent_type) == 1 "Not supporting mixed element blocks right now" 
    #         elem_tags = convert(Vector{Int}, elem_tags[1])
    #         node_tags = convert(Vector{Int}, node_tags[1])
    #         el_type, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(ent_type[1]) 
    #         @show elem_tags
    #         @show node_tags
    #         @show gmsh.model.mesh.getNodes(dim, sset_id)
    #     end
    # end
    sset_ids = convert(Vector{Int}, sset_ids)
    names = Dict(zip(sset_ids, names))

    return elems, names, nodes, sides, side_nodes
end

end # module
