module GmshExt

import FiniteElementContainers.GmshMesh
using DocStringExtensions
using FiniteElementContainers
using Gmsh: Gmsh, gmsh

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
const GmshFile = FileMesh{Nothing, GmshMesh}

# TODO flesh this out more
# should we really do anything beyond linear elements?
const gmsh_to_exo = Dict{String, String}(
    "Quadrilateral 4" => "Quad4",
    "Quadrilateral 9" => "Quad9",
    "Triangle 3"      => "Tri3",
    "Triangle 6"      => "Tri6",
    "Tetrahedron 4"   => "Tet4",
    "Hexahedron 8"    => "Hex8",
)
const gmsh_local_sides = Dict{String, Vector{Vector{Int}}}(
    # 2D: edges of Tri3 / Quad4 -- matches FiniteElementContainers._create_local_edges
    "Quadrilateral 4" => [[1, 2], [2, 3], [3, 4], [4, 1]],
    "Quadrilateral 9" => [[1, 2, 5], [2, 3, 6], [3, 4, 7], [4, 1, 8]],
    "Triangle 3"      => [[1, 2], [2, 3], [3, 1]],
    "Triangle 6"      => [[1, 2, 4], [2, 3, 5], [3, 1, 6]],
    # 3D: faces of Tet4 / Hex8, standard Exodus face/node ordering
    "Hexahedron 8"    => [[1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7],
                          [1, 5, 8, 4], [1, 4, 3, 2], [5, 6, 7, 8]],
    "Tetrahedron 4"   => [[1, 2, 4], [2, 3, 4], [1, 4, 3], [1, 3, 2]],

)

# gmsh boundary element type names that can appear as the *boundary*
# entities of a given parent dimension. Used only to know how many nodes
# to expect / sanity check; the actual matching is purely by node-set.
const gmsh_boundary_types_by_dim = Dict{Int, Vector{String}}(
    1 => ["Line 2", "Line 3"],
    2 => ["Triangle 3", "Quadrilateral 4"],
)

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
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

"""
$(TYPEDEF)
"""
function FiniteElementContainers.element_blocks(mesh::GmshFile)
    dim = num_dimensions(mesh)
    phys_groups = gmsh.model.getPhysicalGroups()
    temp = filter(pg -> pg[1] == dim, phys_groups)
    block_ids = map(x -> x[2], temp)
    names = map(x -> gmsh.model.getPhysicalName(dim, x), block_ids)
    conns = Dict{String, Matrix{Int}}()
    el_types = Dict{String, String}()
    el_id_maps = Dict{String, Vector{Int}}()

    for (block_id, name) in zip(block_ids, names)
        temp_el_types = String[]
        temp_conns = Matrix{Int}[]
        temp_elem_tags = Vector{Int}[]
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, block_id)
            ent_type, elem_tags, node_tags = gmsh.model.mesh.getElements(dim, ent)
            @assert length(ent_type) == 1 "Not supporting mixed element blocks right now" 
            node_tags = convert(Vector{Int}, node_tags[1])
            elem_tags = convert(Vector{Int}, elem_tags[1])
            el_type, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(ent_type[1]) 
            conn = reshape(node_tags, Int(num_nodes), length(node_tags) ÷ num_nodes)
            push!(temp_conns, conn)
            push!(temp_el_types, el_type)
            push!(temp_elem_tags, elem_tags)
        end
        @assert length(temp_conns) == 1 && length(temp_el_types) == 1
        conns[name] = temp_conns[1]
        el_types[name] = gmsh_to_exo[temp_el_types[1]]

        # global element tags for this block, in the same column order as
        # conns[name] -- this is what _setup_sideset uses via indexin to
        # convert global sideset element tags into local block-column indices
        el_id_maps[name] = temp_elem_tags[1]
    end
    block_ids = convert(Vector{Int}, block_ids)
    names_map = Dict(zip(block_ids, names))
    return conns, el_id_maps, names, names_map, el_types
end

"""
$(TYPEDEF)
"""
function FiniteElementContainers.element_ids(mesh::GmshFile)
    dim = num_dimensions(mesh)
    _, elem_tags, _ = gmsh.model.mesh.getElements(dim, -1)
    return convert(Vector{Int}, elem_tags[1])
end

function FiniteElementContainers.finalize(::GmshFile)
    Gmsh.finalize()
end

"""
$(TYPEDEF)
"""
function FiniteElementContainers.nodal_coordinates_and_ids(mesh::GmshFile)
    ids, coords = gmsh.model.mesh.getNodes()
    dim = num_dimensions(mesh)
    coords = reshape(coords, 3, length(coords) ÷ 3)[1:dim, :]
    coords = H1Field(coords)
    ids = convert(Vector{Int}, ids)
    return coords, ids
end

"""
$(TYPEDEF)
"""
function FiniteElementContainers.nodesets(mesh::GmshFile)
    dim = num_dimensions(mesh) - 1
    phys_groups = gmsh.model.getPhysicalGroups()
    temp = filter(pg -> pg[1] == dim, phys_groups)
    ids = map(x -> x[2], temp)
    names = map(x -> gmsh.model.getPhysicalName(dim, x), ids)

    # pre-allocate stuff
    nodes = Dict{String, Vector{Int}}()

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

"""
$(TYPEDSIGNATURES)

Builds a map from a canonical (sorted) set of boundary-entity nodes to the
parent element tag and the local side number of that element, for *all*
`dim`-dimensional elements in the mesh. This lets us match a lower
dimensional boundary element (from a sideset physical group) back to the
volume/area element it bounds, and the side number on that element, the
way Exodus side sets are defined.

Works for both:
- `dim == 2`: boundary = edges of Tri3 / Quad4
- `dim == 3`: boundary = faces of Tet4 / Hex8

The node ordering within a side only matters for orientation-sensitive uses
elsewhere (e.g. `_create_local_edges`); here we key on the *sorted* node
tuple, since a boundary entity and its parent's matching side are made up
of exactly the same set of nodes regardless of traversal order.

Higher order or otherwise unsupported parent element types are skipped
with a warning rather than silently mismatched.
"""
function _build_boundary_to_parent_map(dim::Int)
    side_map = Dict{Vector{Int}, Tuple{Int, Int}}()
    elem_types, elem_tags_by_type, node_tags_by_type = gmsh.model.mesh.getElements(dim, -1)

    for (et, tags, nds) in zip(elem_types, elem_tags_by_type, node_tags_by_type)
        el_type, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(et)
        local_sides = get(gmsh_local_sides, el_type, nothing)
        if local_sides === nothing
            @warn "Element type \"$el_type\" is not supported for sideset construction; boundary elements bordering this type will not be matched"
            continue
        end

        tags = convert(Vector{Int}, tags)
        nds = convert(Vector{Int}, nds)
        nn = Int(num_nodes)

        for (e, tag) in enumerate(tags)
            conn = @views nds[(e - 1) * nn + 1:e * nn]
            for (side_num, local_side) in enumerate(local_sides)
                side_nodes = sort(conn[local_side])
                side_map[side_nodes] = (tag, side_num)
            end
        end
    end

    return side_map
end

"""
$(TYPEDEF)

Constructs side sets (for e.g. Neumann BCs) from gmsh physical groups
of dimension `num_dimensions(mesh) - 1`. Each boundary element in the
physical group is matched against its parent volume/area element via
shared nodes, using `_build_boundary_to_parent_map`, to recover the
(element, side) pairs Exodus-style side sets require.

Supports:
- 2D meshes (boundary = linear edges of Tri3 / Quad4)
- 3D meshes (boundary = linear faces of Tet4 / Hex8)

Higher-order parent elements (e.g. Quad9) are not yet supported and will
produce a warning + skipped boundary elements rather than a hard failure.
"""
function FiniteElementContainers.sidesets(mesh::GmshFile)
    dim = num_dimensions(mesh)
    bdim = dim - 1

    if dim != 2 && dim != 3
        @warn "Gmsh sidesets are only supported for 2D (edge) and 3D (face) meshes. Returning empty sidesets."
        return (
            Dict{String, Vector{Int}}(), Dict{Int, String}(),
            Dict{String, Vector{Int}}(), Dict{String, Vector{Int}}(),
            Dict{String, Matrix{Int}}()
        )
    end

    phys_groups = gmsh.model.getPhysicalGroups()
    temp = filter(pg -> pg[1] == bdim, phys_groups)
    sset_ids = map(x -> x[2], temp)
    names = map(x -> gmsh.model.getPhysicalName(bdim, x), sset_ids)

    elems = Dict{String, Vector{Int}}()
    nodes = Dict{String, Vector{Int}}()
    sides = Dict{String, Vector{Int}}()
    side_nodes = Dict{String, Matrix{Int}}()

    side_map = _build_boundary_to_parent_map(dim)

    for (sset_id, name) in zip(sset_ids, names)
        temp_elems = Int[]
        temp_sides = Int[]
        temp_side_nodes = Vector{Vector{Int}}(undef, 0)
        temp_nodes = Int[]

        for ent in gmsh.model.getEntitiesForPhysicalGroup(bdim, sset_id)
            ent_types, ent_elem_tags, ent_node_tags = gmsh.model.mesh.getElements(bdim, ent)
            for (et, tags, nds) in zip(ent_types, ent_elem_tags, ent_node_tags)
                bnd_type, _, _, num_nodes_elem, _, _ = gmsh.model.mesh.getElementProperties(et)
                nn = Int(num_nodes_elem)

                tags = convert(Vector{Int}, tags)
                nds = convert(Vector{Int}, nds)

                for e in 1:length(tags)
                    conn = nds[(e - 1) * nn + 1:e * nn]
                    key = sort(conn)

                    if !haskey(side_map, key)
                        @warn "Could not find a parent element for boundary $(bnd_type) with nodes $conn in sideset \"$name\"; skipping this boundary element"
                        continue
                    end

                    parent_tag, side_num = side_map[key]
                    push!(temp_elems, parent_tag)
                    push!(temp_sides, side_num)
                    push!(temp_side_nodes, conn)
                    append!(temp_nodes, conn)
                end
            end
        end

        perm = sortperm(temp_elems)
        elems[name] = temp_elems[perm]
        sides[name] = temp_sides[perm]

        if length(temp_elems) == 0
            side_nodes[name] = Matrix{Int}(undef, 0, 0)
        else
            num_nodes_per_side = length(temp_side_nodes[1])
            # all boundary elements in a physical group should share the
            # same face/edge node count; warn (don't crash) if not
            if !all(x -> length(x) == num_nodes_per_side, temp_side_nodes)
                @warn "Mixed boundary element node-counts found in sideset \"$name\"; side_nodes matrix may be malformed"
            end
            sn = reduce(hcat, temp_side_nodes)
            # match the (1, N) flattened convention used in the Exodus ext
            side_nodes[name] = reshape(sn[:, perm], 1, length(temp_nodes))
        end

        nodes[name] = sort(unique(temp_nodes))
    end

    sset_ids = convert(Vector{Int}, sset_ids)
    names_map = Dict(zip(sset_ids, names))

    return elems, names_map, nodes, sides, side_nodes
end

end # module
