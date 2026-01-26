using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays

function create_dof_to_unknown(dof, n_total_dofs, dirichlet_dofs)
    unknown_to_dof = Vector{eltype(dirichlet_dofs)}(undef, n_total_dofs)
    ids = 1:length(dof)
    n = 1
    for dof in ids
        if !insorted(dof, dirichlet_dofs)
            unknown_to_dof[n] = dof
            n += 1
        end
    end
    dof_to_unknown = Dict([(x, y) for (x, y) in zip(unknown_to_dof, 1:length(unknown_to_dof))])
    return dof_to_unknown
end

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])
u(x) = x[1] + x[2]
h = 0.2
Ae = (h^2 / 6) * [
     4.0  -1.0  -1.0  -2.0
    -1.0   4.0  -2.0  -1.0
    -1.0  -2.0   4.0  -1.0
    -2.0  -1.0  -1.0   4.0
]
bc_func(_, _) = 0.
mesh_file = Base.source_dir() * "/square.g"
output_file = Base.source_dir() * "/output.e"
num_dofs = 1
num_ranks = 4
distribute = identity

ranks = distribute(LinearIndices((num_ranks,)))

# serial problem setup
mesh = UnstructuredMesh(mesh_file)
mesh.nodal_coords .= 2. * mesh.nodal_coords
V = FunctionSpace(mesh, H1Field, Lagrange)
uvar = ScalarFunction(V, :u)
asm = SparseMatrixAssembler(uvar)
physics = Poisson(f)
props = create_properties(physics)
dbcs = DirichletBC[
    DirichletBC(:u, bc_func; nodeset_name = :boundary),
]
p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)
boundary_dofs = unique(sort(mapreduce(x -> x.dofs, vcat, p.dirichlet_bcs.bc_caches)))


# decomp mesh and get colorings
map_main(ranks) do rank
    decomp(mesh_file, num_ranks)
end
dofs_to_colors, elems_to_colors = FiniteElementContainers.global_colorings(mesh_file, num_dofs, num_ranks)

meshes = map(ranks) do rank
    mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
    mesh.nodal_coords .= 2. * mesh.nodal_coords
    return mesh
end

# now figure out global ids which are dirichlet bcs
# and remove those from the dofs_to_colors
# or create unknowns_to_colors
unknowns_to_colors = copy(dofs_to_colors)
deleteat!(unknowns_to_colors, boundary_dofs)

num_dofs = map(ranks) do rank
    count(x -> x == rank, unknowns_to_colors)
end

dof_to_unknown = create_dof_to_unknown(asm.dof, sum(num_dofs), boundary_dofs)

# Next create a partition from color
# unknown_parts = partition_from_color(ranks, unknowns_to_colors)
dof_parts = partition_from_color(ranks, dofs_to_colors)
elem_parts = partition_from_color(ranks, elems_to_colors)
unknown_parts = variable_partition(num_dofs, sum(num_dofs))
# unknown_parts = partition_from_color(ranks, unknowns_to_colors)

IIs, JJs, VVs = tuple_of_arrays(map(meshes, elem_parts, ranks) do mesh, part, rank
    IIs, JJs, VVs = Int[], Int[], Float64[]

    for (key, conn) in mesh.element_conns
        for e in axes(conn, 2)
            glob_conn = @views mesh.node_id_map[conn[:, e]]
            for i in axes(glob_conn, 1)
                if insorted(glob_conn[i], boundary_dofs)
                    continue
                end

                for j in axes(glob_conn, 1)
                    if insorted(glob_conn[j], boundary_dofs)
                        continue
                    end

                    push!(IIs, dof_to_unknown[glob_conn[i]])
                    push!(JJs, dof_to_unknown[glob_conn[j]])
                    push!(VVs, Ae[i, j])
                end
            end
        end
    end

    IIs, JJs, VVs
end)

Is, Vs = tuple_of_arrays(map(meshes, elem_parts, ranks) do mesh, part, rank
    Is, Vs = Int[], Float64[]
    ue = zeros(size(Ae, 2))
    ge = similar(ue)
    X = mesh.nodal_coords
    for (key, conn) in mesh.element_conns
        for e in axes(conn, 2)
            glob_conn = @views mesh.node_id_map[conn[:, e]]
            fill!(ue, 0.0)

            for i in axes(glob_conn, 1)
                if insorted(glob_conn[i], boundary_dofs)
                    ue[i] = u(X[:, conn[i, e]])
                end
            end

            mul!(ge, Ae, ue)
            for i in axes(glob_conn, 1)
                if insorted(glob_conn[i], boundary_dofs)
                    continue
                end

                push!(Is, dof_to_unknown[glob_conn[i]])
                push!(Vs, -ge[i])
            end
        end
    end 
    return Is, Vs
end)

A = psparse(IIs, JJs, VVs, unknown_parts, unknown_parts) |> fetch
b = pvector(Is, Vs, unknown_parts) |> fetch
x = IterativeSolvers.cg(A, b, verbose=i_am_main(rank))

# setup analytic solution
map(meshes, dof_parts) do mesh, part
    # ids = global_to_local(part)
    # display(part)
    # display(local_values(part))
    X = mesh.nodal_coords
    u.(eachcol(X))
end
