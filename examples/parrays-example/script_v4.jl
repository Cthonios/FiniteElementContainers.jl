using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])
bc_func(_, _) = 0.
mesh_file = Base.source_dir() * "/square.g"
output_file = Base.source_dir() * "/output.e"

num_dofs = 1
num_ranks = 4
distribute = identity
# distribute = distribute_with_mpi
ranks = distribute(LinearIndices((num_ranks,)))

mesh = UnstructuredMesh(mesh_file, num_ranks, ranks)
V    = FunctionSpace(mesh, H1Field, Lagrange)
u    = ScalarFunction(V, "u")
dbcs = DirichletBC[
    DirichletBC("u", bc_func; nodeset_name = "boundary")
]
dof  = DofManager(u, dbcs, num_ranks, ranks, mesh_file, mesh)
asm  = SparseMatrixAssembler(dof)

# stuff to still work out below
mat_pattern_new = FiniteElementContainers.create_matrix_sparsity_pattern(dof)
vec_pattern_new = FiniteElementContainers.create_vector_sparsity_pattern(dof)
X = nodal_coordinates(dof)

u_analytic(x) = 0.5 * (x[1] + x[2])
h = 0.1
Ae = (h^2 / 6) * [
     4.0  -1.0  -1.0  -2.0
    -1.0   4.0  -2.0  -1.0
    -1.0  -2.0   4.0  -1.0
    -2.0  -1.0  -1.0   4.0
]
VVs_new = map(dof.var.fspace) do fspace
    VVs = Float64[]
    Ae_vec = vec(Ae)
    for b in 1:FiniteElementContainers.num_blocks(fspace)
        conn = connectivity(fspace, b)
        for e in axes(conn, 2)
            for j in axes(conn, 1)
                for i in axes(conn, 1)
                    push!(VVs, Ae[i, j])
                end
            end
        end
    end
    VVs
end

Xs = nodal_coordinates(dof)
Vs_new = map(dof.var.fspace, dof.local_dof_managers, dof.field_partition.parts, partition(Xs)) do fspace, local_dof, field_part, X
    Vs = Float64[]
    ue = zeros(size(Ae, 2))
    ge = similar(ue)
    field_ltg = local_to_global(field_part)
    for b in 1:FiniteElementContainers.num_blocks(fspace)
        block_local_conns = connectivity(fspace, b)
        block_global_conns = field_ltg[block_local_conns]
        for e in axes(block_global_conns, 2)
            fill!(ue, 0.0)
            for i in axes(block_global_conns, 1)
                # this is for bcs here
                # if insorted(block_global_conns[i, e], dof.dirichlet_dofs)
                if insorted(block_local_conns[i, e], local_dof.dirichlet_dofs)
                    ue[i] = u_analytic(fspace.coords[:, block_local_conns[i, e]])
                end
            end

            mul!(ge, Ae, ue)

            for i in axes(block_global_conns, 1)
                push!(Vs, -ge[i])
            end
        end
    end
    Vs
end

A_new = psparse(mat_pattern_new, VVs_new) |> fetch
b_new = pvector(vec_pattern_new, Vs_new) |> fetch
x_new = IterativeSolvers.cg(A_new, b_new, verbose = i_am_main(rank))

U = create_field(dof)
# update_field_dirichlet_bcs!(U, dof)
update_field_unknowns!(U, dof, x_new)

map(partition(U), partition(Xs), dof.var.fspace, mesh, ranks) do U_local, X, fspace, mesh_local, rank
    u_temp = ScalarFunction(fspace, "u")
    x_temp = VectorFunction(fspace, "x")
    mesh_file = mesh_local.mesh_obj.file_name
    output_file_temp = output_file * ".$(num_ranks).$(rank - 1)"
    pp = PostProcessor(mesh_local, output_file_temp, u_temp, x_temp)
    write_times(pp, 1, 0.0)
    write_field(pp, 1, ["u"], U_local)
    write_field(pp, 1, ["x_x", "x_y"], X)
    close(pp)
end

map_main(ranks) do rank
    # output_file = splitext(mesh_file)[1] * ".e.$(num_ranks)."
    epu(output_file * ".$(num_ranks).$(rank - 1)")
end
