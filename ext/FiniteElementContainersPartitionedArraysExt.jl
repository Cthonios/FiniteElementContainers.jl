module FiniteElementContainersPartitionedArraysExt

using Exodus
using FiniteElementContainers
using PartitionedArrays

function FiniteElementContainers.UnstructuredMesh(file_name::String, ranks, np, n_dofs)
    # run decomp
    map(ranks) do rank
        if rank == 1
            decomp(file_name, np)
        end
    end

    # setup partitions
    global_to_color = Exodus.collect_global_to_color(file_name, np, n_dofs)
    parts = partition_from_color(ranks, file_name, global_to_color)

    node_id_maps = map(ranks) do rank
        file_name_rank = "$file_name.$np.$(rank - 1)"
        exo = ExodusDatabase(file_name_rank, "r")
        Exodus.read_id_map(exo, NodeMap)
    end

    node_id_maps, parts
end

end # module