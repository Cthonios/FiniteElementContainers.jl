# For a periodic pair in direction dir_id, match on the OTHER coordinates
# function _transverse_key(coords, node, dir_id, tolerance)
#     ndim = size(coords, 1)
#     return tuple([round(Int, coords[d, node] / tolerance) for d in 1:ndim if d != dir_id]...)
# end

struct PeriodicBC{F} <: AbstractBC{F}
    direction::String
    func::F
    side_a_sset::String
    side_b_sset::String
    var_name::String

    function PeriodicBC(
        var_name::String, direction::String, 
        func,
        side_a_sset::String, side_b_sset::String,
    )
        return new{typeof(func)}(direction, func, side_a_sset, side_b_sset, var_name)
    end
end

struct PeriodicBCContainer{
    IV <: AbstractVector{<:Integer},
    RV <: AbstractVector{<:Number}
} <: AbstractBCContainer{IV, RV}
    side_a_dofs::IV
    side_a_nodes::IV
    side_b_dofs::IV
    side_b_nodes::IV
    vals::RV

    # TODO add sanity check that tolerance isn't stupid based
    # on mesh sizes
    # TODO also move SVector stuff into barrier func
    function PeriodicBCContainer(
        mesh, dof::DofManager, pbc::PeriodicBC;
        tolerance=1.e-6
    )
        # get direction
        if pbc.direction == "x"
            dir_id = 1
        elseif pbc.direction == "y"
            dir_id = 2
        elseif pbc.direction == "z"
            @assert size(mesh.nodal_coords, 1) == 3
            dir_id = 3
        end

        # set up some book keeping
        side_a_bk = BCBookKeeping(mesh, dof, pbc.var_name, sideset_name = pbc.side_a_sset)
        side_b_bk = BCBookKeeping(mesh, dof, pbc.var_name, sideset_name = pbc.side_b_sset)

        # sort the side a sets
        dof_perm = _unique_sort_perm(side_a_bk.dofs)
        side_a_dofs = side_a_bk.dofs[dof_perm]
        side_a_nodes = side_a_bk.nodes[dof_perm]
        resize!(side_a_bk.dofs, length(side_a_dofs))
        resize!(side_a_bk.nodes, length(side_a_nodes))
        copyto!(side_a_bk.dofs, side_a_dofs)
        copyto!(side_a_bk.nodes, side_a_nodes)

        # now find the nodes in side b associated to those in side a
        coords = mesh.nodal_coords
        coords_to_side_a = Dict{Int, Int}()
        coords_to_side_b = Dict{Int, Int}()
        
        for node in side_a_nodes
            temp = round(Int, coords[dir_id, node] / tolerance)
            # temp = _transverse_key(coords, node, dir_id, tolerance)
            coords_to_side_a[temp] = node
        end

        for node in side_b_bk.nodes
            temp = round(Int, coords[dir_id, node] / tolerance)
            # temp = _transverse_key(coords, node, dir_id, tolerance)
            coords_to_side_b[temp] = node
        end

        @assert length(coords_to_side_a) == length(coords_to_side_b) "Side a and side b have different numbers of nodes"

        side_b_nodes = Int[]
        for node in side_a_nodes
            temp_coord = round(Int, coords[dir_id, node] / tolerance)
            side_b_node = coords_to_side_b[temp_coord]
            push!(side_b_nodes, side_b_node)
        end

        dofs_all = reshape(1:length(dof), size(dof))
        dof_id = _dof_index_from_var_name(dof, pbc.var_name)
        side_b_dofs = Int[]
        for node in side_b_nodes
            push!(side_b_dofs, dofs_all[dof_id, node])
        end
        
        vals = zeros(length(side_a_dofs))

        return PeriodicBCContainer(
            side_a_dofs, side_a_nodes,
            side_b_dofs, side_b_nodes,
            vals
        )
    end

    function PeriodicBCContainer{IV, RV}() where {IV, RV}
        new{IV, RV}(Int[], Int[], Int[], Int[], Float64[])
    end

    function PeriodicBCContainer(
        side_a_dofs::IV, side_a_nodes::IV,
        side_b_dofs::IV, side_b_nodes::IV, vals::RV
    ) where {IV, RV}
        new{IV, RV}(side_a_dofs, side_a_nodes, side_b_dofs, side_b_nodes, vals)
    end
end

function Adapt.adapt_structure(to, bc::PeriodicBCContainer)
    return PeriodicBCContainer(
        adapt(to, bc.side_a_dofs),
        adapt(to, bc.side_a_nodes),
        adapt(to, bc.side_b_dofs),
        adapt(to, bc.side_b_nodes),
        adapt(to, bc.vals)
    )
end

function _update_bc_values!(bc::PeriodicBCContainer, func, X, t)
    fec_foreach(bc.vals) do n
        node = bc.side_b_nodes[n]
        ND = num_fields(X)
        if ND == 1
            X_temp = SVector{ND, eltype(X)}(X[1, node])
        elseif ND == 2
            X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node])
        elseif ND == 3
            X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node], X[3, node])
        end
        bc.vals[n] = func.func(X_temp, t)
    end
    return nothing
end

struct PeriodicBCFunction{F} <: AbstractBCFunction{F}
    func::F

    function PeriodicBCFunction(func::F) where F <: Function
        new{F}(func)
    end

    function PeriodicBCFunction{F}(func::F) where F <: Function
        new{F}(func)
    end
end

struct PeriodicBCs{
    BCFuncs,
    IV      <: AbstractVector{<:Integer},
    RV      <: AbstractVector{<:Number}
} <: AbstractBCs{BCFuncs}
    bc_caches::Vector{PeriodicBCContainer{IV, RV}}
    bc_funcs::BCFuncs

    function PeriodicBCs(mesh, dof, periodic_bcs)
        if length(periodic_bcs) == 0
            bc_cache = PeriodicBCContainer{Vector{Int}, Vector{Float64}}[]
            bc_funcs = PeriodicBCFunction[]
            new{typeof(bc_funcs), Vector{Int}, Vector{Float64}}(
                bc_cache, bc_funcs
            )
        end

        periodic_bc_funcs = map(x -> PeriodicBCFunction(x.func), periodic_bcs)
        periodic_bcs = PeriodicBCContainer.((mesh,), (dof,), periodic_bcs)
        return PeriodicBCs(periodic_bcs, periodic_bc_funcs)
    end

    function PeriodicBCs{F}(mesh, dof, periodic_bcs) where F <: Function
        IV = Vector{Int}
        RV = Vector{Float64}
        if length(periodic_bcs) == 0
            bc_cache = PeriodicBCContainer{IV, RV}[]
            bc_funcs = PeriodicBCFunction{F}[]
            return new{typeof(bc_funcs), IV, RV}(
                bc_cache, bc_funcs
            )
        end

        bc_funcs = map(x -> PeriodicBCFunction(x.func), periodic_bcs)
        bc_caches = PeriodicBCContainer.((mesh,), (dof,), periodic_bcs)
        new{typeof(bc_funcs), IV, RV}(
            bc_caches, bc_funcs
        )
    end

    function PeriodicBCs(bc_caches::Vector{PeriodicBCContainer{IV, RV}}, bc_funcs::Funcs) where {IV, RV, Funcs}
        new{Funcs, IV, RV}(bc_caches, bc_funcs)
    end
end

function Adapt.adapt_structure(to, pbcs::PeriodicBCs)
    # NOTE
    # below logic is needed due to improper
    # adapt mapping for an empty array in julia 1.10/1.11
    # where Vector{T}(undef, 0) gets mappend to Vector{Any}
    if length(pbcs) > 0
        bc_caches = map(x -> adapt(to, x), pbcs.bc_caches)
    else
        temp_int = adapt(to, zeros(Int, 0))
        temp_floats = adapt(to, zeros(Float64, 0))
        bc_cache = PeriodicBCContainer(
            copy(temp_int), copy(temp_int),
            copy(temp_int), copy(temp_int),
            temp_floats
        )
        bc_caches = Vector{typeof(bc_cache)}(undef, 0)
    end
    return PeriodicBCs(
        bc_caches,
        pbcs.bc_funcs
    )
end

# below is for lagrange multiplier approach
function _constraint_matrix(dof::DofManager, pbcs::PeriodicBCs)
    Is, Js, Vs = Int[], Int[], Float64[]
    n = 1
    for bc in pbcs.bc_caches
        for (dof_a, dof_b) in zip(bc.side_a_dofs, bc.side_b_dofs)
            # side a dof contribution
            push!(Is, n)
            push!(Js, dof_a)
            push!(Vs, -1.)

            # side b dof contribution
            push!(Is, n)
            push!(Js, dof_b)
            push!(Vs, 1.)
            n = n + 1
        end
    end

    return sparse(Is, Js, Vs)
end

function _constraint_matrix_mask(dof_manager::DofManager, pbc::PeriodicBCs)
    Is, Js, Vs = Int[], Int[], Float64[]
    side_a_dofs = mapreduce(x -> x.side_a_dofs, vcat, pbc.bc_caches)
    side_b_dofs = mapreduce(x -> x.side_b_dofs, vcat, pbc.bc_caches)
    # @show length(side_a_dofs)
    for dof in 1:length(dof_manager)
        if dof in side_b_dofs
            id = findfirst(isequal(dof), side_b_dofs)
            if id > length(dof_manager)
                @assert false
            end
            # push!(Is, dof)
            # push!(Js, side_a_dofs[id])
            # push!(Vs, 1.)

            push!(Is, dof)
            push!(Js, side_b_dofs[id])
            push!(Vs, 1.)
        else
            push!(Is, dof)
            push!(Js, dof)
            push!(Vs, 1.)
        end
    end
    return sparse(Is, Js, Vs)
end

function _create_constraint_field(dof::DofManager, pbcs::PeriodicBCs)
    n_constraints = mapreduce(x -> length(x.side_a_dofs), +, pbcs.bc_caches)
    return zeros(n_constraints)
end

Base.length(bcs::PeriodicBCs) = length(bcs.bc_funcs)

function Base.show(io::IO, bcs::PeriodicBCs)
    # for (n, (cache, func)) in enumerate(zip(bcs.bc_cache, bcs.bc_funcs))
    show(io, "PeriodicBC:")
    show(io, bcs.bc_cache)
    # show(io, bcs.bc_lengths)
    # show(io, func)
    show(io, "\n")
    # end
end

function periodic_dofs(bcs::PeriodicBCs)
    if length(bcs) > 0
        side_a_dofs = mapreduce(x -> x.side_a_dofs, vcat, bcs.bc_caches)
        side_b_dofs = mapreduce(x -> x.side_b_dofs, vcat, bcs.bc_caches)
        perm = _unique_sort_perm(side_b_dofs) # sort by side b, side a will seem unsorted
        side_a_dofs = side_a_dofs[perm]
        side_b_dofs = side_b_dofs[perm]
    else
        return Vector{Int}(undef, 0), Vector{Int}(undef, 0)
    end
    return side_a_dofs, side_b_dofs
end

function update_bc_values!(bcs::PeriodicBCs, X, t)
    # for bc_cache, bc_func in zbcs.bc_caches
    for n in axes(bcs.bc_caches, 1)
        _update_bc_values!(bcs.bc_caches[n], bcs.bc_funcs[n], X, t)
    end
    return nothing
end

function update_field_periodic_bcs!(U, bcs::PeriodicBCs)
    for n in axes(bcs.bc_caches, 1)
        cache = bcs.bc_caches[n]
        fec_foreach(cache.side_b_dofs) do I
            side_a_dof = cache.side_a_dofs[I]
            side_b_dof = cache.side_b_dofs[I]
            U[side_b_dof] = U[side_a_dof] + cache.vals[I]
        end
    end
end
