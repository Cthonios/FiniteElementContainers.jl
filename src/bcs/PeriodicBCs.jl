struct PeriodicBC{F} <: AbstractBC{F}
    direction::Symbol
    func::F
    side_a_sset::Symbol
    side_b_sset::Symbol
    var_name::Symbol
end

function PeriodicBC(
    var_name::String, direction::String, 
    side_a_sset::String, side_b_sset::String,
    func
)
    return PeriodicBC(
        Symbol(direction), func, 
        Symbol(side_a_sset), Symbol(side_b_sset), 
        Symbol(var_name)
    )
end

function PeriodicBC(
    var_name::Symbol, direction::Symbol, 
    side_a_sset::Symbol, side_b_sset::Symbol,
    func
)
    return PeriodicBC(direction, func, side_a_sset, side_b_sset, var_name)
end

struct PeriodicBCContainer{
    IV <: AbstractVector{<:Integer},
    RV <: AbstractVector{<:Number}
} <: AbstractBCContainer
    side_a_dofs::IV
    side_a_nodes::IV
    side_b_dofs::IV
    side_b_nodes::IV
    vals::RV
end

# TODO add sanity check that tolerance isn't stupid based
# on mesh sizes
# TODO also move SVector stuff into barrier func
function PeriodicBCContainer(
    mesh, dof::DofManager, pbc::PeriodicBC;
    tolerance=1.e-6
)
    # get direction
    if pbc.direction == :x
        dir_id = 1
    elseif pbc.direction == :y
        dir_id = 2
    elseif pbc.direction == :z
        @assert size(mesh.nodal_coords, 1) == 3
        dir_id = 3
    end

    # set up some book keeping
    side_a_bk = BCBookKeeping(mesh, dof, pbc.var_name, sset_name=pbc.side_a_sset)
    side_b_bk = BCBookKeeping(mesh, dof, pbc.var_name, sset_name=pbc.side_b_sset)

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
        coords_to_side_a[temp] = node
    end

    for node in side_b_bk.nodes
        temp = round(Int, coords[dir_id, node] / tolerance)
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

function Adapt.adapt_structure(to, bc::PeriodicBCContainer)
    return PeriodicBCContainer(
        adapt(to, bc.side_a_dofs),
        adapt(to, bc.side_a_nodes),
        adapt(to, bc.side_b_dofs),
        adapt(to, bc.side_b_nodes),
        adapt(to, bc.vals)
    )
end

struct PeriodicBCFunction{F} <: AbstractBCFunction{F}
    func::F
end

struct PeriodicBCs{
    IV      <: AbstractVector{<:Integer},
    RV      <: AbstractVector{<:Number},
    BCFuncs <: NamedTuple
}
    bc_caches::Vector{PeriodicBCContainer{IV, RV}}
    bc_funcs::BCFuncs
end

function PeriodicBCs(mesh, dof, periodic_bcs)
    if length(periodic_bcs) == 0
        bc_caches = PeriodicBCContainer{Vector{Int}, Vector{Float64}}[]
        bc_funcs = NamedTuple()
        return PeriodicBCs(bc_caches, bc_funcs)
    end

    syms = map(x -> Symbol("periodic_bc_$x"), 1:length(periodic_bcs))
    periodic_bc_funcs = NamedTuple{tuple(syms...)}(
        map(x -> PeriodicBCFunction(x.func), periodic_bcs)
    )
    periodic_bcs = PeriodicBCContainer.((mesh,), (dof,), periodic_bcs)

    # TODO add hooks for dof removal if that's the way we
    # want to implement it downstream, but make it an optional
    # kwarg

    return PeriodicBCs(periodic_bcs, periodic_bc_funcs)
end

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

function _create_constraint_field(dof::DofManager, pbcs::PeriodicBCs)
    n_constraints = mapreduce(x -> length(x.side_a_dofs), +, pbcs.bc_caches)
    return zeros(n_constraints)
end
