abstract type AbstractInitialCondition{F} end
abstract type AbstractInitialConditionContainer end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```InitialCondition```.
"""
struct InitialCondition{F} <: AbstractInitialCondition{F}
    func::F
    block_name::EntityName
    nset_name::EntityName
    sset_name::EntityName
    var_name::String

    """
    $(TYPEDEF)
    $(TYPEDSIGNATURES)
    $(TYPEDFIELDS)
    """
    function InitialCondition(
        var_name::String, func::Function;
        block_name::EntityName = nothing,
        nodeset_name::EntityName = nothing,
        sideset_name::EntityName = nothing
    )
        if block_name === nothing && nodeset_name === nothing && sideset_name === nothing
            _entity_not_provided_error(
            "block_name, nodeset_name, or sideset_name required" *
            " as input arguments in DirichletBC"
            )
        end
        count = (block_name !== nothing) +
                (nodeset_name !== nothing) +
                (sideset_name !== nothing)
        if count != 1
            _unsure_entity_type_error("More than one entity type specificed in DirichletBC")
        end
        # new{typeof(func)}(func, block_name, nodeset_name, sideset_name, var_name)
        return InitialCondition{typeof(func)}(var_name, func, block_name, nodeset_name, sideset_name)
    end

    function InitialCondition{F}(var_name::String, func::F, block_name, nodeset_name, sideset_name) where F
        new{typeof(func)}(func, block_name, nodeset_name, sideset_name, var_name)
    end
end

struct InitialConditionContainer{
    IV <: AbstractArray{<:Integer, 1},
    RV <: AbstractArray{<:Number, 1}
} <: AbstractInitialConditionContainer
    dofs::IV
    locations::IV
    vals::RV

    # NOTE hardcoded to H1Field behavior the way
    # TODO maybe we should initialize things off of fspaces?
    # Or actually, use the information from the var in dof
    # to "do the right thing" depending upon the field type
    # this is set up
    function InitialConditionContainer(mesh, dof, ic::InitialCondition)
        # bk = BCBookKeeping(mesh, dof, ic.var_name; block_name = ic.block_name)
        dof_index = _dof_index_from_var_name(dof, ic.var_name)
        all_dofs = reshape(1:length(dof), size(dof))
        if ic.block_name !== nothing
            conns = mesh.element_conns[ic.block_name]
            nodes = sort(unique(conns))
        elseif ic.nset_name !== nothing
            nodes = mesh.nodeset_nodes[ic.nset_name]
        elseif ic.sset_name !== nothing
            nodes = mesh.sideset_nodes[ic.sset_name]
        end

        # gather dofs associated with nodes
        dofs = all_dofs[dof_index, nodes]

        # sort nodes and dofs for dirichlet bc
        dof_perm = _unique_sort_perm(dofs)
        dofs = dofs[dof_perm]
        nodes = nodes[dof_perm]

        vals = zeros(length(dofs))
        new{typeof(dofs), typeof(vals)}(dofs, nodes, vals)
    end
end

function Adapt.adapt_structure(to, ic::InitialConditionContainer)
    return InitialConditionContainer(
        adapt(to, ic.dofs),
        adapt(to, ic.locations),
        adapt(to, ic.vals)
    )
end

function Base.length(ic::InitialConditionContainer)
    return length(ic.dofs)
end

function Base.show(io::IO, ic::InitialConditionContainer)
    println(io, "$(typeof(ic).name.name):")
    println(io, "  Number of active dofs = $(length(ic.dofs))")
end

KA.get_backend(ic::InitialConditionContainer) = KA.get_backend(ic.dofs)

function _update_field_ics!(U, ic::InitialConditionContainer)
    fec_foreach(ic.dofs) do n
        dof = ic.dofs[n]
        val = ic.vals[n]
        U[dof] = val
    end
    return nothing
end

function _update_ic_values!(ic::InitialConditionContainer, func, X)
    fec_foreach(ic.locations) do n
        ND = num_fields(X)
        loc = ic.locations[n]
      
        # hacky for now, but it works
        # can't do X[:, node] on the GPU, this results in a dynamic
        # function call
        if ND == 1
            X_temp = SVector{ND, eltype(X)}(X[1, loc])
        elseif ND == 2
            X_temp = SVector{ND, eltype(X)}(X[1, loc], X[2, loc])
        elseif ND == 3
            X_temp = SVector{ND, eltype(X)}(X[1, loc], X[2, loc], X[3, loc])
        end
        ic.vals[n] = func(X_temp)
    end
end

struct InitialConditionFunction{F}
    func::F
end

function (func::InitialConditionFunction{F})(x) where F
    return func.func(x)
end

struct InitialConditions{
    ICFuncs,
    IV      <: AbstractArray{<:Integer, 1},
    RV      <: AbstractArray{<:Number, 1},
}
    ic_caches::Vector{InitialConditionContainer{IV, RV}}
    ic_funcs::ICFuncs

    function InitialConditions(ic_caches::Vector{InitialConditionContainer{IV, RV}}, ic_funcs) where {IV, RV}
        new{typeof(ic_funcs), IV, RV}(ic_caches, ic_funcs)
    end

    function InitialConditions(mesh, dof, ics)
        if length(ics) == 0
            ic_caches = InitialConditionContainer{Vector{Int}, Vector{Float64}}[]
            ic_funcs = InitialConditionFunction[]
        else
            ic_caches = InitialConditionContainer.((mesh,), (dof,), ics)
            ic_funcs = map(x -> InitialConditionFunction(x.func), ics)
        end
        return InitialConditions(ic_caches, ic_funcs)
    end

    # juliac safe, all funcs must have same type
    function InitialConditions{F}(mesh, dof, ics) where F <: Function
        ic_funcs = InitialConditionFunction{F}[]
        if length(ics) == 0
            IV = Vector{Int}
            RV = Vector{Float64}
            ic_caches = InitialConditionContainer{IV, RV}[]
        else
            ic_caches = InitialConditionContainer.((mesh,), (dof,), ics)
            for ic in ics
                push!(ic_funcs, InitialConditionFunction{F}(ic.func))
            end
            IV = typeof(ic_caches[1].dofs)
            RV = typeof(ic_caches[1].vals)
        end
        new{typeof(ic_funcs), IV, RV}(ic_caches, ic_funcs)
    end
end

function Adapt.adapt_structure(to, ics::InitialConditions)
    # NOTE
    # below logic is needed due to improper
    # adapt mapping for an empty array in julia 1.10/1.11
    # where Vector{T}(undef, 0) gets mappend to Vector{Any}
    if length(ics.ic_caches) > 0
        ic_caches = map(x -> adapt(to, x), ics.ic_caches)
    else
        temp_int = adapt(to, zeros(Int, 0))
        temp_floats = adapt(to, zeros(Float64, 0))
        ic_caches = InitialConditionContainer{typeof(temp_int), typeof(temp_floats)}[]
    end

    return InitialConditions(
        ic_caches,
        ics.ic_funcs
    )
end

function Base.show(io::IO, ics::InitialConditions)
    for (n, (cache, func)) in enumerate(zip(ics.ic_caches, ics.ic_funcs))
        show(io, "IC_$n")
        show(io, cache)
        show(io, func)
    end
end

function update_field_ics!(U::AbstractField, ics::InitialConditions)
    for ic in ics.ic_caches
        _update_field_ics!(U, ic)
    end
    return nothing
end

function update_ic_values!(ics::InitialConditions, X::AbstractField)
    for n in axes(ics.ic_caches, 1)
        _update_ic_values!(ics.ic_caches[n], ics.ic_funcs[n], X)
    end
    return nothing
end
