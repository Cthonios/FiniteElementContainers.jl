abstract type AbstractInitialCondition{F} end
abstract type AbstractInitialConditionContainer end

struct InitialCondition{F} <: AbstractInitialCondition{F}
    var_name::String
    func::F
    block_name::String

    # function InitialCondition(var_name::String, func, block_name::String)
    #     new{typeof(func)}(block_name, func, var_name)
    # end
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
        bk = BCBookKeeping(mesh, dof, ic.var_name; block_name = ic.block_name)
        vals = zeros(length(bk.dofs))
        new{typeof(bk.dofs), typeof(vals)}(bk.dofs, bk.nodes, vals)
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
    IV      <: AbstractArray{<:Integer, 1},
    RV      <: AbstractArray{<:Number, 1},
    ICFuncs <: NamedTuple
}
    ic_caches::Vector{InitialConditionContainer{IV, RV}}
    ic_funcs::ICFuncs
end

function InitialConditions(mesh, dof, ics)

    if length(ics) == 0
        ic_caches = InitialConditionContainer{Vector{Int}, Vector{Float64}}[]
        ic_funcs = NamedTuple()
    else
        ic_caches = InitialConditionContainer.((mesh,), (dof,), ics)
        syms = map(x -> Symbol("initial_condition_$x"), 1:length(ics))
        ic_funcs = NamedTuple{tuple(syms...)}(map(x -> InitialConditionFunction(x.func), ics))
    end
    return InitialConditions(ic_caches, ic_funcs)
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
        adapt(to, ics.ic_funcs)
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
    for ic in values(ics.ic_caches)
        _update_field_ics!(U, ic)
    end
    return nothing
end

function update_ic_values!(ics::InitialConditions, X::AbstractField)
    for (ic, func) in zip(values(ics.ic_caches), values(ics.ic_funcs))
        _update_ic_values!(ic, func, X)
    end
    return nothing
end
