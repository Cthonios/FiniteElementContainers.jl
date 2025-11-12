abstract type AbstractInitialCondition{F} end
abstract type AbstractInitialConditionContainer end

struct InitialCondition{F} <: AbstractInitialCondition{F}
    block_name::Symbol
    func::F
    var_name::Symbol
end

function InitialCondition(var_name::Symbol, block_name::Symbol, func::Function)
    return InitialCondition(block_name, func, var_name)
end

function InitialCondition(var_name::String, block_name::String, func::Function)
    return InitialCondition(Symbol(var_name), Symbol(block_name), func)
end

struct InitialConditionContainer{
    IT <: Integer,
    RT <: Number,
    IV <: AbstractArray{IT, 1},
    RV <: AbstractArray{RT, 1}
} <: AbstractInitialConditionContainer
    dofs::IV
    locations::IV
    vals::RV
end

# NOTE hardcoded to H1Field behavior the way
# TODO maybe we should initialize things off of fspaces?
# Or actually, use the information from the var in dof
# to "do the right thing" depending upon the field type
# this is set up
function InitialConditionContainer(mesh, dof, ic::InitialCondition)
    bk = BCBookKeeping(mesh, dof, ic.var_name; block_name=ic.block_name)
    vals = zeros(length(bk.dofs))
    return InitialConditionContainer(bk.dofs, bk.nodes, vals)
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

function _update_ic_values!(ic::InitialConditionContainer, func, X, ::KA.CPU)
    ND = num_fields(X)
    for (n, location) in enumerate(ic.locations)
        X_temp = @views SVector{ND, eltype(X)}(X[:, location])
        ic.vals[n] = func.func(X_temp)
    end
end

# COV_EXCL_START
KA.@kernel function _update_ic_values_kernel!(bc::InitialConditionContainer, func, X)
    I = KA.@index(Global)
    ND = num_fields(X)
    loc = ic.locations[I]
  
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
    bc.vals[I] = func(X_temp)
end
# COV_EXCL_STOP

function _update_ic_values!(ic::InitialConditionContainer, func, X, backend::KA.Backend)
    kernel! = _update_ic_values_kernel!(backend)
    kernel!(ic, func, X, ndrange = length(bc))
    return nothing
end

function update_ic_values!(ics, funcs, X)
    for (ic, func) in zip(values(ics), values(funcs))
        backend = KA.get_backend(ic)
        _update_ic_values!(ic, func, X, backend)
    end
    return nothing
end

function _update_field_ics!(U, ic::InitialConditionContainer, ::KA.CPU)
    for (dof, val) in zip(ic.dofs, ic.vals)
        U[dof] = val
    end
    return nothing
end

# COV_EXCL_START
KA.@kernel function _update_field_ics_kernel!(U, ic::InitialConditionContainer)
    I = KA.@index(Global)
    dof = ic.dofs[I]
    val = ic.vals[I]
    U[dof] = val
end
# COV_EXCL_STOP

function _update_field_ics!(U, ic::InitialConditionContainer, backend::KA.Backend)
    kernel! = _update_field_ics_kernel!(backend)
    kernel!(U, ic, ndrange = length(ic))
    return nothing
end

function update_field_ics!(U, ics::NamedTuple)
    for ic in values(ics)
        _update_field_ics!(U, ic, KA.get_backend(U))
    end
    return nothing
end

struct InitialConditionFunction{F}
    func::F
end

struct InitialConditions{ICCaches, ICFuncs}
    ic_caches::ICCaches
    ic_funcs::ICFuncs
end

function InitialConditions(mesh, dof, ics)

    if length(ics) == 0
        return InitialConditions(NamedTuple(), NamedTuple())
    end

    syms = map(x -> Symbol("initial_condition_$x"), 1:length(ics))
    ic_funcs = NamedTuple{tuple(syms...)}(map(x -> InitialConditionFunction(x.func), ics))
    ic_containers = InitialConditionContainer.((mesh,), (dof,), ics)

    ic_containers = NamedTuple{tuple(syms...)}(tuple(ic_containers...))
    return InitialConditions(ic_containers, ic_funcs)
end

function Adapt.adapt_structure(to, ics::InitialConditions)
    return InitialConditions(
        adapt(to, ics.ic_caches), 
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

function update_field_ics!(U, ics::InitialConditions)
    update_field_ics!(U, ics.ic_caches)
    return nothing
end

function update_ic_values!(ics, X)
    update_ic_values!(ics.ic_caches, ics.ic_funcs, X)
    return nothing
end
