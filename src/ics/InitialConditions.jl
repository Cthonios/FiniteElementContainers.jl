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
    block_conns = getproperty(mesh.element_conns, ic.block_name)
    block_nodes = unique(sort(block_conns.data))
    dof_index = _dof_index_from_var_name(dof, ic.var_name)

    all_dofs = reshape(1:length(dof), size(dof))
    block_dofs = all_dofs[dof_index, block_nodes]
    vals = zeros(length(block_dofs))
    return InitialConditionContainer(block_dofs, block_nodes, vals)
end

function Base.length(ic::InitialConditionContainer)
    return length(ic.dofs)
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

function update_field_ics!(U, ics)
    for ic in values(ics)
        _update_field_ics!(U, ic, KA.get_backend(U))
    end
    return nothing
end

struct InitialConditionFunction{F}
    func::F
end

function create_ics(mesh, dof::DofManager, ics::Vector{<:InitialCondition})
    if length(ics) == 0
        return NamedTuple(), NamedTuple()
    end

    syms = map(x -> Symbol("initial_condition_$x"), 1:length(ics))
    ic_funcs = NamedTuple{tuple(syms...)}(map(x -> InitialConditionFunction(x.func), ics))
    ic_containers = InitialConditionContainer.((mesh,), (dof,), ics)

    ic_containers = NamedTuple{tuple(syms...)}(tuple(ic_containers...))
    return ic_containers, ic_funcs
end
