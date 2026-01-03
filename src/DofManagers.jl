"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractDofManager{
    IT <: Integer,
    IDs <: AbstractArray{IT, 1}
} end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
struct DofManager{
    Condensed, # boolean flag for whether or not to seperate data between unknowns and constrained dofs
    # when creating unknowns
    IT, 
    IDs <: AbstractArray{IT, 1},
    Var <: AbstractFunction
} <: AbstractDofManager{IT, IDs}
    dirichlet_dofs::IDs
    unknown_dofs::IDs
    var::Var
end

# method that initializes dof manager
# with all dofs unknown
"""
$(TYPEDSIGNATURES)
"""
function DofManager(var::AbstractFunction; use_condensed::Bool = false)
    dirichlet_dofs = zeros(Int, 0)
    unknown_dofs = 1:size(var.fspace.coords, 2) * length(names(var)) |> collect
    return DofManager{
        use_condensed,
        eltype(dirichlet_dofs),
        typeof(dirichlet_dofs),
        typeof(var)
    }(dirichlet_dofs, unknown_dofs, var)
end

function Adapt.adapt_structure(to, dof::DofManager{C, IT, IDs, Var}) where {C, IT, IDs, Var}
    dirichlet_dofs = adapt(to, dof.dirichlet_dofs)
    unknowns = adapt(to, dof.unknown_dofs)
    var = adapt(to, dof.var)
    return DofManager{
      C, IT, typeof(dirichlet_dofs), typeof(var) 
    }(dirichlet_dofs, unknowns, var)
  end

_field_type(dof::DofManager) = eval(typeof(dof.var.fspace.coords).name.name)
_is_condensed(dof::DofManager{C, IT, IDs, V}) where {C, IT, IDs, V} = C

"""
$(TYPEDSIGNATURES)
Return the total lenth of dofs in the problem.

E.g. for an H1 space this will the number of nodes times
the number of degrees of freedom per node.
"""
Base.length(dof::DofManager) = length(dof.dirichlet_dofs) + length(dof.unknown_dofs)

"""
$(TYPEDSIGNATURES)
This returns ```(n_dofs, n_entities)```
where ```n_dofs``` is the number of dofs per entity (e.g. node)
and ```n_entities``` is the number of entitities (e.g. node).
"""
function Base.size(dof::DofManager)
    l = length(dof)
    nf = length(names(dof.var))
    return (nf, l ÷ nf)
end

"""
$(TYPEDSIGNATURES)
```size(dof, 1)``` returns the number of dofs
per entity, and ```size(dof, 2)``` returns the 
number of entities.
"""
function Base.size(dof::DofManager, i::Int)
    nf = length(names(dof.var))
    if i == 1
        return nf
    elseif i == 2
        return length(dof) ÷ nf
    else
        @assert false
    end
end

Base.show(io::IO, dof::DofManager) = 
print(io, "DofManager\n", 
          "  Number of entities        = $(size(dof, 2))\n",
          "  Number of dofs per entity = $(size(dof, 1))\n",
          "  Number of total dofs      = $(length(dof))\n",
          "  Storage type              = $(typeof(dof.unknown_dofs))")

KA.get_backend(dof::DofManager) = KA.get_backend(dof.dirichlet_dofs)

"""
$(TYPEDSIGNATURES)
Creates a field where ```typeof(field) <: AbstractField```
based on the variable ```dof``` was created with
"""
function create_field(dof::DofManager)
    backend = KA.get_backend(dof)
    field = KA.zeros(backend, Float64, size(dof))
    return _field_type(dof)(field)
end

"""
$(TYPEDSIGNATURES)
Creates a vector of unknown dofs.

This specific method returns a vector equal in length to the
length of the internally stored list of unknown dofs in ```dof```.

This is used for solution techniques when vector/matrix rows are
removed where dofs are fixed.
"""
function create_unknowns(dof::DofManager{false, IT, IDs, Var}) where {IT, IDs, Var}
    backend = KA.get_backend(dof)
    return KA.zeros(backend, Float64, length(dof.unknown_dofs))
end

"""
$(TYPEDSIGNATURES)
Creates a vector of unknown dofs.

This specific method returns a vector equal in length to the
length of a field created by ```dof```. E.g. all dofs are unknown.

This is used for solution techniques when vector/matrix rows are not 
removed where dofs are fixed.
"""
function create_unknowns(dof::DofManager{true, IT, IDs, Var}) where {IT, IDs, Var}
    backend = KA.get_backend(dof)
    return KA.zeros(backend, Float64, length(dof))
end

# COV_EXCL_START
KA.@kernel function _extract_field_unknowns_kernel!(
    Uu::V, 
    # dof::DofManager{false, IT, IDs, Var}, 
    unknown_dofs::IDs,
    U::AbstractField
) where {V <: AbstractVector{<:Number}, IDs}
    N = KA.@index(Global)
    @inbounds Uu[N] = U[unknown_dofs[N]]
end
# COV_EXCL_STOP

function _extract_field_unknowns!(
    Uu::V, 
    dof::DofManager{false, IT, IDs, Var}, 
    U::AbstractField, 
    backend::KA.Backend
) where {V <: AbstractVector{<:Number}, IT, IDs, Var}
    kernel! = _extract_field_unknowns_kernel!(backend)
    kernel!(Uu, dof.unknown_dofs, U, ndrange = length(Uu))
    return nothing
end

function _extract_field_unknowns!(
    Uu::V, 
    dof::DofManager{false, IT, IDs, Var}, 
    U::AbstractField, 
    ::KA.CPU
) where {V <: AbstractVector{<:Number}, IT, IDs, Var}
    @views Uu .= U[dof.unknown_dofs]
    return nothing
end

"""
$(TYPEDSIGNATURES)
Updates the entries of ```Uu``` with the unknown dofs
in the field ```U```.
"""
function extract_field_unknowns!(
    Uu::V,
    dof::DofManager{false, IT, IDs, Var},
    U::AbstractField
) where {V <: AbstractVector{<:Number}, IT, IDs, Var}
    backend = KA.get_backend(dof)
    @assert KA.get_backend(U) == backend
    @assert KA.get_backend(Uu) == backend
    _extract_field_unknowns!(Uu, dof, U, backend)
    return nothing
end

function function_space(dof::DofManager)
    return dof.var.fspace
end

"""
$(TYPEDSIGNATURES)
Takes in a list of dof ids associated with dirichlet bcs
and updates the internals of ```dof``` to reflect these.

NOTE: This clears all existing bcs in ```dof``` and
starts fresh.
"""
function update_dofs!(dof::DofManager, dirichlet_dofs::V) where V <: AbstractArray{<:Integer, 1}
    ND, NI = size(dof)
    Base.resize!(dof.dirichlet_dofs, length(dirichlet_dofs))
    Base.resize!(dof.unknown_dofs, ND * NI)

    for i in axes(dirichlet_dofs, 1)
        d = dirichlet_dofs[i]
        @assert d >= 1 && d <= ND * NI
    end

    dof.dirichlet_dofs .= dirichlet_dofs
    dof.unknown_dofs .= 1:ND * NI
    deleteat!(dof.unknown_dofs, dof.dirichlet_dofs)
    @assert length(dof.dirichlet_dofs) + length(dof.unknown_dofs) == ND * NI
    return nothing
end

# # COV_EXCL_START
# KA.@kernel function _update_field_unknowns_kernel!(
#     U::AbstractField, 
#     dof::DofManager{false, IT, IDs, Var}, 
#     Uu::V
# ) where {V <: AbstractVector{<:Number}, IT, IDs, Var}
#     N = KA.@index(Global)
#     @inbounds U.data[dof.unknown_dofs[N]] = Uu[N]
# end
# # COV_EXCL_STOP

# COV_EXCL_START
KA.@kernel function _update_field_unknowns_kernel!(
    U::AbstractField, 
    # dof::DofManager{true, IT, IDs, Var}, 
    unknown_dofs::IDs,
    Uu::V,
    flag::Bool
) where {V <: AbstractVector{<:Number}, IDs}
    N = KA.@index(Global)
    if flag
        @inbounds U.data[unknown_dofs[N]] = Uu[unknown_dofs[N]]
    else
        @inbounds U.data[unknown_dofs[N]] = Uu[N]
    end
end
# COV_EXCL_STOP
  
function _update_field_unknowns!(
    U::AbstractField, 
    dof::DofManager{flag, IT, IDs, Var}, 
    Uu::T, 
    backend::KA.Backend
) where {
    T <: AbstractVector{<:Number},
    flag, IT, IDs, Var
}
    kernel! = _update_field_unknowns_kernel!(backend)
    kernel!(U, dof.unknown_dofs, Uu, flag, ndrange = length(dof.unknown_dofs))
    return nothing
end
  
# Need a seperate CPU method since CPU is basically busted in KA
function _update_field_unknowns!(
    U::AbstractField, 
    dof::DofManager{false, IT, IDs, Var}, 
    Uu::T, 
    ::KA.CPU
) where {T <: AbstractVector{<:Number}, IT, IDs, Var}
    U[dof.unknown_dofs] .= Uu
    return nothing
end

function _update_field_unknowns!(
    U::AbstractField, 
    dof::DofManager{true, IT, IDs, Var}, 
    Uu::T, 
    ::KA.CPU
) where {T <: AbstractVector{<:Number}, IT, IDs, Var}
    @views U[dof.unknown_dofs] .= Uu[dof.unknown_dofs]
    return nothing
end

"""
$(TYPEDSIGNATURES)
Updates the unknowns of the field ```U``` based on 
the values of ```Uu```.
"""
function update_field_unknowns!(
    U::AbstractField, 
    dof::DofManager,
    Uu::V
) where V <: AbstractVector{<:Number}
    backend = KA.get_backend(dof)
    @assert KA.get_backend(U) == backend
    @assert KA.get_backend(Uu) == backend

    if _is_condensed(dof)
        @assert length(Uu) == length(U)
    else
        @assert length(Uu) == length(dof.unknown_dofs)
    end

    _update_field_unknowns!(U, dof, Uu, backend)
    return nothing
end

"""
$(TYPEDSIGNATURES)
Takes in a field and updates the field

I think this one can be removed/deprecated
"""
function update_field_unknowns!(
    U::F, dof::DofManager, Uu::F
) where F <: AbstractField
    update_field_unknowns!(U, dof, Uu.data)
end
