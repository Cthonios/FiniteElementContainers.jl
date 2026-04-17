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

    function DofManager{Condensed}(dirichlet_dofs, unknown_dofs, var) where Condensed
        new{Condensed, eltype(dirichlet_dofs), typeof(dirichlet_dofs), typeof(var)}(
            dirichlet_dofs, unknown_dofs, var
        )
    end
end

# method that initializes dof manager
# with all dofs unknown
"""
$(TYPEDSIGNATURES)
"""
function DofManager(var::AbstractFunction; use_condensed::Bool = false)
    return DofManager{use_condensed}(var)
end

function DofManager{Condensed}(var::AbstractFunction) where Condensed
    dirichlet_dofs = zeros(Int, 0)
    unknown_dofs = 1:size(var.fspace.coords, 2) * length(names(var)) |> collect
    # return DofManager{
    #     use_condensed,
    #     eltype(dirichlet_dofs),
    #     typeof(dirichlet_dofs),
    #     typeof(var)
    # }(dirichlet_dofs, unknown_dofs, var)
    return DofManager{Condensed}(dirichlet_dofs, unknown_dofs, var)
end

# function Adapt.adapt_structure(to, dof::DofManager{C, IT, IDs, Var}) where {C, IT, IDs, Var}
    # dirichlet_dofs = adapt(to, dof.dirichlet_dofs)
    # unknowns = adapt(to, dof.unknown_dofs)
    # var = adapt(to, dof.var)
    # return DofManager{
    #   C, IT, typeof(dirichlet_dofs), typeof(var) 
    # }(dirichlet_dofs, unknowns, var)
function Adapt.adapt_structure(to, dof::DofManager)
    return DofManager{_is_condensed(dof)}(
        adapt(to, dof.dirichlet_dofs),
        adapt(to, dof.unknown_dofs),
        adapt(to, dof.var)
    )
  end

# _field_type(dof::DofManager) = eval(typeof(dof.var.fspace.coords).name.name)
function _field_type(dof::DofManager)
    coords = dof.var.fspace.coords
    if isa(coords, H1Field)
        return H1Field
    else
        @assert false "Finish me"
    end
end

_is_condensed(::DofManager{C, IT, IDs, V}) where {C, IT, IDs, V} = C

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
function create_field(dof::DofManager, el_type::Type{RT} = Float64) where RT <: Number
    return _create_field(KA.get_backend(dof), el_type, dof)
end

function _create_field(backend::KA.CPU, ::Type{RT}, dof::DofManager) where RT <: Number
    coords = dof.var.fspace.coords
    data = _dense_array(backend, RT, length(dof))
    if isa(coords, H1Field)
        field = H1Field{RT, Vector{RT}, num_fields(dof.var)}(data)
    else
        @assert false "Finish me"
    end
    return field
end

function _create_field(backend::KA.Backend, ::Type{RT}, dof::DofManager) where RT <: Number
    data = KA.zeros(backend, RT, size(dof))
    return _field_type(dof)(data)
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
    return _create_unknowns(KA.get_backend(dof), dof)
end

function _create_unknowns(::KA.CPU, dof::DofManager{false, IT, IDs, Var}) where {IT, IDs, Var}
    return zeros(Float64, length(dof.unknown_dofs))
end

function _create_unknowns(backend::KA.Backend, dof::DofManager{false, IT, IDs, Var}) where {IT, IDs, Var}
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

function extract_field_unknowns!(
    Uu::V,
    unknown_dofs::I,
    U::AbstractField
) where {V <: AbstractVector{<:Number}, I <: AbstractVector{<:Integer}}
    fec_foreach(Uu) do n
        Uu[n] = U[unknown_dofs[n]]
    end
    return nothing
end

function extract_field_unknowns!(
    Uu::V,
    dof::DofManager,
    U::AbstractField
) where V <: AbstractVector{<:Number}
    extract_field_unknowns!(Uu, dof.unknown_dofs, U)
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

################################################

# Instead of a closure, define an explicit functor
# struct UpdateFieldFunctorConsts{U <: AbstractArray{<:Integer}}
#     unknown_dofs::U
# end

# struct UpdateFieldFunctor{
#     F <: AbstractField,
#     U <: UpdateFieldFunctorConsts,
#     V
# }
#     U::F
#     # unknown_dofs::I
#     unknown_dofs::U
#     Uu::V
#     flag::Bool
# end

# # The call method is a plain function - no hidden captures
# function (f::UpdateFieldFunctor)(n)
#     @inbounds dof = f.unknown_dofs.unknown_dofs[n]
#     if f.flag
#         @inbounds f.U.data[dof] = f.Uu[dof]
#     else
#         @inbounds f.U.data[dof] = f.Uu[n]
#     end
#     return nothing
# end

# ##################################################

function _update_field_unknowns!(
    U::AbstractField,
    unknown_dofs::I,
    Uu::V,
    flag::Bool
) where {I <: AbstractVector{<:Integer}, V <: AbstractVector{<:Number}}
    fec_foreach(unknown_dofs) do n
        @inbounds dof = unknown_dofs[n]
        if flag
            @inbounds U.data[dof] = Uu[dof]
        else
            @inbounds U.data[dof] = Uu[n]
        end
    end
    # kernel_consts = UpdateFieldFunctorConsts(unknown_dofs)
    # kernel = UpdateFieldFunctor(U, kernel_consts, Uu, flag)
    # fec_foreach(kernel, unknown_dofs)
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
    Uu::V,
    enzyme_safe::Bool = false
) where V <: AbstractVector{<:Number}
    backend = KA.get_backend(dof)
    @assert KA.get_backend(U) == backend
    @assert KA.get_backend(Uu) == backend

    if _is_condensed(dof)
        @assert length(Uu) == length(U)
    else
        @assert length(Uu) == length(dof.unknown_dofs)
    end

    if enzyme_safe
        _update_field_unknowns_enzyme_safe!(U, dof, Uu, KA.get_backend(U))
    else
        # _update_field_unknowns!(U, dof, Uu, backend)
        _update_field_unknowns!(U, dof.unknown_dofs, Uu, _is_condensed(dof))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)
Takes in a field and updates the field

I think this one can be removed/deprecated
"""
function update_field_unknowns!(
    U::F, dof::DofManager, Uu::F,
    enzyme_safe::Bool = false
) where F <: AbstractField
    update_field_unknowns!(U, dof, Uu.data, enzyme_safe)
end
