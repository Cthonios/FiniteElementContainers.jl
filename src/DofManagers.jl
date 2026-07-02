const CONSTRAINED_DOF     = -1
const DIRICHLET_DOF       = -1
const PERIODIC_SIDE_B_DOF = -2

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractDofManager{
    IT  <: Integer,
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
    dof_to_unknown::IDs
    periodic_side_a_dofs::IDs
    periodic_side_b_dofs::IDs
    periodic_side_b_to_side_a_unknown::IDs
    unknown_dofs::IDs
    var::Var

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
        dof_to_unknown = 1:num_entities(var.fspace) * length(names(var)) |> collect
        periodic_side_a_dofs = zeros(Int, 0)
        periodic_side_b_dofs = zeros(Int, 0)
        periodic_side_b_to_side_a_unknown = zeros(Int, 0)
        unknown_dofs = 1:num_entities(var.fspace) * length(names(var)) |> collect
        return DofManager{Condensed}(
            dirichlet_dofs, dof_to_unknown,
            periodic_side_a_dofs, periodic_side_b_dofs,
            periodic_side_b_to_side_a_unknown, 
            unknown_dofs, var
        )
    end

    function DofManager{Condensed}(
        dirichlet_dofs::IDs, dof_to_unknown,
        periodic_side_a_dofs, periodic_side_b_dofs,
        periodic_side_b_to_side_a_unknown, 
        unknown_dofs, var::V
    ) where {Condensed, IDs, V}
        new{Condensed, eltype(dirichlet_dofs), IDs, V}(
            dirichlet_dofs, dof_to_unknown,
            periodic_side_a_dofs, periodic_side_b_dofs,
            periodic_side_b_to_side_a_unknown, 
            unknown_dofs, var
        )
    end
end

function Adapt.adapt_structure(to, dof::DofManager)
    return DofManager{_is_condensed(dof)}(
        adapt(to, dof.dirichlet_dofs),
        adapt(to, dof.dof_to_unknown),
        adapt(to, dof.periodic_side_a_dofs),
        adapt(to, dof.periodic_side_b_dofs),
        adapt(to, dof.periodic_side_b_to_side_a_unknown),
        adapt(to, dof.unknown_dofs),
        adapt(to, dof.var)
    )
end

_ids_type(::DofManager{C, IT, IDs, V}) where {C, IT, IDs, V} = IDs
_is_condensed(::DofManager{C, IT, IDs, V}) where {C, IT, IDs, V} = C

"""
$(TYPEDSIGNATURES)
Return the total lenth of dofs in the problem.

E.g. for an H1 space this will the number of nodes times
the number of degrees of freedom per node.
"""
function Base.length(dof::DofManager) 
    cache = dof
    return length(cache.dirichlet_dofs) + length(cache.periodic_side_b_dofs) + length(cache.unknown_dofs)
end

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

Base.show(io::IO, dof::DofManager; pad::String = "") = 
print(io, "$(pad)DofManager\n", 
          "$(pad)  Number of entities        = $(size(dof, 2))\n",
          "$(pad)  Number of dofs per entity = $(size(dof, 1))\n",
          "$(pad)  Number of total dofs      = $(length(dof))\n",
          "$(pad)  Number of dirichlet dofs  = $(length(dof.dirichlet_dofs))\n",
          "$(pad)  Number of periodic dofs   = $(length(dof.periodic_side_b_dofs))\n",
          "$(pad)  Number of unknown dofs    = $(length(dof.unknown_dofs))\n",
          "$(pad)  Storage type              = $(typeof(dof.unknown_dofs))")

KA.get_backend(dof::DofManager) = KA.get_backend(dof.dirichlet_dofs)

"""
$(TYPEDSIGNATURES)
Creates a field where ```typeof(field) <: AbstractField```
based on the variable ```dof``` was created with
"""
function create_field(dof::DofManager, float_type::Type{RT} = Float64) where RT <: Number
    backend = KA.get_backend(dof)
    data = fec_dense_array(backend, float_type, length(dof))
    var = dof.var
    fspace = var.fspace
    if _field_type(fspace) == H1Field
        field = H1Field{RT, typeof(data), num_fields(var)}(data)
    elseif _field_type(fspace) == HcurlField
        field = HcurlField{RT, typeof(data), num_fields(var)}(data)
    elseif _field_type(fspace) == HdivField
        field = HdivField{RT, typeof(data), num_fields(var)}(data)
    elseif _field_type(fspace) == L2Field
        field = L2Field{RT, typeof(data), num_fields(var)}(undef, block_quadrature_sizes(fspace))
        fill!(field, zero(RT))
    end
    return field
end

function create_field(dof::Tuple)
    return map(create_field, dof)
end

"""
$(TYPEDSIGNATURES)
Creates a vector of unknown dofs.

The backend of ``dof`` will be used as the backend for the generated array.

There is an optional argument ``float_type`` that defaults to ``Float64``.
"""
function create_unknowns(dof::DofManager{Flag, IT, IDs, Var}, float_type = Float64) where {Flag, IT, IDs, Var}
    backend = KA.get_backend(dof)
    if Flag
        return fec_dense_array(backend, float_type, length(dof))
    else
        return fec_dense_array(backend, float_type, length(dof.unknown_dofs))
    end
end

function create_unknowns(dof::Tuple)
    # data = BlockedArray{Float64}(undef, map(length, dof) |> collect)
    data = BlockedArray{Float64}(undef, map(x -> length(x.unknown_dofs), dof) |> collect)
    fill!(data, zero(eltype(data)))
    return data
end

"""
Map a single global DOF index to its reduced unknown index.
 
  - Returns `-1`  for Dirichlet DOFs         (caller should skip these).
  - Returns the side_a's unknown index       for periodic side_b DOFs.
  - Returns the DOF's own unknown index      for free/side_a DOFs.
 
Both `dof_to_unknown` and `periodic_side_b_to_side_a_unknown` are flat vectors indexed
by global DOF, so this is a pure array lookup with no allocation — safe to
call inside GPU kernels.
"""
@inline function dof_to_unknown_index(dof::DofManager, n::Int)
    dtu = dof.dof_to_unknown[n]
    if dtu == DIRICHLET_DOF
        return DIRICHLET_DOF
    elseif dtu == PERIODIC_SIDE_B_DOF
        side_a_unknown = dof.periodic_side_b_to_side_a_unknown[n]
        # @assert side_a_unknown != PERIODIC_SIDE_B_DOF
        # TODO need some kind of check here
        return side_a_unknown
    else
        return dtu
    end
end

function extract_field_unknowns!(
    Uu::V,
    dof::DofManager,
    U::AbstractField
) where V <: AbstractVector{<:Number}
    unknown_dofs = dof.unknown_dofs
    fec_foreach(Uu) do n
        Uu[n] = U[unknown_dofs[n]]
    end
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
function update_dofs!(
    dof::DofManager, dirichlet_dofs::V,
    periodic_side_a_dofs::V, periodic_side_b_dofs::V
) where V <: AbstractArray{<:Integer, 1}

    # Resolve chains and deduplicate corners first
    resolved_side_a, resolved_side_b = _resolve_periodic_chains(
        periodic_side_a_dofs, periodic_side_b_dofs
    )
    pairs           = unique(collect(zip(resolved_side_a, resolved_side_b)))
    resolved_side_a = first.(pairs)
    resolved_side_b = last.(pairs)

    ND, NI = size(dof)
    Base.resize!(dof.dirichlet_dofs,       length(dirichlet_dofs))
    Base.resize!(dof.periodic_side_a_dofs, length(resolved_side_a))  # resolved length
    Base.resize!(dof.periodic_side_b_dofs, length(resolved_side_b))  # resolved length
    Base.resize!(dof.unknown_dofs,         ND * NI)

    # bounds checks on resolved arrays
    for d in dirichlet_dofs
        @assert d >= 1 && d <= ND * NI
    end
    for (d_a, d_b) in zip(resolved_side_a, resolved_side_b)
        @assert d_a >= 1 && d_a <= ND * NI
        @assert d_b >= 1 && d_b <= ND * NI
    end

    # store resolved arrays
    dof.dirichlet_dofs       .= dirichlet_dofs
    dof.periodic_side_a_dofs .= resolved_side_a
    dof.periodic_side_b_dofs .= resolved_side_b

    # build unknown_dofs from the resolved side_b set
    sorted_periodic_b = sort(resolved_side_b)
    dof.unknown_dofs .= 1:ND * NI
    deleteat!(dof.unknown_dofs, union(sort(dirichlet_dofs), sorted_periodic_b))
    @assert length(dof.dirichlet_dofs) + length(dof.periodic_side_b_dofs) + 
            length(dof.unknown_dofs) == ND * NI

    # dof_to_unknown: free/side_a DOF -> reduced index, 0 otherwise
    dof_to_unknown = zeros(Int, ND * NI)
    for (k, d) in enumerate(dof.unknown_dofs)
        dof_to_unknown[d] = k
    end

    # add dirichlet dofs
    for d in dof.dirichlet_dofs
        dof_to_unknown[d] = DIRICHLET_DOF
    end

    # add side b dofs
    for d in dof.periodic_side_b_dofs
        dof_to_unknown[d] = PERIODIC_SIDE_B_DOF
    end

    resize!(dof.dof_to_unknown, ND * NI)
    dof.dof_to_unknown .= dof_to_unknown

    # periodic_side_b_to_side_a_unknown: side_b DOF -> side_a reduced index, PERIODIC_SIDE_B_DOF otherwise
    periodic_side_b_to_side_a_unknown = zeros(Int, ND * NI)
    for (side_a_dof, side_b_dof) in zip(resolved_side_a, resolved_side_b)
        side_a_unknown = dof_to_unknown[side_a_dof]
        @assert side_a_unknown != 0 "Side A DOF $side_a_dof mapped to 0 — chain resolution incomplete"
        periodic_side_b_to_side_a_unknown[side_b_dof] = side_a_unknown
    end

    resize!(dof.periodic_side_b_to_side_a_unknown, ND * NI)
    dof.periodic_side_b_to_side_a_unknown .= periodic_side_b_to_side_a_unknown

    return nothing
end

function _resolve_periodic_chains(
    side_a_dofs::AbstractVector{<:Integer},
    side_b_dofs::AbstractVector{<:Integer},
)
    # Build initial side_b -> side_a map
    side_b_to_side_a = Dict{Int,Int}()
    for (a, b) in zip(side_a_dofs, side_b_dofs)
        side_b_to_side_a[b] = a
    end

    # Resolve chains: if a side_a is itself a side_b, follow the chain
    # until we reach a DOF that is not a side_b anywhere
    function canonical(d)
        while haskey(side_b_to_side_a, d)
            d = side_b_to_side_a[d]
        end
        return d
    end

    for b in keys(side_b_to_side_a)
        side_b_to_side_a[b] = canonical(side_b_to_side_a[b])
    end

    resolved_a = [side_b_to_side_a[b] for b in side_b_dofs]
    return resolved_a, side_b_dofs
end

# COV_EXCL_START
KA.@kernel function _update_field_unknowns_kernel_1!(
    U::AbstractField, 
    unknown_dofs::IDs,
    Uu::V
) where {V <: AbstractVector{<:Number}, IDs}
    N = KA.@index(Global)
    @inbounds U.data[unknown_dofs[N]] = Uu[unknown_dofs[N]]
end
# COV_EXCL_STOP
  
# COV_EXCL_START
KA.@kernel function _update_field_unknowns_kernel_2!(
    U::AbstractField, 
    unknown_dofs::IDs,
    Uu::V
) where {V <: AbstractVector{<:Number}, IDs}
    N = KA.@index(Global)
    @inbounds U.data[unknown_dofs[N]] = Uu[N]
end
# COV_EXCL_STOP

function _update_field_unknowns!(
    backend::KA.Backend,
    U::AbstractField, 
    dof::DofManager{flag, IT, IDs, Var}, 
    Uu::T
) where {
    T <: AbstractVector{<:Number},
    flag, IT, IDs, Var
}
    if _is_condensed(dof)
        kernel! = _update_field_unknowns_kernel_1!(backend)
    else
        kernel! = _update_field_unknowns_kernel_2!(backend)
    end
    kernel!(U, dof.unknown_dofs, Uu, ndrange = length(dof.unknown_dofs))
    return nothing
end
  
# Need a seperate CPU method since CPU is basically busted in KA
function _update_field_unknowns!(
    ::KA.CPU,
    U::AbstractField, 
    dof::DofManager{false, IT, IDs, Var}, 
    Uu::T
) where {T <: AbstractVector{<:Number}, IT, IDs, Var}
    U[dof.unknown_dofs] .= Uu
    return nothing
end

function _update_field_unknowns!(
    ::KA.CPU,
    U::AbstractField, 
    dof::DofManager{true, IT, IDs, Var}, 
    Uu::T, 
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
    # enzyme_safe::Bool = false
) where V <: AbstractVector{<:Number}
    backend = KA.get_backend(dof)
    @assert KA.get_backend(U) == backend
    @assert KA.get_backend(Uu) == backend

    if _is_condensed(dof)
        @assert length(Uu) == length(U) "Unknown size $(length(Uu)), field size $(size(U))"
    else
        @assert length(Uu) == length(dof.unknown_dofs) "Unknown size $(length(Uu)), field size $(size(U))"
    end

    _update_field_unknowns!(KA.get_backend(U), U, dof, Uu)
    return nothing
end

"""
$(TYPEDSIGNATURES)
Takes in a field and updates the field
"""
function update_field_unknowns!(
U::F, dof::DofManager, Uu::F
) where F <: AbstractField
    update_field_unknowns!(U, dof, Uu.data)
end
