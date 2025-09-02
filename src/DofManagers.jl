abstract type AbstractDofManager{
    IT <: Integer,
    IDs <: AbstractArray{IT, 1}
} end

struct DofManager{
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
function DofManager(var::AbstractFunction)
    dirichlet_dofs = zeros(Int, 0)
    unknown_dofs = 1:size(var.fspace.coords, 2) * length(names(var)) |> collect
    return DofManager(dirichlet_dofs, unknown_dofs, var)
end

_field_type(dof::DofManager) = eval(typeof(dof.var.fspace.coords).name.name)

Base.length(dof::DofManager) = length(dof.dirichlet_dofs) + length(dof.unknown_dofs)

function Base.size(dof::DofManager)
    l = length(dof)
    nf = length(names(dof.var))
    return (nf, l ÷ nf)
end

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

function create_field(dof::DofManager)
    backend = KA.get_backend(dof)
    field = KA.zeros(backend, Float64, size(dof))
    return _field_type(dof)(field)
end

function create_unknowns(dof::DofManager)
    backend = KA.get_backend(dof)
    return KA.zeros(backend, Float64, length(dof.unknown_dofs))
end

function function_space(dof::DofManager)
    return dof.var.fspace
end

function update_dofs!(dof::DofManager, dirichlet_dofs)
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

# COV_EXCL_START
KA.@kernel function _update_field_unknowns_kernel!(U::H1Field, dof::DofManager, Uu::T) where T <: AbstractArray{<:Number, 1}
    N = KA.@index(Global)
    @inbounds U[dof.unknown_dofs[N]] = Uu[N]
end
# COV_EXCL_STOP
  
function _update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T, backend::KA.Backend) where T <: AbstractArray{<:Number, 1}
    kernel! = _update_field_unknowns_kernel!(backend)
    kernel!(U, dof, Uu, ndrange = length(Uu))
    return nothing
end
  
# Need a seperate CPU method since CPU is basically busted in KA
function _update_field_unknowns!(U::H1Field, dof::DofManager, Uu::T, ::KA.CPU) where T <: AbstractArray{<:Number, 1}
    U[dof.unknown_dofs] .= Uu
    return nothing
end

function update_field_unknowns!(
    U::AbstractField, 
    dof::DofManager,
    Uu::V
) where V <: AbstractVector{<:Number}
    backend = KA.get_backend(dof)
    @assert KA.get_backend(U) == backend
    @assert KA.get_backend(Uu) == backend
    _update_field_unknowns!(U, dof, Uu, backend)
    return nothing
end
