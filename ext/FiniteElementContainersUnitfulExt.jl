module FiniteElementContainersUnitfulExt

using FiniteElementContainers
using Tensors
using Unitful

# function Tensors.gradient(f::F, x::Tensor{NO, ND, T, NT}) where {F <: Function, NO, ND, T <: Quantity, NT}
function Tensors.gradient(f::F, v::V) where {F<:Function, ND, T<:Quantity, NT, V<:Union{Tensor{2, ND, T, NT}, Tensor{1, ND, T, NT}}}
    val = f(v)
    grad = Tensors.gradient(f, ustrip(v))
    ret_unit = unit(val) / unit(v)
    data = broadcast(Unitful.:*, grad.data, ret_unit)
    ret_unit = typeof(data[1])
    # return Tensor{NO, ND, typeof(data)[1], NT}(data)
    # typeof(grad)
    ret_type = typeof(grad)
    return Tensor{
        ret_type.parameters[1], ret_type.parameters[2], 
        ret_unit, ret_type.parameters[4]
    }(data)
end

# move to constitutive models or
# better yet, open a PR in Tensors.jl
function Unitful.:*(
    tensor::Tensor{NO, ND, T, NT},
    unit::Unitful.Units
) where {NO, ND, T <: Number, NT}
    data = broadcast(Unitful.:*, tensor.data, unit)
    return Tensor{NO, ND, typeof(data[1]), NT}(data)
end

function Unitful.ustrip(
    tensor::Tensor{NO, ND, T, NT}
) where {NO, ND, T <: Quantity, NT}
    data = broadcast(ustrip, tensor.data)
    return Tensor{NO, ND, typeof(data[1]), NT}(data)
end

function Unitful.:*(
    field::H1Field{T, NF, V, S},
    unit::Unitful.Units
) where {
    T <: Number,
    NF,
    V <: AbstractArray{T, 1},
    S <: NamedTuple
}
    vals = Unitful.:*(field.vals, unit)
    return H1Field{eltype(vals), NF, typeof(vals), S}(vals)
end

function Unitful.unit(tensor::Tensor{NO, ND, T, NT}) where {NO, ND, T <: Quantity, NT}
    return unit(tensor.data[1])
end

function Unitful.ustrip(
    field::H1Field{T, NF, V, S}
) where {
    T <: Quantity,
    NF,
    V <: AbstractArray{T, 1},
    S <: NamedTuple
}
    vals = ustrip(field.vals)
    return H1Field{eltype(vals), NF, typeof(vals), S}(vals)
end

end # module
