function setup_quadrature_point_coordinates!(
  ξs::Matrix{T1},
  el_coords::T2, re::ReferenceFE
) where {T1 <: AbstractArray, T2 <: AbstractArray}

  for e in axes(ξs, 2)
    for q in axes(ξs, 1)
      @inbounds @fastmath ξs[q, e] = el_coords[e] * shape_function_values(re, q)
    end
  end
end

function setup_shape_function_values!(
  Ns::Matrix{T},
  re::ReferenceFE
) where T <: AbstractArray

  for e in axes(Ns, 2)
    for q in axes(Ns, 1)
      @inbounds @fastmath Ns[q, e] = shape_function_values(re, q)
    end
  end
end

function setup_shape_function_gradients_and_JxWs!(
  ∇N_Xs::Matrix{T1}, JxWs::Matrix{Rtype},
  el_coords::T2, re::ReferenceFE
) where {T1 <: AbstractArray, T2 <: AbstractArray, Rtype}
  @inbounds @fastmath begin
    for e in axes(∇N_Xs, 2)
      for q in axes(∇N_Xs, 1)
        # J = el_coords[e] * shape_function_gradients(re, q)
        J = (shape_function_gradients(re, q)' * el_coords[e]')'
        J_inv = inv(J)
        ∇N_Xs[q, e] = (J_inv * shape_function_gradients(re, q)')'
        # ∇N_Xs[q, e] = shape_function_gradients(re, q) * J_inv'
        # ∇N_Xs[q, e] = J_inv * shape_function_gradients(re, q)
        JxWs[q, e] = det(J) * quadrature_weight(re, q)
      end
    end
  end
end

"""
"""
struct FunctionSpaceInterpolant{N, D, Rtype, L}
  ξ::SVector{D, Rtype}
  N::SMatrix{N, 1, Rtype, N}
  ∇N_X::SMatrix{N, D, Rtype, L}
  # ∇N_X::SMatrix{D, N, Rtype, L}
  JxW::Rtype
end

struct FunctionSpace{Itype, N, D, Rtype, L} 
  fspace::StructArray{
    FunctionSpaceInterpolant{N, D, Rtype, L}, 2, 
    NamedTuple{(:ξ, :N, :∇N_X, :JxW), 
      Tuple{
        Matrix{SVector{D, Rtype}}, 
        Matrix{SVector{N, Rtype}}, 
        Matrix{SMatrix{N, D, Rtype, L}}, 
        # Matrix{SMatrix{D, N, Rtype, L}},
        Matrix{Rtype}
      }
    }, Int64
  }
  conn::Matrix{Itype}
end
Base.axes(f::FunctionSpace, i::Int) = Base.OneTo(size(f, i))
Base.getindex(f::FunctionSpace, q::Int, e::Int) = f.fspace[q, e]
Base.size(f::FunctionSpace) = size(f.fspace)
Base.size(f::FunctionSpace, i::Int) = size(f.fspace, i)

"""
"""
function FunctionSpace(
  coords::Matrix{<:AbstractFloat}, block::Block{I, B},
  re::ReferenceFE{Itype, N, D, Rtype, L1, L2}
) where {I, B, Itype, N, D, Rtype, L1, L2}

  el_coords = @views reinterpret(SMatrix{D, N, Rtype, L1}, vec(coords[:, block.conn]))

  ξs = Matrix{SVector{D, Rtype}}(undef, length(re.interpolants), block.num_elem)
  Ns = Matrix{SVector{N, Rtype}}(undef, length(re.interpolants), block.num_elem)
  ∇N_Xs = Matrix{SMatrix{N, D, Rtype, L1}}(undef, length(re.interpolants), block.num_elem)
  # ∇N_Xs = Matrix{SMatrix{D, N, Rtype, L}}(undef, length(re.interpolants), block.num_elem)
  JxWs = Matrix{Rtype}(undef, length(re.interpolants), block.num_elem)

  setup_quadrature_point_coordinates!(ξs, el_coords, re)
  setup_shape_function_values!(Ns, re)
  setup_shape_function_gradients_and_JxWs!(∇N_Xs, JxWs, el_coords, re)

  fspace = StructArray{FunctionSpaceInterpolant{N, D, Rtype, L1}}((ξs, Ns, ∇N_Xs, JxWs))
  return FunctionSpace{Itype, N, D, Rtype, L1}(fspace, block.conn)
end
