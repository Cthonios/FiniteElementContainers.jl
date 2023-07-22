function setup_element_coordinates!(
  el_coords::Vector{SMatrix{D, N, Ftype, L}},
  coords::Matrix{<:AbstractFloat}, block::Block{B}
) where {N, D, Ftype, L, B}
  for e in eachindex(el_coords)
    el_coords[e] = @views coords[:, block.conn[:, e]]
  end 
  return el_coords
end

function setup_quadrature_point_coordinates!(
  ξs::Matrix{SVector{D, Ftype}}, 
  el_coords::Base.ReinterpretArray{StaticArraysCore.SMatrix{D, N, Ftype, L}, 1, Ftype, Vector{Ftype}, false},
  re::ReferenceFE{Itype, N, D, Ftype, L}
) where {Itype, N, D, Ftype, L}

  for e in axes(ξs, 2)
    for q in axes(ξs, 1)
      ξs[q, e] = el_coords[e] * shape_function_values(re, q)
    end
  end
end

function setup_shape_function_values!(
  Ns::Matrix{SVector{N, Ftype}}, re::ReferenceFE{Itype, N, D, Ftype, L}
) where {Itype, N, D, Ftype, L}

  for e in axes(Ns, 2)
    for q in axes(Ns, 1)
      Ns[q, e] = shape_function_values(re, q)
    end
  end
end

function setup_shape_function_gradients_and_JxWs!(
  ∇N_Xs::Matrix{SMatrix{N, D, Ftype, L}}, JxWs::Matrix{Ftype},
  el_coords::Base.ReinterpretArray{StaticArraysCore.SMatrix{D, N, Ftype, L}, 1, Ftype, Vector{Ftype}, false},
  re::ReferenceFE{Itype, N, D, Ftype, L}
) where {Itype, N, D, Ftype, L}

  for e in axes(∇N_Xs, 2)
    for q in axes(∇N_Xs, 1)
      J = el_coords[e] * shape_function_gradients(re, q)
      J_inv = inv(J)
      ∇N_Xs[q, e] = (J_inv * shape_function_gradients(re, q)')'
      JxWs[q, e] = det(J) * quadrature_weight(re, q)
    end
  end
end

"""
"""
struct FunctionSpaceInterpolant{N, D, Ftype, L}
  ξ::SVector{D, Ftype}
  N::SVector{N, Ftype}
  ∇N_X::SMatrix{N, D, Ftype, L}
  JxW::Ftype
end

"""
"""
struct FunctionSpace{Itype, N, D, Ftype, L, Q}
  connectivity::Matrix{Itype}
  interpolants::StructArray{
    FunctionSpaceInterpolant{N, D, Ftype, L}, 2, 
    NamedTuple{
      (:ξ, :N, :∇N_X, :JxW), 
      Tuple{
        Matrix{SVector{D, Ftype}}, 
        Matrix{SVector{N, Ftype}}, 
        Matrix{SMatrix{N, D, Ftype, L}}, 
        Matrix{Ftype}
      }
    }, 
    Int64
  }
end

"""
"""
connectivity(f::FunctionSpace, e::Int) = f.connectivity[:, e]

"""
"""
function Base.getindex(f::FunctionSpace{Itype, N, D, Ftype, L, Q}, q::Int, e::Int) where {Itype, N, D, Ftype, L, Q}
  return LazyRow(f.interpolants, q + Q * (e - 1))
end
"""
"""
Base.length(f::FunctionSpace) = length(f.interpolants)
"""
"""
Base.size(f::FunctionSpace{Itype, N, D, Ftype, L, Q}) where {Itype, N, D, Ftype, L, Q} = Q, length(f) // Q |> Int

"""
"""
function Base.size(f::FunctionSpace{Itype, N, D, Ftype, L, Q}, i::Int) where {Itype, N, D, Ftype, L, Q}
  if i == 1
    Q
  elseif i == 2
    length(f) // Q |> Int
  else
    throw(BoundsError("Invalid index"))
  end
end

"""
"""
function FunctionSpace(
  coords::Matrix{<:AbstractFloat}, block::Block{B},
  re::ReferenceFE{Itype, N, D, Ftype, L}
) where {B, Itype, N, D, Ftype, L}

  # only temporary - don't store this
  # el_coords = Vector{SMatrix{D, N, Ftype, L}}(undef, block.num_elem)
  # setup_element_coordinates!(el_coords, coords, block)

  # el_coords = copy(reinterpret(SMatrix{D, N, Ftype, L}, vec(coords[:, block.conn])))
  el_coords = reinterpret(SMatrix{D, N, Ftype, L}, vec(coords[:, block.conn]))

  ξs = Matrix{SVector{D, Ftype}}(undef, length(re.interpolants), block.num_elem)
  Ns = Matrix{SVector{N, Ftype}}(undef, length(re.interpolants), block.num_elem)
  ∇N_Xs = Matrix{SMatrix{N, D, Ftype, L}}(undef, length(re.interpolants), block.num_elem)
  JxWs = Matrix{Ftype}(undef, length(re.interpolants), block.num_elem)

  setup_quadrature_point_coordinates!(ξs, el_coords, re)
  setup_shape_function_values!(Ns, re)
  setup_shape_function_gradients_and_JxWs!(∇N_Xs, JxWs, el_coords, re)

  interpolants = StructArray{FunctionSpaceInterpolant{N, D, Ftype, L}}((ξs, Ns, ∇N_Xs, JxWs))
  n_q_points = length(re.interpolants)

  return FunctionSpace{B, N, D, Ftype, L, n_q_points}(block.conn, interpolants)
end
