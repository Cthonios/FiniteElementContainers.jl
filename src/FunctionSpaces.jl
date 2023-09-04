element_types = Dict{String, Type{<:ReferenceFiniteElements.ReferenceFEType}}(
  "HEX8"  => Hex8,
  "QUAD4" => Quad4,
  "QUAD9" => Quad9,
  "TET4"  => Tet4,
  "TET10" => Tet10,
  "TRI3"  => Tri3,
  "TRI6"  => Tri6
)

abstract type AbstractGradientConversion end
struct NoGradientConversion <: AbstractGradientConversion end
struct PlaneStrainGradientConversion <: AbstractGradientConversion end
voigt_dim(::Type{PlaneStrainGradientConversion}) = 3

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
        J = (shape_function_gradients(re, q)' * el_coords[e]')'
        J_inv = inv(J)
        ∇N_Xs[q, e] = (J_inv * shape_function_gradients(re, q)')'
        JxWs[q, e] = det(J) * quadrature_weight(re, q)
      end
    end
  end
end

function convert_shape_function_gradient(
  ::Type{PlaneStrainGradientConversion},
  ∇N_X::SMatrix{N, 2, Rtype, L},
  B_cache::Matrix{Rtype}
) where {N, Rtype, L}

  B_cache .= 0.0
  for n in 1:N
    B_cache[1, 1 + 2 * (n - 1)] = ∇N_X[n, 1]
    B_cache[2, 2 + 2 * (n - 1)] = ∇N_X[n, 2]
    B_cache[3, 1 + 2 * (n - 1)] = ∇N_X[n, 2]
    B_cache[3, 2 + 2 * (n - 1)] = ∇N_X[n, 1]
  end

  return SMatrix{3, 2 * N, Rtype, 3 * 2 * N}(B_cache)
end

function setup_converted_gradients!(
  Bs::Matrix{T1}, ∇N_Xs::Matrix{T2}, type::Type{Conv}
) where {T1 <: AbstractArray, T2 <: AbstractArray, Conv <: AbstractGradientConversion}

  N, D = size(∇N_Xs[1, 1])
  Rtype = eltype(∇N_Xs[1, 1])
  B_cache = Matrix{Rtype}(undef, voigt_dim(type), D * N)
  for e in axes(Bs, 2)
    for q in axes(Bs, 1)
      Bs[q, e] = convert_shape_function_gradient(type, ∇N_Xs[q, e], B_cache)
    end
  end
end

abstract type AbstractQuadratureValues end

struct QuadratureValues{N, D, Rtype, L1} <: AbstractQuadratureValues
  X::SVector{D, Rtype}
  N::SMatrix{N, 1, Rtype, N}
  ∇N_X::SMatrix{N, D, Rtype, L1}
  JxW::Rtype
end

struct QuadratureValuesWithSymmetricGradient{N, D, Rtype, L1, N_V, NxNDof, L2} <: AbstractQuadratureValues
  X::SVector{D, Rtype}
  N::SMatrix{N, 1, Rtype, N}
  ∇N_X::SMatrix{N, D, Rtype, L1}
  B::SMatrix{N_V, NxNDof, Rtype, L2}
  JxW::Rtype
end

function FunctionSpace(
  mesh::Mesh{Rtype, I, B},
  block::Block{I, B},
  re::ReferenceFE{Itype, N, D, Rtype, L1, L2};
  n_state_variables::Int = 0,
  shape_function_gradient_conversion::Type{Conv} = NoGradientConversion
) where {Rtype, I, B, Itype, N, D, L1, L2, Conv <: AbstractGradientConversion}

  # now make element level coords for calculating other things
  coords    = mesh.coords
  el_coords = @views reinterpret(SMatrix{D, N, Rtype, L1}, vec(coords[:, block.conn]))

  Xs    = Matrix{SVector{D, Rtype}}(undef, length(re.interpolants), block.num_elem)
  Ns    = Matrix{SVector{N, Rtype}}(undef, length(re.interpolants), block.num_elem)
  ∇N_Xs = Matrix{SMatrix{N, D, Rtype, L1}}(undef, length(re.interpolants), block.num_elem)
  JxWs  = Matrix{Rtype}(undef, length(re.interpolants), block.num_elem)

  setup_quadrature_point_coordinates!(Xs, el_coords, re)
  setup_shape_function_values!(Ns, re)
  setup_shape_function_gradients_and_JxWs!(∇N_Xs, JxWs, el_coords, re)

  if !(shape_function_gradient_conversion <: NoGradientConversion)
    N_V = voigt_dim(shape_function_gradient_conversion)
    Bs = Matrix{SMatrix{N_V, N * D, Rtype, N_V * N * D}}(undef, length(re.interpolants), block.num_elem)
    setup_converted_gradients!(Bs, ∇N_Xs, shape_function_gradient_conversion)
    fspace = StructArray{QuadratureValuesWithSymmetricGradient{N, D, Rtype, L1, N_V, N * D, N_V * N * D}}((Xs, Ns, ∇N_Xs, Bs, JxWs))
  else
    fspace = StructArray{QuadratureValues{N, D, Rtype, L1}}((Xs, Ns, ∇N_Xs, JxWs))
  end
  
  return fspace
end

function FunctionSpace(
  mesh::Mesh{Rtype, I, B},
  block_id::Union{Int, String},
  q_degree::Int;
  n_state_variables::Int = 0,
  shape_function_gradient_conversion::Type{Conv} = NoGradientConversion
) where {Rtype, I, B, Conv <: AbstractGradientConversion}

  # unpack stuff from mesh
  block  = filter(x -> x.id == block_id, mesh.blocks)[1]

  # make reference finite element for this block
  el_type = element_types[uppercase(block.elem_type)](q_degree)
  re      = ReferenceFE(el_type, I, Rtype)

  return FunctionSpace(mesh, block, re; 
                       n_state_variables=n_state_variables,
                       shape_function_gradient_conversion=shape_function_gradient_conversion)
end
