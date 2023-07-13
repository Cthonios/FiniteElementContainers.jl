function setup_element_coordinates!(
  el_coords::Vector{SMatrix{N, D, Ftype, L}},
  coords::Matrix{<:AbstractFloat}, block::Block{B}
) where {N, D, Ftype, L, B}
  for n in eachindex(el_coords)
    el_coords[n] = SMatrix{D, N, Ftype, L}(view(coords, :, view(block.conn, :, n)))
  end 
end

function setup_quadrature_point_coordinates!(
  ξs::Matrix{SVector{D, Ftype}}, 
  el_coords::Vector{SMatrix{D, N, Ftype, L}}, re::ReferenceFE{N, D, L, Itype, Ftype}
) where {D, N, L, Itype, Ftype}

  for e in eachindex(el_coords)
    for q in eachindex(re.ws)
      # ξs[q, e] = 
      # @show size(el_coords[e])
      # @show size(re.Ns[q])
      # temp = el_coords[e] * re.Ns[q]
      # @show size(temp)
      ξs[q, e] = SVector{D, Ftype}(el_coords[e] * re.Ns[q])
    end
  end
end

function setup_shape_function_values!(
  Ns::Matrix{SVector{N, Ftype}}, re::ReferenceFE{N, D, L, Itype, Ftype}
) where {N, D, L, Itype, Ftype}

  for e in axes(Ns, 2)
    for q in axes(Ns, 1)
      Ns[q, e] = re.Ns[q]
    end
  end
end

function setup_shape_function_gradients_and_JxWs!(
  ∇N_Xs::Matrix{SMatrix{N, D, Ftype, L}}, JxWs::Matrix{Ftype},
  el_coords::Vector{SMatrix{D, N, Ftype, L}}, re::ReferenceFE{N, D, L, Itype, Ftype}
) where {N, D, L, Itype, Ftype}

  for e in axes(∇N_Xs, 2)
    for q in axes(∇N_Xs, 1)
      J = el_coords[e] * re.∇N_ξs[q]
      J_inv = inv(J)
      ∇N_Xs[q, e] = (J_inv * re.∇N_ξs[q]')'
      JxWs[q, e] = det(J) * re.ws[q]
    end
  end
end

struct FunctionSpace{N, D, L, Itype, Ftype}
  conn::Matrix{Itype}
  ξs::Matrix{SVector{D, Ftype}}
  Ns::Matrix{SVector{N, Ftype}}
  ∇N_Xs::Matrix{SMatrix{N, D, Ftype, L}}
  JxWs::Matrix{Ftype}
end

connectivity(f::FunctionSpace) = getfield(f, :conn)
connectivity(f::FunctionSpace, e::Integer) = view(getfield(f, :conn), :, e)

quadrature_point_coordinates(f::FunctionSpace) = getfield(f, :ξs)
quadrature_point_coordinates(f::FunctionSpace, e::Integer) = view(getfield(f, :ξs), :, e)
quadrature_point_coordinates(f::FunctionSpace, q::Integer, e::Integer) = view(getfield(f, :ξs), q, e)

shape_function_gradients(f::FunctionSpace) = getfield(f, :∇N_Xs)
shape_function_gradients(f::FunctionSpace, e::Integer) = view(getfield(f, :∇N_Xs), :, e)
shape_function_gradients(f::FunctionSpace, q::Integer, e::Integer) = view(getfield(f, :∇N_Xs), q, e)

shape_function_values(f::FunctionSpace) = getfield(f, :Ns)
shape_function_values(f::FunctionSpace, e::Integer) = view(getfield(f, :Ns), :, e)
shape_function_values(f::FunctionSpace, q::Integer, e::Integer) = view(getfield(f, :Ns), q, e)

JxWs(f::FunctionSpace) = getfield(f, :JxWs)
JxWs(f::FunctionSpace, e::Integer) = view(getfield(f, :JxWs), :, e)
JxWs(f::FunctionSpace, q::Integer, e::Integer) = view(getfield(f, :JxWs), q, e)

function Base.getindex(f::FunctionSpace, e::Integer)
  return (
    connectivity(f, e), 
    quadrature_point_coordinates(f, e),
    shape_function_values(f, e),
    shape_function_gradients(f, e), 
    JxWs(f, e)
  )
end

function Base.getindex(f::FunctionSpace, q::Integer, e::Integer)
  return (
    quadrature_point_coordinates(f, q, e),
    shape_function_values(f, q, e),
    shape_function_gradients(f, q, e),
    JxWs(f, q, e)
  )
end
Base.length(f::FunctionSpace) = size(f.conn, 2)

function FunctionSpace(
  coords::Matrix{<:AbstractFloat}, block::Block{B}, 
  re::ReferenceFE{N, D, L, Itype, Ftype}
) where {B, N, D, L, Itype, Ftype}

  el_coords = Vector{SMatrix{D, N, Ftype, L}}(undef, block.num_elem)
  ξs = Matrix{SVector{D, Ftype}}(undef, length(re.ws), block.num_elem)
  Ns = Matrix{SVector{N, Ftype}}(undef, length(re.ws), block.num_elem)
  ∇N_Xs = Matrix{SMatrix{N, D, Ftype, L}}(undef, length(re.ws), block.num_elem)
  JxWs = Matrix{Ftype}(undef, length(re.ws), block.num_elem)

  setup_element_coordinates!(el_coords, coords, block)
  setup_quadrature_point_coordinates!(ξs, el_coords, re)
  setup_shape_function_values!(Ns, re)
  setup_shape_function_gradients_and_JxWs!(∇N_Xs, JxWs, el_coords, re)

  return FunctionSpace{N, D, L, Itype, Ftype}(
    block.conn, ξs, Ns, ∇N_Xs, JxWs
  )
end
