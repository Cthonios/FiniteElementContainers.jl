"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct IncompressiblePlaneStress <: AbstractMechanicsFormulation{2}
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(::IncompressiblePlaneStress, ∇N_X)
  N   = size(∇N_X, 1)
  # tup = ntuple(i -> 0.0, Val(4 * 2 * N))
  tup = zeros(SVector{4 * 2 * N, eltype(∇N_X)})

  for n in 1:N
    k = 2 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)
    tup = setindex(tup, 0.0,        k + 2)

    k = 2 * (n - 1) + 2 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 2)
    
    k = 2 * (n - 1)  + 2 * 2 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)
    tup = setindex(tup, 0.0,        k + 2)

    k = 2 * (n - 1) + 3 * 2 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 2], k + 2)
  end

  return SMatrix{2 * N, 4, eltype(∇N_X), 2 * N * 4}(tup.data)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_symmetric_gradient(::IncompressiblePlaneStress, ∇N_X)
  N   = size(∇N_X, 1)
  # tup = ntuple(i -> 0.0, Val(3 * 2 * N))
  tup = zeros(SVector{3 * 2 * N, eltype(∇N_X)})

  for n in 1:N
    k = 2 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)
    tup = setindex(tup, 0.0,        k + 2)

    k = 2 * (n - 1) + 2 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 2], k + 2)

    k = 2 * (n - 1) + 2 * 2 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 2)
  end

  return SMatrix{2 * N, 3, eltype(∇N_X), 2 * N * 3}(tup.data)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_values(::IncompressiblePlaneStress, N)
  N_nodes = size(N, 1)
  # tup = ntuple(i -> 0.0, Val(2 * N_nodes))
  tup = zeros(SVector{2 * N_nodes, eltype(N)})

  for n in 1:N_nodes
    tup = setindex(tup, N[n], n)
  end

  for n in 1:N_nodes
    tup = setindex(tup, N[n], n + N_nodes)
  end

  # return SVector{2 * N_nodes, eltype(N)}(tup)
  return tup
end

# """
# $(TYPEDSIGNATURES)
# """
# function modify_field_gradients(::IncompressiblePlaneStress, ∇u_q::SMatrix{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
#   return Tensor{2, 3, T, 9}((
#     ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
#     ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
#     0.0,        0.0,        1.0 / det(∇u_q)
#   ))
# end

"""
$(TYPEDSIGNATURES)
"""
function modify_field_gradients(::IncompressiblePlaneStress, ∇u_q::SMatrix{2, 2, T, 4}, ::Type{<:SArray}) where T <: Number
  return SMatrix{3, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        1.0 / det(∇u_q)
  ))
end

# """
# $(TYPEDSIGNATURES)
# """
# function modify_field_gradients(::IncompressiblePlaneStress, ∇u_q::Tensor{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
#   return Tensor{2, 3, T, 9}((
#     ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
#     ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
#     0.0,        0.0,        1.0 / det(∇u_q)
#   ))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function extract_stress(::IncompressiblePlaneStress, P::Tensor{2, 3, T, 9}) where T <: Number
#   P_vec = tovoigt(SVector, P)
#   return SVector{4, T}((
#     P_vec[1], P_vec[9], 
#     P_vec[6], P_vec[2]
#   ))
# end
# """
# $(TYPEDSIGNATURES)
# """
# function extract_stiffness(::IncompressiblePlaneStress, A::Tensor{4, 3, T, 81}) where T <: Number
#   A_mat = tovoigt(SMatrix, A)
#   return SMatrix{4, 4, T, 16}((
#     A_mat[1, 1], A_mat[9, 1], A_mat[6, 1], A_mat[2, 1],
#     A_mat[1, 9], A_mat[9, 9], A_mat[6, 9], A_mat[2, 9],
#     A_mat[1, 6], A_mat[9, 6], A_mat[6, 6], A_mat[2, 6],
#     A_mat[1, 2], A_mat[9, 2], A_mat[6, 2], A_mat[2, 2],
#   ))
# end
