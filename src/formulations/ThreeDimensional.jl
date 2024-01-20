"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ThreeDimensional <: AbstractMechanicsFormulation
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(::ThreeDimensional, ∇N_X)
  N   = size(∇N_X, 1)
  tup = ntuple(i -> 0.0, Val(9 * 3 * N))

  for n in 1:N
    k = 3 * (n - 1) 
    tup = set(tup, ∇N_X[n, 1], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 1], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 1], k + 3)

    #
    k = 3 * (n - 1) + 3 * 3 * N
    tup = set(tup, ∇N_X[n, 2], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 2], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 2], k + 3)

    #
    k = 3 * (n - 1) + 6 * 3 * N
    tup = set(tup, ∇N_X[n, 3], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 7 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 3], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 8 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 3], k + 3)
  end

  return SMatrix{3 * N, 9, eltype(∇N_X), 3 * N * 9}(tup)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_symmetric_gradient(::Type{ThreeDimensional}, ∇N_X)
  N   = size(∇N_X, 1)
  tup = ntuple(i -> 0.0, Val(6 * 3 * N))

  for n in 1:N
    k = 3 * (n - 1)
    tup = set(tup, ∇N_X[n, 1], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 2], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 3], k + 3)

    k = 3 * (n - 1) + 3 * 3 * N
    tup = set(tup, ∇N_X[n, 2], k + 1)
    tup = set(tup, ∇N_X[n, 1], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 3], k + 2)
    tup = set(tup, ∇N_X[n, 2], k + 3)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = set(tup, ∇N_X[n, 3], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 1], k + 3)

  end
  return SMatrix{3 * N, 6, eltype(∇N_X), 3 * N * 6}(tup)
end

"""
$(TYPEDSIGNATURES)
"""
function modify_field_gradients(::ThreeDimensional, ∇u_q, ::Type{<:SArray})
  return ∇u_q
end

"""
$(TYPEDSIGNATURES)
"""
function modify_field_gradients(::ThreeDimensional, ∇u_q, ::Type{<:Tensor})
  return Tensor{2, 3, eltype(∇u_q), 9}(∇u_q)
  # return Tensor{2, 3, eltype(∇u_q), 9}((
  #   ∇u_q[1, 1], ∇u_q[2, 1], ∇u_q[3, 1],
  #   ∇u_q[1, 2], ∇u_q[2, 2], ∇u_q[3, 2],
  #   ∇u_q[1, 3], ∇u_q[2, 3], ∇u_q[3, 3],
  # ))
end

"""
$(TYPEDSIGNATURES)
"""
modify_field_gradients(form::ThreeDimensional, ∇u_q, type = Tensor) =
modify_field_gradients(form, ∇u_q, type)

"""
$(TYPEDSIGNATURES)
"""
function extract_stress(::ThreeDimensional, P::Tensor{2, 3, T, 9}) where T <: Number
  P_vec = tovoigt(SVector, P)
  return SVector{9, T}((
    P_vec[1], P_vec[9], P_vec[8],
    P_vec[6], P_vec[2], P_vec[7],
    P_vec[5], P_vec[4], P_vec[3]
  ))
end

"""
$(TYPEDSIGNATURES)
"""
function extract_stiffness(::ThreeDimensional, A_in::Tensor{4, 3, T, 81}) where T <: Number
  A = tovoigt(SMatrix, A_in)
  return SMatrix{9, 9, T, 81}((
    A[1, 1], A[9, 1], A[8, 1], A[6, 1], A[2, 1], A[7, 1], A[5, 1], A[4, 1], A[3, 1],
    A[1, 9], A[9, 9], A[8, 9], A[6, 9], A[2, 9], A[7, 9], A[5, 9], A[4, 9], A[3, 9],
    A[1, 8], A[9, 8], A[8, 8], A[6, 8], A[2, 8], A[7, 8], A[5, 8], A[4, 8], A[3, 8],
    A[1, 6], A[9, 6], A[8, 6], A[6, 6], A[2, 6], A[7, 6], A[5, 6], A[4, 6], A[3, 6],
    A[1, 2], A[9, 2], A[8, 2], A[6, 2], A[2, 2], A[7, 2], A[5, 2], A[4, 2], A[3, 2],
    A[1, 7], A[9, 7], A[8, 7], A[6, 7], A[2, 7], A[7, 7], A[5, 7], A[4, 7], A[3, 7],
    A[1, 5], A[9, 5], A[8, 5], A[6, 5], A[2, 5], A[7, 5], A[5, 5], A[4, 5], A[3, 5],
    A[1, 4], A[9, 4], A[8, 4], A[6, 4], A[2, 4], A[7, 4], A[5, 4], A[4, 4], A[3, 4],
    A[1, 3], A[9, 3], A[8, 3], A[6, 3], A[2, 3], A[7, 3], A[5, 3], A[4, 3], A[3, 3],
  ))
end
