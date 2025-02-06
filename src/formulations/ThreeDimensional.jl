"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ThreeDimensional <: AbstractMechanicsFormulation{3}
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(::ThreeDimensional, ∇N_X)
  N   = size(∇N_X, 1)
  # tup = ntuple(i -> 0.0, Val(9 * 3 * N))
  tup = zeros(SVector{9 * 3 * N, eltype(∇N_X)})

  for n in 1:N
    k = 3 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, ∇N_X[n, 1], k + 3)

    #
    k = 3 * (n - 1) + 3 * 3 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 2], k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, ∇N_X[n, 2], k + 3)

    #
    k = 3 * (n - 1) + 6 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 7 * 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 3], k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 8 * 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, ∇N_X[n, 3], k + 3)
  end

  return SMatrix{3 * N, 9, eltype(∇N_X), 3 * N * 9}(tup.data)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_symmetric_gradient(::ThreeDimensional, ∇N_X)
  N   = size(∇N_X, 1)
  # tup = ntuple(i -> 0.0, Val(6 * 3 * N))
  tup = zeros(SVector{6 * 3 * N, eltype(∇N_X)})

  for n in 1:N
    k = 3 * (n - 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 2], k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, ∇N_X[n, 3], k + 3)

    k = 3 * (n - 1) + 3 * 3 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 2)
    tup = setindex(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 3], k + 2)
    tup = setindex(tup, ∇N_X[n, 2], k + 3)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 1)
    tup = setindex(tup, 0.0,        k + 2)
    tup = setindex(tup, ∇N_X[n, 1], k + 3)

  end
  return SMatrix{3 * N, 6, eltype(∇N_X), 3 * N * 6}(tup.data)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_values(::ThreeDimensional, N)
  N_nodes = size(N, 1)
  # tup = ntuple(i -> 0.0, Val(3 * N_nodes))
  tup = zeros(SVector{3 * N_nodes, eltype(N)})

  for n in 1:N_nodes
    tup = setindex(tup, N[n], n)
  end

  for n in 1:N_nodes
    tup = setindex(tup, N[n], n + N_nodes)
  end

  for n in 1:N_nodes
    tup = setindex(tup, N[n], n + 2 * N_nodes)
  end

  # return SVector{3 * N_nodes, eltype(N)}(tup)
  return tup
end

"""
$(TYPEDSIGNATURES)
"""
function modify_field_gradients(::ThreeDimensional, ∇u_q, ::Type{<:SArray})
  return ∇u_q
end
