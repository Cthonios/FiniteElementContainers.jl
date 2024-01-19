"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ThreeDimensional <: AbstractMechanicsFormulation
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(::Type{ThreeDimensional}, ∇N_X)
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
