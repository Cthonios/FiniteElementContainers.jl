"""
$(TYPEDEF)
"""
abstract type AbstractElementFormulation{ND, NF} end

"""
$(TYPEDSIGNATURES)
"""
num_fields(::AbstractElementFormulation{ND, NF}) where {ND, NF} = NF
num_dimensions(::AbstractElementFormulation{ND, NF}) where {ND, NF} = ND

function discrete_gradient end # eventually to be deprecated
function discrete_symmetric_gradient end # eventually to be deprecated
function discrete_values end # eventually to be deprecated
function modify_field_gradients(::AbstractElementFormulation, ∇u)
    return ∇u
end
# vector solution case (gradient is a matrix or tensor)
"""
$(TYPEDSIGNATURES)
Method to in place take a set of ```vals``` and calculate
f_i = G * vals where is the discrete_gradient
"""
function scatter_with_gradients!(
  storage::AbstractField, form::AbstractElementFormulation{ND, NF},
  e, conns, ∇N_X, vals
) where {ND, NF}
  N = size(∇N_X, 1)

  # potentially more conversion types here
  if isa(vals, Tensor)
    vals = extract_stress(SMatrix, form, vals)
  end

  for n in 1:N
    global_base = NF * (conns[n] - 1)

    for d in axes(vals, 1)
      contrib = zero(eltype(storage))
      for j in axes(vals, 2)
        contrib += ∇N_X[n, j] * vals[d, j]
      end
      fec_atomic_add!(storage, global_base + d, contrib)
    end
  end
end
# scalar case
"""
$(TYPEDSIGNATURES)
scalar case to calculate in place things like
K_el = k∇N_X ⋅ ∇N_X'
"""
function scatter_with_gradients_and_gradients!(
  storage::AbstractVector,
  ::AbstractElementFormulation{ND, NF},
  e,
  conns,
  ∇N_X,
  k::T
) where {ND, NF, T <: Number}
  @assert ND == size(∇N_X, 2)
  N        = size(∇N_X, 1)
  NDPE     = N * NF
  start_id = (e - 1) * NDPE * NDPE + 1
  ids      = start_id:(start_id + NDPE * NDPE - 1)
  inc      = 1

  for n1 in 1:N
    for n2 in 1:N
      contrib = zero(T)
      for j in 1:ND
        contrib += ∇N_X[n1, j] * k * ∇N_X[n2, j]
      end
      storage[ids[inc]] += contrib
      inc += 1
    end
  end

  return nothing
end
"""
Case for total lagrange formulations
of solid mechanics or other physics
that need fourth order full 81 component tensors in 3d
"""
function scatter_with_gradients_and_gradients!(
  storage::AbstractVector,
  form::AbstractElementFormulation{ND, NF},
  e,
  conns,
  ∇N_X,
  A::Tensor{4, 3, T, 81}
) where {ND, NF, T <: Number}
  K_voigt  = extract_stiffness(form, A)
  @assert ND == size(∇N_X, 2)
  N        = size(∇N_X, 1)
  NDPE     = N * NF
  start_id = (e - 1) * NDPE * NDPE + 1
  ids      = start_id:(start_id + NDPE * NDPE - 1)
  inc      = 1

  for n2 in 1:N
    for d2 in 1:NF
      for n1 in 1:N
        for d1 in 1:NF
          contrib = zero(T)
          for j1 in 1:ND
            a = (j1 - 1) * ND + d1
            for j2 in 1:ND
              b = (j2 - 1) * ND + d2
              contrib += ∇N_X[n1, j1] * K_voigt[a, b] * ∇N_X[n2, j2]
            end
          end

          storage[ids[inc]] += contrib
          inc += 1
        end
      end
    end
  end

  return nothing
end
# implement for those that have it
"""
Scalar equation specialization
f_el = N * vals
"""
function scatter_with_values!(
  storage::AbstractField, ::AbstractElementFormulation{ND, NF},
  e, conns, N, vals::T
) where {ND, NF, T <: Number}
  # TODO make this compile time checked through generics
  n_dofs = size(storage, 1)
  @assert n_dofs == NF
  for n in 1:length(N)
    contrib = N[n] * vals
    global_id = n_dofs * (conns[n] - 1) + 1
    fec_atomic_add!(storage, global_id, contrib)
  end
end
"""
General vector specialization
f_el = N * vals
"""
function scatter_with_values!(
    storage::AbstractField, ::AbstractElementFormulation{ND, NF},
    e, conns, N, vals::SVector{NF, T}
) where {ND, NF, T <: Number}
    # TODO make this compile time checked through generics
    n_dofs = size(storage, 1)
    @assert n_dofs == NF
    for nf in axes(storage, 1)
      for n in 1:length(N)
        contrib = N[n] * vals
        global_id = n_dofs * (conns[n] - 1) + nf
        fec_atomic_add!(storage, global_id, contrib[nf])
      end
    end
end
"""
scalar implementation
"""
function scatter_with_values_and_values!(
  storage::AbstractVector,
  ::AbstractElementFormulation{ND, NF},
  e,
  conns,
  N,
  ρ::T
) where {ND, NF, T <: Number}
  Nn   = length(N)
  NDPE = Nn * NF
  start_id = (e - 1) * NDPE * NDPE + 1
  ids      = start_id:(start_id + NDPE * NDPE - 1)
  inc      = 1

  for n2 in 1:Nn
    for n1 in 1:Nn
      contrib = N[n1] * ρ * N[n2]
      for d in 1:NF
        storage[ids[inc]] += contrib
        inc += 1
      end
    end
  end

  return nothing
end

# doesn't implement extract stress/stiffness

##########################################################################
# General non-mechanics default
##########################################################################
struct GeneralFormulation{ND, NF} <: AbstractElementFormulation{ND, NF}
end

##########################################################################
# Mechanics base
##########################################################################
abstract type AbstractMechanicsElementFormulation{ND} <: AbstractElementFormulation{ND, ND} end
function extract_stiffness end
function extract_stress end
function scatter_with_symmetric_gradient! end

##########################################################################
# Axisymmetric strain kinematics
##########################################################################
# TODO

##########################################################################
# Plane strain kinematics
##########################################################################
"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct PlaneStrain <: AbstractMechanicsElementFormulation{2}
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(::PlaneStrain, ∇N_X)
  N   = size(∇N_X, 1)
  tup = zeros(SVector{4 * 2 * N, eltype(∇N_X)})

  for n in 1:N
    k = 2 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)

    k = 2 * (n - 1) + 2 * N
    tup = setindex(tup, ∇N_X[n, 1], k + 2)
    
    k = 2 * (n - 1)  + 2 * 2 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)

    k = 2 * (n - 1) + 3 * 2 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 2)
  end
  return SMatrix{2 * N, 4, eltype(∇N_X), 2 * N * 4}(tup.data)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_symmetric_gradient(::PlaneStrain, ∇N_X)
  N   = size(∇N_X, 1)
  tup = zeros(SVector{3 * 2 * N, eltype(∇N_X)})

  for n in 1:N
    k = 2 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)

    k = 2 * (n - 1) + 2 * N
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
function discrete_values(::PlaneStrain, N)
  N_nodes = size(N, 1)
  tup = zeros(SVector{2 * N_nodes, eltype(N)})

  for n in 1:N_nodes
    tup = setindex(tup, N[n], n)
  end

  for n in 1:N_nodes
    tup = setindex(tup, N[n], n + N_nodes)
  end

  return tup
end

function extract_stress(::PlaneStrain, S::SymmetricTensor{2, 3, T, 6}) where T <: Number
  return SVector{3, T}((S[1, 1], S[2, 2], S[1, 2]))
end

"""
$(TYPEDSIGNATURES)
"""
function extract_stress(::PlaneStrain, P::Tensor{2, 3, T, 9}) where T <: Number
  return SVector{4, T}((
    P[1, 1], P[2, 1],
    P[1, 2], P[2, 2]
  ))
end

function extract_stress(::Type{<:SMatrix}, ::PlaneStrain, P::Tensor{2, 3, T, 9}) where T <: Number
  return SMatrix{2, 2, T, 4}((
    P[1, 1], P[2, 1],
    P[1, 2], P[2, 2]
  ))
end

"""
$(TYPEDSIGNATURES)
"""
function extract_stiffness(::PlaneStrain, A::Tensor{4, 3, T, 81}) where T <: Number
  A_mat = tovoigt(SMatrix, A)
  return SMatrix{4, 4, T, 16}((
    A_mat[1, 1], A_mat[9, 1], A_mat[6, 1], A_mat[2, 1],
    A_mat[1, 9], A_mat[9, 9], A_mat[6, 9], A_mat[2, 9],
    A_mat[1, 6], A_mat[9, 6], A_mat[6, 6], A_mat[2, 6],
    A_mat[1, 2], A_mat[9, 2], A_mat[6, 2], A_mat[2, 2],
  ))
end

"""
$(TYPEDSIGNATURES)
"""
function modify_field_gradients(::PlaneStrain, ∇u_q::SMatrix{2, 2, T, 4}) where T <: Number
  return Tensor{2, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        0.0
  ))
end

function scatter_with_symmetric_gradients!(storage::AbstractField, form::PlaneStrain, e, conns, ∇N_X, S::SymmetricTensor{2, 3, T, 6}) where T <: Number
  N      = size(∇N_X, 1)
  n_dofs = size(storage, 1)
  S_vec = extract_stress(form, S)
  for n in 1:N
      contrib_odd  = ∇N_X[n, 1] * S_vec[1] + ∇N_X[n, 2] * S_vec[3]  # row 2n-1
      contrib_even = ∇N_X[n, 2] * S_vec[2] + ∇N_X[n, 1] * S_vec[3]  # row 2n

      global_id_odd  = n_dofs * (conns[n] - 1) + 1
      global_id_even = n_dofs * (conns[n] - 1) + 2

      # Atomix.@atomic storage.data[global_id_odd]  += contrib_odd
      # Atomix.@atomic storage.data[global_id_even] += contrib_even
      fec_atomic_add!(storage, global_id_odd, contrib_odd)
      fec_atomic_add!(storage, global_id_even, contrib_even)
  end

  return nothing
end

##########################################################################
# 3D kinematics
##########################################################################
"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ThreeDimensional <: AbstractMechanicsElementFormulation{3}
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(::ThreeDimensional, ∇N_X)
  N   = size(∇N_X, 1)
  tup = zeros(SVector{9 * 3 * N, eltype(∇N_X)})

  for n in 1:N
    k = 3 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)

    k = 3 * (n - 1) + 3 * N
    tup = setindex(tup, ∇N_X[n, 1], k + 2)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = setindex(tup, ∇N_X[n, 1], k + 3)

    k = 3 * (n - 1) + 3 * 3 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 2)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 3)

    k = 3 * (n - 1) + 6 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 1)

    k = 3 * (n - 1) + 7 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 2)

    k = 3 * (n - 1) + 8 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 3)
  end

  return SMatrix{3 * N, 9, eltype(∇N_X), 3 * N * 9}(tup.data)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_symmetric_gradient(::ThreeDimensional, ∇N_X)
  N   = size(∇N_X, 1)
  tup = zeros(SVector{6 * 3 * N, eltype(∇N_X)})

  for n in 1:N
    k = 3 * (n - 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 1)

    k = 3 * (n - 1) + 3 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 2)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 3)

    k = 3 * (n - 1) + 3 * 3 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 2)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 2)
    tup = setindex(tup, ∇N_X[n, 2], k + 3)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = setindex(tup, ∇N_X[n, 3], k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 3)

  end
  return SMatrix{3 * N, 6, eltype(∇N_X), 3 * N * 6}(tup.data)
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_values(::ThreeDimensional, N)
  N_nodes = size(N, 1)
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

  return tup
end

function extract_stress(::ThreeDimensional, S::SymmetricTensor{2, 3, T, 6}) where T <: Number
  return SVector{6, T}(
    S[1, 1], S[2, 2], S[3, 3],
    S[1, 2], S[2, 3], S[3, 1]
  )
end

"""
$(TYPEDSIGNATURES)
"""
function extract_stress(::ThreeDimensional, P::Tensor{2, 3, T, 9}) where T <: Number
  return SVector{9, T}(P.data)
end

function extract_stress(::Type{<:SMatrix}, ::ThreeDimensional, P::Tensor{2, 3, T, 9}) where T <: Number
  return SMatrix{3, 3, T, 9}(P.data)
end

"""
$(TYPEDSIGNATURES)
"""
function extract_stiffness(::ThreeDimensional, A_in::Tensor{4, 3, T, 81}) where T <: Number
  return SMatrix{9, 9, T, 81}(A_in.data)
end

"""
$(TYPEDSIGNATURES)
"""
function modify_field_gradients(::ThreeDimensional, ∇u_q::SMatrix{3, 3, T, 9}) where T <: Number
  return Tensor{2, 3, T, 9}(∇u_q.data)
end

function scatter_with_symmetric_gradients!(
  storage::AbstractField, 
  form::ThreeDimensional, 
  e, 
  conns, 
  ∇N_X,
  S::SymmetricTensor{2, 3, T, 6}
) where T <: Number
  N       = size(∇N_X, 1)
  n_dofs  = size(storage, 1)
  S_vec   = extract_stress(form, S)
  for n in 1:N
      contrib_1 = ∇N_X[n, 1] * S_vec[1] + ∇N_X[n, 2] * S_vec[4] + ∇N_X[n, 3] * S_vec[6]
      contrib_2 = ∇N_X[n, 2] * S_vec[2] + ∇N_X[n, 1] * S_vec[4] + ∇N_X[n, 3] * S_vec[5]
      contrib_3 = ∇N_X[n, 3] * S_vec[3] + ∇N_X[n, 2] * S_vec[5] + ∇N_X[n, 1] * S_vec[6]

      global_base = n_dofs * (conns[n] - 1)
      # Atomix.@atomic storage.data[global_base + 1] += contrib_1
      # Atomix.@atomic storage.data[global_base + 2] += contrib_2
      # Atomix.@atomic storage.data[global_base + 3] += contrib_3
      fec_atomic_add!(storage, global_base + 1, contrib_1)
      fec_atomic_add!(storage, global_base + 2, contrib_2)
      fec_atomic_add!(storage, global_base + 3, contrib_3)
  end

  return nothing
end
