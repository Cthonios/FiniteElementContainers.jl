"""
$(TYPEDEF)
"""
abstract type AbstractElementFormulation{ND} end

"""
$(TYPEDSIGNATURES)
"""
num_dimensions(::AbstractElementFormulation{ND}) where ND = ND

function discrete_gradient end # eventually to be deprecated
function discrete_symmetric_gradient end # eventually to be deprecated
function discrete_values end # eventually to be deprecated
function modify_field_gradients(::AbstractElementFormulation{ND}, ∇u) where ND
    return ∇u
end
function project_with_gradients!(
    storage::AbstractField, form::AbstractElementFormulation{ND},
    e, conns, ∇N_X, vals
) where ND
    N = size(∇N_X, 1)
    for n in 1:N
        global_base = ND * (conns[n] - 1)
    
        for d in axes(vals, 1)
            contrib = zero(eltype(storage))
            for j in axes(vals, 2)
                contrib += ∇N_X[n, j] * vals[d, j]
            end
            # Atomix.@atomic storage.data[global_base + d] += contrib
            fec_atomic_add!(storage, global_base + d, contrib)
        end
    end
end
# function project_with_gradients_and_gradients!(
#   storage::AbstractField, form::AbstractElementFormulation{ND},
#   e, conns, ∇N_X, vals::T
# ) where {ND, T <: Number}

# end
# scalar case
function project_with_gradients_and_gradients!(
  storage::AbstractVector,
  form::AbstractElementFormulation{ND},
  e,
  conns,
  ∇N_X,
  k::T,
  JxW::T
) where {ND, T <: Number}
  N = size(∇N_X, 1)

  for n1 in 1:N
    global_i = conns[n1]

    for n2 in 1:N
      global_j = conns[n2]

      contrib = zero(T)
      for j in 1:ND
          contrib += ∇N_X[n1, j] * k * ∇N_X[n2, j]
      end

      # storage[global_i, global_j] += JxW * contrib
      # TODO finish me
      @assert false
    end
  end

  return nothing
end
function project_with_gradients_and_gradients!(
  storage::AbstractVector,
  form::AbstractElementFormulation{ND},
  e,
  conns,
  ∇N_X,
  A::Tensor{4, 3, T, 81},  # material tangent, ND2 = ND^2
  JxW::T
) where {ND, T <: Number}
  K_voigt = extract_stiffness(form, A)
  N = size(∇N_X, 1)

  for n1 in 1:N
    global_base_i = ND * (conns[n1] - 1)

    for n2 in 1:N
      global_base_j = ND * (conns[n2] - 1)

      for d1 in 1:ND   # row DOF for node n1
        for d2 in 1:ND  # col DOF for node n2
          # Compute G[row, :]' * K * G[col, :]
          # where row = global_base_i + d1, col = global_base_j + d2
          # G[row, a] = ∇N_X[n1, ceil(a/ND)] if (a-1)%ND+1 == d1, else 0
          # i.e. for column block a = (j-1)*ND + d1, G[row,a] = ∇N_X[n1, j]
          contrib = zero(T)
          for j1 in 1:ND   # gradient direction for node n1
              a = (j1 - 1) * ND + d1   # voigt row index into K
              for j2 in 1:ND  # gradient direction for node n2
                  b = (j2 - 1) * ND + d2  # voigt col index into K
                  contrib += ∇N_X[n1, j1] * K_voigt[a, b] * ∇N_X[n2, j2]
              end
          end

          # global_i = global_base_i + d1
          # global_j = global_base_j + d2
          # storage[global_i, global_j] += JxW * contrib
          # TODO finish me, above is not right for indexing
          @assert false
        end
      end
    end
  end

  return nothing
end
# implement for those that have it
function project_with_values!(
  storage::AbstractField, ::AbstractElementFormulation{ND},
  e, conns, N, vals::T
) where {ND, T <: Number}
  # TODO make this compile time checked through generics
  n_dofs = size(storage, 1)
  @assert n_dofs == ND
  # for d in axes(storage, 1)
  for n in 1:length(N)
    contrib = N[n] * vals
    global_id = n_dofs * (conns[n] - 1) + 1
    # Atomix.@atomic storage.data[global_id] += contrib
    fec_atomic_add!(storage, global_id, contrib)
  end
  # end
end
function project_with_values!(
    storage::AbstractField, ::AbstractElementFormulation{ND},
    e, conns, N, vals::SVector{ND, T}
) where {ND, T <: Number}
    # TODO make this compile time checked through generics
    n_dofs = size(storage, 1)
    @assert n_dofs == ND
    for d in axes(storage, 1)
      for n in 1:length(N)
        contrib = N[n] * vals
        global_id = n_dofs * (conns[n] - 1) + d
        # Atomix.@atomic storage.data[global_id] += contrib[d]
        fec_atomic_add!(storage, global_id, contrib[d])
      end
    end
end
# doesn't implement extract stress/stiffness

##########################################################################
# General non-mechanics default
##########################################################################
struct GeneralFormulation{ND} <: AbstractElementFormulation{ND}
end

##########################################################################
# Mechanics base
##########################################################################
abstract type AbstractMechanicsElementFormulation{ND} <: AbstractElementFormulation{ND} end
function extract_stiffness end
function extract_stress end
function project_with_symmetric_gradient! end

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

function project_with_symmetric_gradients!(storage::AbstractField, form::PlaneStrain, e, conns, ∇N_X, S::SymmetricTensor{2, 3, T, 6}) where T <: Number
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

function project_with_symmetric_gradients!(
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
