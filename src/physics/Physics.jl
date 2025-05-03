abstract type AbstractPhysics{NF, NP, NS} end

num_fields(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NF
num_properties(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NP
num_states(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NS

function create_properties(physics::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS}
  @assert false "You need to implement the create_properties method for physics $(physics) or 
  type $(typeof(physics))!"
end

function create_properties(::AbstractPhysics{NF, 0, NS}) where {NF, NS}
  return SVector{0, Float64}()
end

function create_initial_state(::AbstractPhysics{NF, NP, 0}) where {NF, NP}
  return SVector{0, Float64}()
end

# Can we make something that makes interfacing with kernels easier?
# How can we make something like this work nicely with AD?
# struct PhysicsQuadratureState{T, ND, NN, NF, NP, NS, NDxNF, NNxND, NNxNNxND}
#   # u::SVector{NF, T}
#   # ∇u::SMatrix{NF, ND, T, NDxNF}
#   # props::SVector{NP, T}
#   # state_old::SVector{NS, T}
#   # interpolants and gauss weight at quadrature point
#   N::SVector{NN, T}
#   ∇N_ξ::SMatrix{NN, ND, T, NNxND}
#   ∇∇N_ξ::SArray{Tuple{NN, ND, ND}, T, 3, NNxNNxND}
#   JxW::T
#   # element level fields
#   u_el::SMatrix{}
# end 

# physics like methods
function damping end
function energy end
function mass end
function residual end
function stiffness end

# optimization like methods
function gradient end
function hessian end
function value end
