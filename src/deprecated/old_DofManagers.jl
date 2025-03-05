# abstract type DofManager{T, N, ND, NN, Bools} <: NodalField{T, N, ND, NN, Bools} end
# abstract type DofManager{T, N, ND, NN} end

"""
$(TYPEDEF)

```T```       - Storage type of Int internally\n
```ND```      - Number of dofs per node\n
```NN```      - number of total nodes\n
```ArrType``` - The type of storage for creating fields\n
```V```       - Actual array storage type\n

Constructors\n
```DofManager{A}(ND::Int, NN::Int)  where A```\n
```DofManager{ND, NN, I, ArrType}() where {ND, NN, I, ArrType}```\n
```DofManager{ND, NN, ArrType}()    where {ND, NN, ArrType}```

"""
# struct DofManager{T, ND, NN, ArrType, V <: AbstractArray{T, 1}}
struct DofManager{T, ND, NN, ArrType, V <: AbstractArray{T}}
  unknown_dofs::V
end

"""
```DofManager{A}(ND::Int, NN::Int) where A```
"""
function DofManager{A}(ND::Int, NN::Int) where A
  unknown_dofs = 1:ND * NN |> collect
  return DofManager{Int64, ND, NN, A, typeof(unknown_dofs)}(unknown_dofs)
end

"""
```DofManager{ND, NN, I, ArrType}() where {ND, NN, I, ArrType}```
"""
function DofManager{ND, NN, I, ArrType}() where {ND, NN, I, ArrType}
  unknown_dofs = 1:ND * NN |> collect
  return DofManager{I, ND, NN, ArrType, typeof(unknown_dofs)}(unknown_dofs)
end

"""
```DofManager{ND, NN, ArrType}() where {ND, NN, ArrType}```
"""
DofManager{ND, NN, ArrType}() where {ND, NN, ArrType} = DofManager{ND, NN, Int64, ArrType}()

"""
Default constructor from a mesh
"""
function DofManager(mesh::FileMesh, n_dofs::Int)
  NNodes = num_nodes(mesh) |> Int64
  dof = DofManager{n_dofs, NNodes, Vector{Float64}}()
  return dof
end

Base.show(io::IO, dof::DofManager) = 
print(io, "DofManager\n", 
          "  Number of nodes         = $(num_nodes(dof))\n",
          "  Number of dofs per node = $(num_dofs_per_node(dof))\n",
          "  Storage type            = $(typeof(dof.unknown_dofs))")

Base.eltype(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = T
Base.size(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = (ND, NN)
Base.size(::DofManager{T, ND, NN, A, V}, ::Val{1}) where {T, ND, NN, A, V} = ND
Base.size(::DofManager{T, ND, NN, A, V}, ::Val{2}) where {T, ND, NN, A, V} = NN
Base.size(dof::DofManager, n::Int) = size(dof, Val(n))
num_dofs_per_node(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = ND
num_nodes(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = NN

num_unknowns(dof::DofManager) = length(dof.unknown_dofs)
num_bcs(dof::DofManager) = (num_dofs_per_node(dof) * num_nodes(dof) - num_unknowns(dof))


"""
$(TYPEDSIGNATURES)
"""
# create_fields(::DofManager{T, ND, NN, Matrix{R}, V}) where {T, ND, NN, R, V} = 
# zero(SimpleNodalField{R, 2, ND, NN, Matrix{R}})
"""
$(TYPEDSIGNATURES)
"""
# create_fields(::DofManager{T, ND, NN, Vector{R}, V}) where {T, ND, NN, R, V} = 
# zero(VectorizedNodalField{R, 2, ND, NN, Vector{R}})
# create_fields(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A <: AbstractArray{<:Number, 1}, V} = 
# zero(VectorizedNodalField{eltype(A), 2, ND, NN, A})
create_fields(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A <: AbstractArray{<:Number, 1}, V} = 
zero(NodalField{eltype(A), ND, A}, NN)
"""
$(TYPEDSIGNATURES)
"""
dof_ids(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = 1:ND * NN
"""
$(TYPEDSIGNATURES)
"""
create_unknowns(dof::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = 
zeros(eltype(A), length(dof.unknown_dofs))
"""
$(TYPEDSIGNATURES)
"""
function update_fields!(U::NodalField, dof::DofManager, Uu::V) where V <: AbstractArray{<:Number, 1}
  @assert length(Uu) == length(dof.unknown_dofs)
  U[dof.unknown_dofs] = Uu
end

"""
$(TYPEDSIGNATURES)
"Default" method to reset all dofs to unknown
"""
function update_unknown_dofs!(dof::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V}
  resize!(dof.unknown_dofs, ND * NN)
  dof.unknown_dofs .= dof_ids(dof)
  return nothing
end

"""
$(TYPEDSIGNATURES)
Method when all dofs are updated at once

First it resets all dofs to unknowns, then one by one sets dofs to bcs in dofs

Assumes there's a unique set of nodes provided
"""
function update_unknown_dofs!(dof_manager::DofManager, dofs::V) where V <: AbstractVector{<:Integer}
  ND = num_dofs_per_node(dof_manager)
  NN = num_nodes(dof_manager)
  resize!(dof_manager.unknown_dofs, ND * NN)
  dof_manager.unknown_dofs .= 1:ND * NN
  deleteat!(dof_manager.unknown_dofs, dofs)
  return nothing
end

# """
# $(TYPEDSIGNATURES)
# eventually use this method
# """
function bc_ids(dof_manager::DofManager, nsets::NSets, dofs::Dofs) where {NSets, Dofs}
  D = num_dofs_per_node(dof_manager)

  n_bc_nodes = 0
  for nset in nsets
    n_bc_nodes += length(nset)
  end

  bc_nodes = Vector{eltype(nsets[1])}(undef, n_bc_nodes)
  index = 1
  for n in axes(nsets, 1)
    dof = dofs[n]
    for node in nsets[n]
      bc_nodes[index] = D * (node - 1) + dof
      index += 1
    end
  end 
  sort!(unique!(bc_nodes))
end

# TODO is this deprecated?
# """
# $(TYPEDSIGNATURES)
# """
function unknown_ids(dof_manager::DofManager, nsets::NSets, dofs::Dofs, bc_dofs) where {NSets, Dofs}
  # bc_dofs = bc_ids(dof_manager, nsets, dofs)
  dofs    = dof_ids(dof_manager) |> vec |> collect
  setdiff!(dofs, bc_dofs)
  dofs
end
