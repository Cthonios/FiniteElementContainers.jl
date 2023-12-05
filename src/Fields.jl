abstract type AbstractField{T, N} <: FEMContainer end
Base.size(f::F) where F <: AbstractField = size(f.values)
# Base.convert(type::Type, f::F) where F <: Field = 

struct NodalField{T, N, Arr <: AbstractArray{T, N}} <: AbstractField{T, N}
  values::Arr
end

function NodalField(n_fields::Int, n_nodes, type::Type{<:Number} = Float64)
  vals = zeros(type, n_fields, n_nodes)
  return NodalField{type, 2, typeof(vals)}(vals)
end

function NodalField(mesh::Mesh, n_fields::Int, type::Type{<:Number} = Float64) where Mesh <: AbstractMesh
  return NodalField(n_fields, num_nodes(mesh), type)
end






const NodalCoordinates{T, N, Arr} = NodalField{T, N, Arr}

function NodalCoordinates(mesh::Mesh) where Mesh <: AbstractMesh
  # return NodalField{}
  coords = coordinates(mesh)
  return NodalField{eltype(coords), 2, typeof(coords)}(coords)
end
