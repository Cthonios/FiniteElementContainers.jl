module FiniteElementContainersPartitionedArraysExt

using Exodus
using FiniteElementContainers
using PartitionedArrays

function PartitionedArrays.uniform_partition(ranks, ::Type{<:ExodusDatabase}, base_file::String)
  parts = uniform_partition(ranks, base_file)
  
end

end # module