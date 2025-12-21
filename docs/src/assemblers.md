```@meta
CurrentModule = FiniteElementContainers
```

# Assemblers
This section describes the assemblers that are currently available and their abstract interface.

All assemblers must possess at minimum a ```DofManager```.

The assemblers possess all the baggage to use the internal method ```sparse!``` in ```SparseArrays.jl```. This method allows for a zero allocation instantiation of a ```SparseMatrixCSC``` type on the CPU. There are also methods available to ease in the conversion of CSC types and other sparse types such as CSR. 

On the GPU, however, this type is first converted to an appropriate COO matrix type on the desired backend. There is unfortunately not a unified sparse matrix API in ```julia``` for GPUs, so we implement this functionality in package extensions. On CUDA, for example, the operational sequence to get a ```CuSparseMatrixCSC``` is to first
sort the COO ```(row, col, val)``` triplets so they are ordered by row and then column. Then a ```CuSparseMatrixCOO``` type is created and converted to a ```CuSparseMatrixCSC``` type via ```CUDA.jl``` methods. An identical approach is taken for RocM types.

NOTE: This is one of the most actively developed areas of the package. Please use caution with any method beginning with a "_" as these are internal methods that will change without notice.

## Matrices
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Matrix.jl"]
Order = [:function]
```

## Matrix Action
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["MatrixAction.jl"]
Order = [:function]
```

## Matrix Action
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["NeumannBC.jl"]
Order = [:function]
```

## Scalar
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["QuadratureQuantity.jl"]
Order = [:function]
```

## Vector
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Vector.jl"]
Order = [:function]
```

## Abstract Interface
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Assemblers.jl"]
Order = [:type, :function]
```

## SparseMatrixAssembler
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["SparseMatrixAssembler.jl"]
Order = [:type, :function]
```

## SparsityPattern
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["SparsityPattern.jl"]
Order = [:type, :function]
```
