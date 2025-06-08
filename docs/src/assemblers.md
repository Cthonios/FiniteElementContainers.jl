```@meta
CurrentModule = FiniteElementContainers
```

# Assemblers
This section describes the assemblers that are currently available and their abstract interface.

All assemblers must possess at minimum a ```DofManager```.

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
Pages = ["Scalar.jl"]
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
