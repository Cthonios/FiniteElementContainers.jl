```@meta
CurrentModule = FiniteElementContainers
```

# Assemblers
This section describes the assemblers that are currently available and their abstract interface.

All assemblers must possess at minimum a ```DofManager```.

## Abstract Interface
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Assemblers.jl"]
Order = [:type, :function]
```

## Abstract Interface - CPU Specialization
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["CPUGeneral.jl"]
Order = [:type, :function]
```


## Abstract Interface - GPU Specialization
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["GPUGeneral.jl"]
Order = [:type, :function]
```

## SparsityPattern
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["SparsityPattern.jl"]
Order = [:type, :function]
```

## SparseMatrixAssembler
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["SparseMatrixAssembler.jl"]
Order = [:type, :function]
```
