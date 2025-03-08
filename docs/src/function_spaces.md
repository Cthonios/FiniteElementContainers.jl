# FunctionSpace

```@repl fspace
using Exodus, FiniteElementContainers
mesh = UnstructuredMesh("../../test/poisson/poisson.g")
V = FunctionSpace(mesh, H1Field, Lagrange)
```

## API
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["FunctionSpaces.jl"]
Order = [:type, :function]
```