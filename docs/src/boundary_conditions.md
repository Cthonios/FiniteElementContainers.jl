# Boundary Conditions
This section describes the user facing API for boundary conditions
along with the implementation details.

## DirichletBC
We can set up dirichlet boundary conditions on a variable ```u```
and sideset ```sset_1``` with a zero function as follows.
```@repl
using FiniteElementContainers
bc_func(x, t) = 0.
bc = DirichletBC(:u, :sset_1, bc_func)
```
Internally this is eventually converted in a ```DirichletBCContainer```

### API
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["DirichletBCs.jl"]
Order = [:type, :function]
```

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["NeumannBCs.jl"]
Order = [:type, :function]
```

## Boundary Condition Implementation Details
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["BoundaryConditions.jl"]
Order = [:type, :function]
```
