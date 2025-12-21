# Boundary Conditions
This section describes the user facing API for boundary conditions
along with the implementation details.

## DirichletBC
We can set up dirichlet boundary conditions on a variable ```u```
and sideset ```sset_1``` with a zero function as follows.
```@repl
using FiniteElementContainers
bc_func(x, t) = 0.
bc = DirichletBC(:u, bc_func; sideset_name = :sset_1)
```
Internally this is eventually converted in a ```DirichletBCContainer```

Dirichlet bcs can be setup on element blocks, nodesets, or sidesets. The appropriate keyword argument needs to be supplied with the ```DirichletBC``` constructor.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["DirichletBCs.jl"]
Order = [:type, :function]
```

# NeumannBC
We can setup Neumann bcs on a variable ```u``` and sideset ```sset_1``` with a
simple constant function as follows
```@repl
using FiniteElementContainers
using StaticArrays
bc_func(x, t) = SVector{1, Float64}(1.)
bc = NeumannBC(:u, :sset_1, bc_func)
```
Note that in comparison to the dirichlet bc example above, the function in this case returns a ```SVector``` of size 1. This will hold for any variable ```u``` that has a single dof. For vector variables, e.g. a traction vector in continuum mechanics, would need something like
```@repl
using FiniteElementContainers
using StaticArrays
ND = 2
bc_func(x, t) = SVector{ND, Float64}(1.)
bc = NeumannBC(:u, :sset_1, bc_func)
```
where ```ND``` is the number of dimensions.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["NeumannBCs.jl"]
Order = [:type, :function]
```

# PeriodicBC
Periodic boundary conditions are very much a work in progress. There is currently
some machinary to implement a Lagrange multiplier approach. 

Stay tuned.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["PeriodicBCs.jl"]
Order = [:type, :function]
```

## Boundary Condition Implementation Details
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["BoundaryConditions.jl"]
Order = [:type, :function]
```
