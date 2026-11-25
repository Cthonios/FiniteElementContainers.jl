```@meta
CurrentModule = FiniteElementContainers
```

# Boundary Conditions
This section describes the user facing API for boundary conditions
along with the implementation details.

## DirichletBC
We can set up dirichlet boundary conditions on a variable ```u```
and sideset ```sset_1``` with a zero function as follows.
```@repl
using FiniteElementContainers
bc_func(x, t) = 0.
bc = DirichletBC("u", bc_func; sideset_name = "sset_1")
```
Internally this is eventually converted in a ```DirichletBCContainer```

Dirichlet bcs can be setup on element blocks, nodesets, or sidesets. The appropriate keyword argument needs to be supplied with the ```DirichletBC``` constructor.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["bcs/DirichletBC.jl"]
Order = [:type, :function]
```

# NeumannBC
We can setup Neumann bcs on a variable ```u``` and sideset ```sset_1``` with a
simple constant function as follows
```@repl
using FiniteElementContainers
using StaticArrays
bc_func(x, t) = SVector{1, Float64}(1.)
bc = NeumannBC("u", bc_func, "sset_1")
```
Note that in comparison to the dirichlet bc example above, the function in this case returns a ```SVector``` of size 1. This will hold for any variable ```u``` that has a single dof. For vector variables, e.g. a traction vector in continuum mechanics, would need something like
```@repl
using FiniteElementContainers
using StaticArrays
ND = 2
bc_func(x, t) = SVector{ND, Float64}(1.)
bc = NeumannBC("u", bc_func, "sset_1")
```
where ```ND``` is the number of dimensions.

```@autodocs
Modules = [FiniteElementContainers]
Pages = ["bcs/NeumannBC.jl"]
Order = [:type, :function]
```

# PeriodicBC
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["bcs/PeriodicBC.jl"]
Order = [:type, :function]
```

# PointSource
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["bcs/PointSource.jl"]
Order = [:type, :function]
```

# RobinBC
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["bcs/RobinBC.jl"]
Order = [:type, :function]
```

# Source
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["bcs/Source.jl"]
Order = [:type, :function]
```

## Boundary Condition Implementation Details
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["bcs/BoundaryConditions.jl"]
Order = [:type, :function]
```
