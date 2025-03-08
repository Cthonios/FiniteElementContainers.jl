```@meta
CurrentModule = FiniteElementContainers
DocTestFilters = [
    r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
    r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
    r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
]
```

# DofManager
The ```DofManager``` is a struct that keeps track of which dofs are unknown or constrained. This can work with simple or mixed finite element spaces of various types. It is a glorified book keeper.

A ```DofManager``` can be created as follows. First we must create functions for our variables of interest from their associated function spaces.

```@repl dof
using Exodus, FiniteElementContainers
mesh = UnstructuredMesh("../../test/poisson/poisson.g")
V = FunctionSpace(mesh, H1Field, Lagrange)
u = VectorFunction(V, :u)
t = ScalarFunction(V, :t)
```
Now we can supply these variables to the ```DofManager``` which takes varargs as inputs
```@repl dof
dof = DofManager(u, t)
```
The print methods for this struct show simple metadata about the current dofs for each possible function space.

A set of unknowns can be set up as follows
```@repl dof
field = create_unknowns(dof)
```

We can create fields of the right size from the ```DofManager``` with the following methods

```@repl dof
field = create_field(dof, H1Field)
```

These methods take the backed of ```dof``` into account to ensure that the fields or unknowns produced are on the same device, e.g. CPU/GPU if ```dof``` is on the CPU/GPU.

This struct is created with all dofs initially set as unknown. To modify the unknowns we can do the following

## API
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["DofManagers.jl"]
Order = [:type, :function]
```