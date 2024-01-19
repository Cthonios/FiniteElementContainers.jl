```@meta
CurrentModule = FiniteElementContainers
DocTestFilters = [
    r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
    r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
    r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
]
```

# DofManager
The ```DofManager``` is a simple lightweight struct that keeps track of which dofs are unknown or constrained.

```@docs
DofManager
```

A ```DofManager``` can be create as follows using a simple non-vectorized storage for fields it will create

```jldoctest dof
julia> using FiniteElementContainers

julia> dof = DofManager{2, 10, Matrix{Float64}}()
DofManager
  Number of nodes         = 10
  Number of dofs per node = 2
  Storage type            = Vector{Int64}

```

We can then set some nodes as boundary condition degrees of freedom. Let's pick the following nodes ```[1, 2]``` with dofs ```[1, 1]``` fixed. This works out to global dofs ```[1, 3]```. We can then update the ```DofManager``` with a call to ```update_unknown_dofs!``` as follows.

```jldoctest dof
julia> bc_dofs = [1, 3]
2-element Vector{Int64}:
 1
 3

julia> update_unknown_dofs!(dof, bc_dofs)

```

Now the ```DofManager``` has its dofs properly set and we can create properly sized unknown vectors. This is done with a call to ```create_unknowns```.

```jldoctest dof
julia> Uu = create_unknowns(dof)
18-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0

```

This is useful for creating zero arrays that are properly sized to the current number of total unknown degrees of freedom. 

We can also create properly sized ```NodalField```s with a ```DofManager``` with a call to ```create_fields```.

```jldoctest dof
julia> U = create_fields(dof)
2×10 FiniteElementContainers.SimpleNodalField{Float64, 2, 2, 10, Matrix{Float64}}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

```

This creates a ```NodalField``` (specifically a ```SimpleNodalField``` since we initialized the ```DofManager``` with ```Matrix{Float64}``` as the storage type) with all zero entries that is sized for the maximum number of possible unknowns, e.g. no fixed Dirichlet BCs.

If we look at the internal storage of ```U```
```jldoctest dof
julia> U.vals
2×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

```
we can see that ```SimpleNodalField``` is simply acting as a wrapper around ```Matrix{Float64}``` with some additional meta-data. If on the other hand we initialize the ```DofManager``` with ```Vector{Float64}``` as the internal storage type we see the following

```jldoctest dof
julia> dof = DofManager{2, 10, Vector{Float64}}()
DofManager
  Number of nodes         = 10
  Number of dofs per node = 2
  Storage type            = Vector{Int64}

julia> U = create_fields(dof)
2×10 FiniteElementContainers.VectorizedNodalField{Float64, 2, 2, 10, Vector{Float64}}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```

```julia
julia> U.vals
20-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 ⋮
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0

```

As you can see this stored as a long vector of numbers rather than a matrix.
