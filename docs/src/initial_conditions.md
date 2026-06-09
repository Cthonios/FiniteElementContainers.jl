# Initial Conditions
# Defining Initial Conditions

Initial conditions provide the starting values for the unknown fields before a simulation begins.

In **FiniteElementContainers**, initial conditions are specified **declaratively** by associating

* a field variable,
* a function,
* and a geometric entity,

rather than by directly modifying the solution vector.

The framework automatically determines which degrees of freedom should be initialized and applies the corresponding values to the field.

---

# Overview

An initial condition consists of three pieces of information:

1. **Which variable should be initialized?**
2. **What function defines its value?**
3. **Where should it be applied?**

For example,

```julia
InitialCondition(
    "temperature",
    x -> 300.0;
    block_name = "solid"
)
```

states

> Initialize the `temperature` field to `300` on every node belonging to the element block `"solid"`.

No degree-of-freedom indexing is required.

---

# Creating an Initial Condition

The user-facing API consists of the `InitialCondition` type.

```julia
InitialCondition(
    variable_name,
    function;
    block_name=...,
    nodeset_name=...,
    sideset_name=...
)
```

The function should accept the spatial coordinates of a node and return the desired field value.

For example, a constant temperature field may be written as

```julia
ic = InitialCondition(
    "temperature",
    x -> 300.0;
    block_name = "block_1"
)
```

while a spatially varying field can be defined as

```julia
ic = InitialCondition(
    "temperature",
    x -> sin(pi*x[1]);
    block_name = "block_1"
)
```

The function is evaluated automatically at every active node.

---

# Selecting the Region

An initial condition must be associated with **exactly one** mesh entity.

For example, to initialize an entire element block,

```julia
InitialCondition(
    "ux",
    x -> 0.0;
    block_name = "solid"
)
```

To initialize only a node set,

```julia
InitialCondition(
    "temperature",
    x -> 500.0;
    nodeset_name = "left"
)
```

or a side set,

```julia
InitialCondition(
    "pressure",
    x -> 1.0;
    sideset_name = "wall"
)
```

Exactly one of

* `block_name`
* `nodeset_name`
* `sideset_name`

must be specified.

Attempting to specify multiple entities simultaneously results in an error.

---

# Multiple Initial Conditions

Multiple initial conditions can be supplied to initialize different variables or different regions.

For example,

```julia
ics = [

    InitialCondition(
        "ux",
        x -> 0.0;
        block_name="solid"
    ),

    InitialCondition(
        "uy",
        x -> 0.0;
        block_name="solid"
    ),

    InitialCondition(
        "temperature",
        x -> 300.0;
        block_name="solid"
    )

]
```

Each initial condition is processed independently and applied to the appropriate degrees of freedom.

---

# Internal Caching

When a simulation is created, the framework converts each `InitialCondition` into an `InitialConditionContainer`.

This preprocessing step performs all mesh-dependent work once:

```julia
InitialConditionContainer(
    mesh,
    dof,
    ic
)
```

During construction the container

* finds the appropriate mesh entities,
* determines the active nodes,
* maps variable names to degree-of-freedom indices,
* computes the global DOF numbers,
* sorts and removes duplicates,
* allocates storage for the initial values.

As a result, no mesh searches are required during the simulation itself.

---

# Computing Initial Values

The values stored in an initial condition are updated by evaluating the user-supplied function at every active node.

Internally this is performed by

```julia
_update_ic_values!(
    ic,
    func,
    X
)
```

which evaluates

```julia
X_temp = SVector(
    X[1,loc],
    X[2,loc],
    X[3,loc]
)

ic.vals[n] = func(X_temp)
```

for every node in the selected region.

The resulting values are cached for later use.

This allows arbitrary spatially varying initial conditions while avoiding repeated function evaluations.

---

# Updating the Solution Field

Once the values have been computed, they are inserted into the solution vector.

Internally,

```julia
_update_field_ics!(
    U,
    ic
)
```

performs

```julia
fec_foreach(ic.dofs) do n

    dof = ic.dofs[n]
    val = ic.vals[n]

    U[dof] = val

end
```

For every active degree of freedom, the cached value is copied directly into the global field vector.

The user never needs to manually manipulate DOF indices.

---

# The `InitialConditions` Collection

Individual initial conditions are grouped together into an `InitialConditions` object.

```julia
ics = InitialConditions(
    mesh,
    dof,
    user_initial_conditions
)
```

This object owns

* the cached DOF mappings,
* the cached nodal values,
* and the user-defined functions.

It provides a convenient mechanism for updating every initial condition simultaneously.

---

# Applying Initial Conditions

Applying all initial conditions to a field is simply

```julia
update_ic_values!(
    ics,
    X
)

update_field_ics!(
    U,
    ics
)
```

The first call evaluates every user-defined function at the mesh coordinates,

while the second copies those values into the appropriate locations of the global solution vector.

---

# GPU Compatibility

The cached `InitialConditionContainer` objects are compatible with the `Adapt` interface and can be transferred directly to accelerator memory.

This allows the same initial condition definitions to be evaluated on

* CPUs,
* GPUs,
* or other accelerator backends

without changing user code.

The user-facing API remains identical regardless of the execution backend.

---

# Summary

Initial conditions in **FiniteElementContainers** are defined by specifying **what field to initialize, what function to evaluate, and where to apply it**.

The framework automatically converts these high-level descriptions into efficient degree-of-freedom mappings, evaluates the functions at the appropriate mesh locations, and updates the solution vector without requiring the user to manually work with mesh connectivity or DOF numbering.

This allows complex spatially varying initial conditions to be expressed using only a few lines of Julia code while retaining high performance on both CPU and GPU architectures.

# API
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["InitialConditions.jl"]
Order = [:type, :function]
```