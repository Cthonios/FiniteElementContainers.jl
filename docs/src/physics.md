# Physics
# Writing Your First Physics Model

The central abstraction in **FiniteElementContainers** is the `AbstractPhysics` interface.

A physics object describes **what equations should be solved**, while the finite element framework handles

* element loops,
* quadrature,
* interpolation,
* sparse assembly,
* MPI parallelism,
* and solver infrastructure.

As a developer, you only write the **local element operators**.

The easiest way to understand the interface is by implementing Poisson's equation and then extending the same ideas to nonlinear solid mechanics.

---

## Defining a new physics type

Every physics model inherits from `AbstractPhysics`.

```julia
struct Poisson{F<:Function} <: AbstractPhysics{1,0,0}
    func::F
end
```

The type parameters

```julia
AbstractPhysics{NF,NP,NS}
```

specify

| Parameter | Meaning                       |
| --------- | ----------------------------- |
| `NF`      | Number of unknown fields      |
| `NP`      | Number of material properties |
| `NS`      | Number of state variables     |

For Poisson,

```julia
AbstractPhysics{1, 0, 0}
```

means

* one scalar unknown
* no material parameters
* no internal state.

By comparison, a nonlinear solid mechanics implementation would be more complicated and
below is an example

```julia
struct SolidMechanics{NF, NP, NS, Form, Model} <: AbstractPhysics{NF,NP,NS}
    formulation::Form
    constitutive_model::Model
end
```

which contains both the kinematic formulation and constitutive law for convenience.

---

## Properties

Physics with material parameters overload

```julia
create_properties(physics, inputs)
```

For solid mechanics,

```julia
function create_properties(physics::SolidMechanics, inputs)
    density = inputs["density"]
    mat_props =
        ConstitutiveModels.initialize_props(
            physics.constitutive_model,
            inputs
        )
    return pushfirst!(Array(mat_props), density)
end
```

constructs the material property vector from the application input file.

The Poisson problem has no material properties, so no implementation is required.

---

# State variables

History-dependent constitutive models must initialize their internal variables.

In our solid mechanics example, we'll simply forward this to a constitutive model from the package [ConstitutiveModels.jl](https://github.com/cthonios/ConstitutiveModels.jl),

```julia
function create_initial_state(physics::SolidMechanics)
    return ConstitutiveModels.initialize_state(physics.constitutive_model)
end
```

while Poisson uses the default empty state,

```julia
SVector{0, Float64}()
```

because it has no history variables.

---

# Solution interpolation

Nearly every element kernel begins the same way.

```julia
interps = map_interpolants(interps, x_el)
```
which takes the reference element interpolants stored in ``interps`` and maps them to the current element coordinates. 

Once interpolants are properly mapped once can interpolate the element level solution field to the current quadrature point with one of the following methods
```julia
u_q = interpolate_field_values(physics, interps, u_el)
```
or
```julia
∇u_q = interpolate_field_gradients(physics, interps, u_el)
```
or
```julia
u_q, ∇u_q = interpolate_field_values_and_gradients(physics, interps, u_el)
```

These helper functions evaluate the finite element solution at the current quadrature point.

You never need to manually evaluate basis functions.

---

# Energy example

The Poisson energy is remarkably simple.

```julia
function energy(
    physics,
    interps,
    x_el,
    t,
    dt,
    u_el,
    u_el_old,
    state_old_q,
    state_new_q,
    props_el
)
    interps = map_interpolants(interps,x_el)
    (; X_q, JxW) = interps
    u_q, ∇u_q = interpolate_field_values_and_gradients(physics, interps, u_el)

    e_q = 0.5 * dot(∇u_q, ∇u_q) - dot(u_q,physics.func(X_q, 0))

    return JxW * e_q
end
```

The framework automatically integrates these quadrature contributions over the element.

For nonlinear solid mechanics the implementation looks almost identical, but instead we leverage ``ConstitutiveModels.jl``

```julia
∇u_q = interpolate_field_gradients(physics, interps, u_el)
ψ_q =
    ConstitutiveModels.helmholtz_free_energy(
        physics.constitutive_model,
        props,
        dt,
        ∇u_q,
        θ,
        state_old_q,
        state_new_q
    )

return JxW*ψ_q
```

The only difference is that the constitutive model supplies the strain energy density.

---

# Residual example

The residual defines the weak form of the PDE.

For Poisson,

```julia
∇u_q =
    interpolate_field_gradients(
        physics,
        interps,
        u_el
    )

R_q =
    ∇u_q*∇N_X'
    -
    N'*physics.func(X_q,0)

return JxW*R_q[:]
```

which corresponds directly to

[
\int
\nabla u\cdot\nabla v
---------------------

fv
,d\Omega.
]

For solid mechanics the same pattern appears,

```julia
P_q =
    ConstitutiveModels.pk1_stress(
        physics.constitutive_model,
        props,
        dt,
        ∇u_q,
        θ,
        state_old_q,
        state_new_q
    )

G_q =
    discrete_gradient(
        physics.formulation,
        ∇N_X
    )

return JxW*(G_q*P_q)
```

The only difference is replacing the Laplacian operator with a constitutive stress tensor.

---

# Tangent/Jacobian example

Newton solvers require the element Jacobian.

For Poisson,

```julia
K_q =
    ∇N_X*∇N_X'

return JxW*K_q
```

which is simply the Laplace operator.

Solid mechanics follows exactly the same pattern,

```julia
A_q =
    ConstitutiveModels.material_tangent(
        physics.constitutive_model,
        props,
        dt,
        ∇u_q,
        θ,
        state_old_q,
        state_new_q
    )

G_q =
    discrete_gradient(
        physics.formulation,
        ∇N_X
    )

return
    JxW*
    G_q*
    extract_stiffness(
        physics.formulation,
        A_q
    )*
    G_q'
```

The finite element framework assembles these local matrices into the global tangent matrix automatically.

---

# Mass

Transient problems implement

```julia
mass(...)
```

The Poisson implementation constructs

```julia
M_el =
    N*N'
```

while solid mechanics simply multiplies by density,

```julia
ρ = props_el[1]

return
    JxW*
    ρ*
    M_el
```

No global assembly logic is required.

---

# Matrix-Free Operators

For large-scale problems it is often preferable to compute

[
Kv
]

instead of assembling

[
K.
]

This is accomplished by implementing

```julia
stiffness_action(...)
mass_action(...)
```

The Poisson implementation,

```julia
return
    JxW*
    ∇N_X*
    (∇N_X'*v_el)
```

computes the matrix-vector product directly without ever forming the stiffness matrix.

The solid mechanics implementation follows the same pattern using the constitutive tangent tensor.

---

# In-Place Assembly

Every finite element operator in **FiniteElementContainers** has two possible implementations.

The first returns a local element quantity

```julia
residual(...)
stiffness(...)
mass(...)
```

while the second scatters contributions **directly into the global assembly storage**

```julia
residual!(...)
stiffness!(...)
mass!(...)
```

The two interfaces compute exactly the same mathematical object, but they differ in how the result is assembled.

---

### Returning an Element Residual

The simplest implementation explicitly constructs the element residual vector and returns it.

For the Poisson equation,

```julia
@inline function FiniteElementContainers.residual(
    physics,
    interps,
    x_el,
    t,
    dt,
    u_el,
    u_el_old,
    state_old_q,
    state_new_q,
    props_el
)

    interps = map_interpolants(interps, x_el)

    (; X_q, N, ∇N_X, JxW) = interps

    ∇u_q =
        interpolate_field_gradients(
            physics,
            interps,
            u_el
        )

    R_q =
        ∇u_q * ∇N_X'
        -
        N' * physics.func(X_q,0)

    return JxW * R_q[:]

end
```

Conceptually, this computes

[
R_e
===

\int_{\Omega_e}
\nabla N^T \nabla u
-------------------

N^T f
,d\Omega,
]

and returns the element vector to the assembly routine.

This approach is easy to read and is often ideal for prototyping new formulations.

---

### Direct Assembly

For production simulations, however, allocating a temporary element vector for every quadrature point and every element can become expensive.

Instead, the framework allows the local contributions to be **scattered directly into the global assembly buffer**.

The Poisson implementation becomes

```julia
@inline function FiniteElementContainers.residual!(
    storage,
    e,
    physics,
    t,
    dt,
    props_el,
    state_old_q,
    state_new_q,
    conn,
    interps,
    x_el,
    u_el,
    u_el_old
)

    interps = map_interpolants(interps, x_el)

    (; X_q, N, ∇N_X, JxW) = interps

    ∇u_q =
        interpolate_field_gradients(
            physics,
            interps,
            u_el
        )

    form =
        GeneralFormulation{
            size(X_q,1),
            num_fields(physics)
        }()

    scatter_with_gradients!(
        storage,
        form,
        e,
        conn,
        ∇N_X,
        JxW*∇u_q
    )

    scatter_with_values!(
        storage,
        form,
        e,
        conn,
        N,
        -JxW*physics.func(X_q,0)
    )

    return nothing

end
```

Notice that **no residual vector is ever explicitly constructed**.

Instead, each contribution is immediately accumulated into the global assembly storage.

---

### Understanding the Scatter Operations

The helper functions encapsulate common finite element assembly patterns.

For example,

```julia
scatter_with_gradients!(
    storage,
    form,
    e,
    conn,
    ∇N_X,
    JxW*∇u_q
)
```

computes the gradient contribution

[
\int_{\Omega_e}
\nabla N^T
\nabla u
,d\Omega
]

and scatters the resulting entries directly into the global residual vector.

Similarly,

```julia
scatter_with_values!(
    storage,
    form,
    e,
    conn,
    N,
    -JxW*physics.func(X_q,0)
)
```

adds the forcing term

[
-\int_{\Omega_e}
N^T f
,d\Omega.
]

The framework handles all indexing, degree-of-freedom ordering, and accumulation internally.

The physics implementation simply specifies the mathematical quantity being integrated.

---

### Why This Matters

The `residual!` interface avoids allocating temporary element vectors entirely.

Instead of

```
quadrature
      ↓
construct element vector
      ↓
return element vector
      ↓
assemble globally
```

the computation becomes

```
quadrature
      ↓
compute local contribution
      ↓
scatter directly into global storage
```

This reduces memory traffic and temporary allocations, which can have a significant impact for large nonlinear simulations.

---

Exactly the same idea applies to the other finite element operators.

For example,

```julia
stiffness!(...)
```

uses helper routines such as

```julia
scatter_with_gradients_and_gradients!(
    storage,
    form,
    e,
    conn,
    ∇N_X,
    JxW
)
```

to assemble

[
\int_{\Omega_e}
\nabla N^T
\nabla N
,d\Omega
]

directly into the global sparse matrix.

Likewise,

```julia
mass!(...)
```

uses

```julia
scatter_with_values_and_values!(...)
```

to assemble

[
\int_{\Omega_e}
N^T N
,d\Omega
]

without ever explicitly constructing an element mass matrix.

---

### Recommendation

For most new physics implementations, it is often easiest to begin by implementing the return-value interface

```julia
energy(...)
residual(...)
stiffness(...)
mass(...)
```

since these closely resemble the mathematical weak form.

Once the formulation has been verified, the corresponding in-place versions

```julia
energy!(...)
residual!(...)
stiffness!(...)
mass!(...)
```

can be implemented using the `scatter_with_*` helper routines to eliminate potentially large ```StaticArrays``` that may blow out the stack and obtain significantly better performance on large-scale simulations.


---

# Summary

Implementing a new finite element formulation typically consists of only a handful of methods:

```text
AbstractPhysics
residual
stiffness
(optional)
create_properties
create_initial_state
energy
mass
mass_action
stiffness_action
```

The Poisson implementation demonstrates the simplest possible PDE, while the `SolidMechanics` implementation shows that the exact same interface naturally extends to nonlinear constitutive models with material parameters and history variables.

In both cases, the developer writes only the local physics. The finite element machinery—interpolation, quadrature, sparse assembly, and solver infrastructure—is completely managed by **FiniteElementContainers**.

## API
```@autodocs
Modules = [FiniteElementContainers]
Pages = ["Physics.jl"]
Order = [:type, :function]
```