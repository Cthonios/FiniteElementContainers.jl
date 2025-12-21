# 3. Coupled Example

This example mimics the second [Moose tutorial](https://mooseframework.inl.gov/getting_started/examples_and_tutorials/examples/ex03_coupling.html).

## Strong Form
Consider the advection equation on a domain ``\Omega \subset \mathbb{R}^d`` with boundary ``\partial \Omega``:

``
\begin{aligned}
-\nabla \cdot \nabla u + \nabla v \cdot\nabla u &= 0 \quad \text{in } \Omega,\newline
-\nabla\cdot\nabla v &= 0\quad \text{in } \Omega,
\end{aligned}
``

with Dirichlet boundary conditions

``
\begin{aligned}
u(x) &= g_1\quad\text{for }x\in\partial\Omega_{d_1} \newline
v(x) &= g_2\quad\text{for }x\in\partial\Omega_{d_2}
\end{aligned}
``

and Neumann boundary conditions

``
\begin{aligned}
-\mathbf{n}\cdot\nabla u &= h_1, \newline
-\mathbf{n}\cdot\nabla v &= h_2
\end{aligned}
``

# Weak form
``
\begin{aligned}
\int_\Omega (\nabla w_1\cdot\nabla u + w_1\nabla v\cdot\nabla u)d\Omega &= \int_{\partial\Omega_1} w_1h_1d\Gamma \newline
\int_\Omega \nabla w_2\cdot\nabla vd\Omega &= \int_{\partial\Omega_2} w_2h_2d\Gamma \newline
\end{aligned}
``

# Implementation
```julia
struct CoupledPhysics <: AbstractPhysics{2, 0, 0}
end

@inline function FiniteElementContainers.residual(
    physics::CoupledPhysics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    ∇u_q = interpolate_field_gradients(physics, interps, u_el)
    ∇u = unpack_field(∇u_q, 1)
    ∇v = unpack_field(∇u_q, 2)
    term = dot(∇u, ∇v)
    R_u = ∇N_X * ∇u + term * N
    R_v = ∇N_X * ∇v
    R = vcat(R_u', R_v')
    return JxW * R[:]
end

# begin lazy below
@inline function FiniteElementContainers.stiffness(
    physics::CoupledPhysics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
    ForwardDiff.jacobian(
        z -> FiniteElementContainers.residual(
            physics, interps, x_el, t, dt, z, u_el_old, state_old_q, state_new_q, props_el
        ), u_el
    )
end
```

# Example problem
```julia
one_func(_, _) = 1.0
two_func(_, _) = 2.0
zero_func(_, _) = 0.0

mesh = UnstructuredMesh("mug.e")
V = FunctionSpace(mesh, H1Field, Lagrange)
physics = CoupledPhysics()
props = create_properties(physics)
u = FiniteElementContainers.GeneralFunction(
    ScalarFunction(V, :u),
    ScalarFunction(V, :v)
)

asm = SparseMatrixAssembler(u; use_condensed=true)
dbcs = [
    DirichletBC(:u, two_func; sideset_name = :bottom)
    DirichletBC(:u, zero_func; sideset_name = :top)
    DirichletBC(:v, one_func; sideset_name = :bottom)
    DirichletBC(:v, zero_func; sideset_name = :top)
]
U = create_field(asm)
p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)

solver = NewtonSolver(DirectLinearSolver(asm))
integrator = QuasiStaticIntegrator(solver)
evolve!(integrator, p)

U = p.h1_field

pp = PostProcessor(mesh, "output.e", u)
write_times(pp, 1, 0.0)
write_field(pp, 1, ("u", "v"), U)
close(pp)
```

Visualized results
![](assets/coupled_problem_u.png)
![](assets/coupled_problem_v.png)
