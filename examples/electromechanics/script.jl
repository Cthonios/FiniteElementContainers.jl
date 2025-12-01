using Exodus
using FiniteElementContainers
using ForwardDiff
using StaticArrays
using Tensors

struct ElectroMechanics <: FiniteElementContainers.AbstractPhysics{4, 4, 0}
end

function FiniteElementContainers.create_properties(physics, props_dict)
    return SVector{4, Float64}(
        props_dict["shear modulus"],
        props_dict["bulk modulus"],
        props_dict["Jm"],
        props_dict["permitivity"]
    )
end

@inline function FiniteElementContainers.energy(
    physics::ElectroMechanics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, props_el
  )
    G, K, Jm, ϵ = props_el

    interps = map_interpolants(interps, x_el)
    (; X_q, N, ∇N_X, JxW) = interps
    U_q, ∇U_q = interpolate_field_values_and_gradients(physics, interps, u_el)

    ∇u_q = Tensor{2, 3, eltype(u_el), 9}(∇U_q[1:3, :])
    F_q = ∇u_q + one(∇u_q)
    J_q = det(F_q)
    C_q = tdot(F_q)
    C_inv_q = inv(C_q)
    E_R_q = Vec{3, eltype(u_el)}(∇U_q[4, :])
    D_R_q = ϵ * J_q * dot(C_inv_q, E_R_q)

    # kinematics
    I_1_bar = tr(J_q^(-2. / 3.) * C_q)

    # constitutive
    ψ_vol  = 0.5 * K * (0.5 * (J_q^2 - 1) - log(J_q))
    ψ_dev  = -G * Jm / 2. * log(1. - (I_1_bar - 3.) / Jm)
    ψ_mech = ψ_vol + ψ_dev
    ψ_elec = (1. / (2 * ϵ * J_q)) * dot(D_R_q, C_q, D_R_q)
    ψ_q    = ψ_mech + ψ_elec
    state_new_q = copy(state_old_q)
    return JxW * ψ_q, state_new_q
end

@inline function FiniteElementContainers.residual(
    physics::ElectroMechanics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, props_el
)
    state_new_q = copy(state_old_q)
    return ForwardDiff.gradient(
        z -> energy(physics, interps, x_el, t, dt, z, u_el_old, state_old_q, props_el)[1], u_el
    ), state_new_q
end

@inline function FiniteElementContainers.stiffness(
    physics::ElectroMechanics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, props_el
)
    state_new_q = copy(state_old_q)
    return ForwardDiff.hessian(
        z -> energy(physics, interps, x_el, t, dt, z, u_el_old, state_old_q, props_el)[1], u_el
    ), state_new_q
end

function run_electromechanics()
    mesh_file = Base.source_dir() * "/cube.g"
    output_file = Base.source_dir() * "/output.e"

    zero_func(_, _) = 0.
    potential_func(_, t) = 0.8 * t

    mesh = UnstructuredMesh(mesh_file)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    displ = VectorFunction(V, :displ)
    φ = ScalarFunction(V, :phi)
    u = FiniteElementContainers.GeneralFunction(displ, φ)
    asm = SparseMatrixAssembler(u; use_condensed=true)
    times = TimeStepper(0., 1., 11)

    dbcs = DirichletBC[
        DirichletBC("displ_x", "ssx-", zero_func)
        DirichletBC("displ_y", "ssy-", zero_func)
        DirichletBC("displ_z", "ssz-", zero_func)
        DirichletBC("phi", "ssz-", zero_func)
        DirichletBC("phi", "ssz+", potential_func)
    ]

    physics = ElectroMechanics()
    props = Dict(
        "shear modulus" => 1.0,
        "bulk modulus"  => 100.0,
        "Jm"            => 7.,
        "permitivity"   => 1.
    )
    props = create_properties(physics, props)
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs, times=times)

    # setup solver and integrator
    # linear_solver = IterativeLinearSolver(asm, :CgSolver)
    linear_solver = DirectLinearSolver(asm)
    solver = NewtonSolver(linear_solver)
    # solver = nsolver(lsolver(asm))
    integrator = QuasiStaticIntegrator(solver)

    pp = PostProcessor(mesh, output_file, u)
    write_times(pp, 1, 0.0)

    for n in 1:11
        evolve!(integrator, p)
        write_times(pp, n + 1, FiniteElementContainers.current_time(p.times))
        write_field(pp, n + 1, ("displ_x", "displ_y", "displ_z", "phi"), p.h1_field)
    end

    close(pp)
end

run_electromechanics()
