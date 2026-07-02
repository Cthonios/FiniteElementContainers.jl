import FiniteElementContainers as FEC
using FiniteElementContainers
using Gmsh
using StaticArrays
using Tensors

struct TwoFieldSolidMechanics{NF, NP, NS} <: AbstractPhysics{NF, NP, NS}
end

struct Displ <: AbstractPhysics{2, 3, 0}
end

struct Pressure <: AbstractPhysics{1, 3, 0}
end

function FiniteElementContainers.create_properties(::TwoFieldSolidMechanics)
    ρ = 1e3
    K = 1.e9
    G = 1.e6
    return SVector{3, Float64}(ρ, K, G)
end

function jacobian(∇u)
    return det(∇u + one(∇u))
end

function pk1_stress_iso(props, ∇u, p)
    κ, μ    = props[2], props[3]
    F       = ∇u + one(∇u)
    J       = det(F)
    J_m_13  = 1. / cbrt(J)
    J_m_23  = J_m_13 * J_m_13
    I_1     = tr(tdot(F))
    F_inv_T = inv(F)'
    P_iso   = μ * J_m_23 * (F - (1. / 3.) * I_1 * F_inv_T)
    return P_iso
end

function pk1_stress_vol(props, ∇u, p)
    κ, μ    = props[2], props[3]
    F       = ∇u + one(∇u)
    J       = det(F)
    F_inv_T = inv(F)'
    # P_vol   = 0.5 * κ * (J * J - 1.) * F_inv_T
    P_vol   = p * J * F_inv_T
    return P_vol
end

material_tangent_iso(props, ∇u, p) = Tensors.gradient(z -> pk1_stress_iso(props, z, p), ∇u) 
material_tangent_vol(props, ∇u, p) = Tensors.gradient(z -> pk1_stress_vol(props, z, p), ∇u) 

@inline function FiniteElementContainers.residual(
    physics::TwoFieldSolidMechanics, interps, x_el, t, dt,
    u_el, u_el_old, state_old_q, state_new_q, props_el
)
    u_el, p_el = u_el
    interps_u, interps_p = interps
    x_el_u, x_el_p = x_el
    interps_u = map_interpolants(interps_u, x_el_u)
    interps_p = map_interpolants(interps_p, x_el_p)
    JxW_u = interps_u.JxW
    JxW_p = interps_p.JxW
    ∇u_q = interpolate_field_gradients(Displ(), interps_u, u_el)
    ∇u_q = modify_field_gradients(PlaneStrain(), ∇u_q)
    p_q = interpolate_field_values(Pressure(), interps_p, p_el)

    # constitutive
    P_iso = pk1_stress_iso(props, ∇u_q, p_q[1])
    P_vol = pk1_stress_vol(props, ∇u_q, p_q[1])
    J = jacobian(∇u_q)

    P_q = extract_stress(PlaneStrain(), P_iso + P_vol)
    G_q = discrete_gradient(PlaneStrain(), interps_u.∇N_X)
    R_u = JxW_u * G_q * P_q
    R_p = JxW_p * (J - one(J)) * interps_p.N
    return R_u, R_p
end

@inline function FiniteElementContainers.stiffness(
    physics::TwoFieldSolidMechanics, interps, x_el, t, dt,
    u_el, u_el_old, state_old_q, state_new_q, props_el
)
    u_el, p_el = u_el
    interps_u, interps_p = interps
    x_el_u, x_el_p = x_el
    interps_u = map_interpolants(interps_u, x_el_u)
    interps_p = map_interpolants(interps_p, x_el_p)
    JxW_u = interps_u.JxW
    JxW_p = interps_p.JxW
    ∇u_q = interpolate_field_gradients(Displ(), interps_u, u_el)
    ∇u_q = modify_field_gradients(PlaneStrain(), ∇u_q)
    p_q = interpolate_field_values(Pressure(), interps_p, p_el)
    J_q = jacobian(∇u_q)
    F_q = ∇u_q + one(∇u_q)
    F_inv_T_q = inv(F_q)'
    dPdp_q = extract_stress(PlaneStrain(), J_q * F_inv_T_q)
    A_iso = material_tangent_iso(props, ∇u_q, p_q[1])
    A_vol = material_tangent_vol(props, ∇u_q, p_q[1])
    G_q = discrete_gradient(PlaneStrain(), interps_u.∇N_X)
    G_pu_x = J_q .* (
        F_inv_T_q[1, 1] * interps_u.∇N_X[: ,1] +
        F_inv_T_q[2, 1] * interps_u.∇N_X[: ,2]
    )

    G_pu_y = J_q * (
        F_inv_T_q[1, 2] * interps_u.∇N_X[:, 1] +
        F_inv_T_q[2, 2] * interps_u.∇N_X[:, 2]
    )
    Nd = length(G_pu_x)

    tup = MVector{2 * Nd, eltype(G_pu_x)}(undef)

    for i in 1:Nd
        tup[2i-1] = G_pu_x[i]
        tup[2i]   = G_pu_y[i]
    end

    G_pu_q = SVector{2 * Nd, eltype(G_pu_x)}(tup)
    K_uu = JxW_u * G_q * extract_stiffness(PlaneStrain(), A_iso + A_vol) * G_q'
    K_up = JxW_u * G_q * dPdp_q * interps_p.N'
    K_pu = JxW_p * interps_p.N * G_pu_q'
    K_pp = zero(SMatrix{length(p_el), length(p_el), Float64, length(p_el)^2})
    return (
        (K_uu, K_up),
        (K_pu, K_pp)
    )
end

mesh_u = UnstructuredMesh(Base.source_dir() * "/geometry_q2.geo")
mesh_p = UnstructuredMesh(Base.source_dir() * "/geometry_q1.geo")

V_u = FunctionSpace(mesh_u, H1Field, Lagrange)
V_p = FunctionSpace(mesh_p, H1Field, Lagrange)
# V_J = FunctionSpace(mesh_p, H1Field, Lagrange)

u = VectorFunction(V_u, "displ")
p = ScalarFunction(V_p, "pressure")
# J = ScalarFunction(V_J, "jacobian")

zero_func(_, _) = 0.0
displ_func(_, t) = 0.1 * t
dbcs_u = DirichletBC[
    DirichletBC("displ_x", zero_func; nodeset_name = "Left")
    DirichletBC("displ_y", zero_func; nodeset_name = "Left")
    DirichletBC("displ_x", zero_func; nodeset_name = "Right")
    DirichletBC("displ_y", displ_func; nodeset_name = "Right")
]
physics = TwoFieldSolidMechanics{3, 0, 0}()
props = create_properties(physics)
times = TimeStepper(0.0, 1.0, 20)

dof_u, dof_p = DofManager(u), DofManager(p)
dof = (dof_u, dof_p)
# dof_u, dof_p, dof_J = DofManager(u), DofManager(p), DofManager(J)
# dof = (dof_u, dof_p, dof_J)
asm = FEC.BlockSparseMatrixAssembler(dof)

p_u    = create_parameters(mesh_u, SparseMatrixAssembler(dof_u), physics, props; dirichlet_bcs = dbcs_u, times = times)
p_p    = create_parameters(mesh_p, SparseMatrixAssembler(dof_p), physics, props; times = times)
params = (p_u, p_p)

FEC.update_dofs!(
    asm,
    (p_u.dirichlet_bcs, p_p.dirichlet_bcs),
    (p_u.periodic_bcs, p_p.periodic_bcs)
)

Uu = create_unknowns(asm)
U  = create_field(asm)

assemble_vector!(asm, residual, Uu, params)
assemble_matrix!(asm, stiffness, Uu, params)