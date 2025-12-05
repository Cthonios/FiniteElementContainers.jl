struct SparseMatrixAssemblerNew{
    Condensed,
    IV  <: AbstractArray{Int, 1},
    RV  <: AbstractArray{Float64, 1},
    Var <: AbstractFunction
} <: AbstractAssembler{DofManager{Condensed, Int, IV, Var}}
    constraint_storage::RV
    dof::DofManager{Condensed, Int, IV, Var}
    matrix_pattern::SparseMatrixPattern{IV, RV}
    vector_pattern::SparseVectorPattern{IV}
end

function SparseMatrixAssemblerNew(dof::DofManager)
    matrix_pattern = SparseMatrixPattern(dof)
    vector_pattern = SparseVectorPattern(dof)

    ND, NN = size(dof)
    n_total_dofs = ND * NN
    constraint_storage = zeros(n_total_dofs)
    constraint_storage[dof.dirichlet_dofs] .= 1.

    return SparseMatrixAssemblerNew(
        constraint_storage, dof,
        matrix_pattern, vector_pattern
    )
end

function SparseMatrixAssemblerNew(var::AbstractFunction; use_condensed::Bool = false)
    dof = DofManager(var; use_condensed=use_condensed)
    return SparseMatrixAssemblerNew(dof)
end

function Adapt.adapt_structure(to, asm::SparseMatrixAssemblerNew)
    return SparseMatrixAssemblerNew(
        adapt(to, asm.constraint_storage),
        adapt(to, asm.dof),
        adapt(to, asm.matrix_pattern),
        adapt(to, asm.vector_pattern)
    )
end

function Base.show(io::IO, asm::SparseMatrixAssemblerNew)
    println(io, "SparseMatrixAssembler")
    println(io, "  ", asm.dof)
end

function create_block_quadrature_cache(asm::SparseMatrixAssemblerNew, ::Type = Float64)
    T = eltype(asm.constraint_storage)
    backend = KA.get_backend(asm)
    fspace = function_space(asm.dof)
    cache = Matrix{T}[]
    for (key, val) in pairs(fspace.ref_fes)
        NQ = ReferenceFiniteElements.num_quadrature_points(val)
        NE = size(getfield(fspace.elem_conns, key), 2)
        field = zeros(T, NQ, NE)
        push!(cache, field)
    end
    cache = NamedTuple{keys(fspace.ref_fes)}(tuple(cache...))
    return adapt(backend, cache)
end

function create_sparse_matrix_cache(asm::SparseMatrixAssemblerNew)
    T = eltype(asm.constraint_storage)
    backend = KA.get_backend(asm)
    return KA.zeros(T, backend, num_entries(asm.matrix_pattern))
end

function create_sparse_vector_cache(asm::SparseMatrixAssemblerNew)
    T = eltype(asm.constraint_storage)
    backend = KA.get_backend(asm)
    return KA.zeros(T, backend, num_entries(asm.vector_pattern))
end

function assemble_scalar_new!(
    dof,
    func, Uu, p
)
    fspace = function_space(dof)
    t = current_time(p.times)
    Δt = time_step(p.times)
    _update_for_assembly!(p, dof, Uu)
    for (
        conns, 
        block_physics, ref_fe,
        state_old, state_new, props
    ) in zip(
        values(fspace.elem_conns), 
        values(p.physics), values(fspace.ref_fes),
        values(p.state_old), values(p.state_new),
        values(p.properties)
    )
        _assemble_block_scalar!(
            conns,
            func,
            block_physics, ref_fe,
            p.h1_coords, t, Δt,
            p.h1_field, p.h1_field_old,
            state_old, state_new, props
        )
    end
end

function _assemble_block_scalar!(
    conns, func, 
    physics, ref_fe, X, t, Δt, U, U_old, state_old, state_new, props
)
    mapreduce(+, axes(conns, 2)) do e
        x_el = _element_level_fields_flat(X, ref_fe, conns, e)
        u_el = _element_level_fields_flat(U, ref_fe, conns, e)
        u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, e)
        props_el = _element_level_properties(props, e)
        w = zero(eltype(X))
        for q in 1:num_quadrature_points(ref_fe)
        # mapreduce(+, 1:num_quadrature_points(ref_fe)) do q
            interps = _cell_interpolants(ref_fe, q)
            state_old_q = _quadrature_level_state(state_old, q, e)
            state_new_q = _quadrature_level_state(state_new, q, e)
            w += func(physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, state_new_q, props_el)
        end
        w
    end
end

function _update_dofs!(assembler::SparseMatrixAssemblerNew, dirichlet_dofs::T) where T <: AbstractArray{<:Integer, 1}
    _update_dofs!(assembler.matrix_pattern, assembler.dof, dirichlet_dofs)
end
