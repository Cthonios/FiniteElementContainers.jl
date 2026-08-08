abstract type AbstractBlockAssembler <: AbstractAssembler end

function create_field(asm::AbstractBlockAssembler)
    return create_field(asm.dof)
end

function create_unknowns(asm::AbstractBlockAssembler)
    return create_unknowns(asm.dof)
end

struct BlockSparseMatrixAssembler{
    I <: AbstractVector{Int},
    R <: AbstractVector{Float64},
    D,
    F,
    U <: BlockedVector,
    S
} <: AbstractBlockAssembler
    dof::D
    matrix_patterns::Matrix{SparseMatrixPattern{I, R}}
    vector_patterns::Vector{SparseVectorPattern{I}}
    residual_storage::F
    residual_unknowns::U
    stiffness_storage::S
end

function BlockSparseMatrixAssembler(dof::Tuple)
    matrix_patterns = Matrix{SparseMatrixPattern{Vector{Int}, Vector{Float64}}}(undef, length(dof), length(dof))
    vector_patterns = Vector{SparseVectorPattern{Vector{Int}}}(undef, length(dof))
    for i in 1:length(dof)
        for j in 1:length(dof)
            if i == 1
                pattern = SparseMatrixPattern(dof[i])
            else
                pattern = SparseMatrixPattern(dof[i], dof[j])
            end
            matrix_patterns[i, j] = pattern
        end
        vector_patterns[i] = SparseVectorPattern(dof[i])
    end
    # n_matrix_entries = matrix_free ? 0 : num_entries(matrix_pattern)
    n_matrix_entries = map(num_entries, matrix_patterns)
    residual = create_field(dof)
    residual_unknowns = create_unknowns(dof)
    # stiffness_storage = zeros(n_matrix_entries)
    stiffness_storage = map(zeros, n_matrix_entries)
    return BlockSparseMatrixAssembler(
        dof, matrix_patterns, vector_patterns,
        residual, residual_unknowns,
        stiffness_storage
    )
end

function Base.show(io::IO, asm::BlockSparseMatrixAssembler)
    sz = size(asm.matrix_patterns)
    println(io, "BlockSparseMatrixAssembler:")
    println(io, "  Block layout = $(sz[1]) x $(sz[2])")
    for dof in asm.dof
        show(io, dof; pad = "  ")
        println(io)
    end
    println("  Matrix sizes:")
    for i in axes(asm.matrix_patterns, 1)
        string = "    "
        for j in axes(asm.matrix_patterns, 2)
            string = string * "($(length(asm.dof[i].unknown_dofs)), $(length(asm.dof[j].unknown_dofs)))"
            if j < size(asm.matrix_patterns, 2)
                string = string * ", "
            end
        end
        println(io, string)
    end
    # println(io, "  Block variables = ")
end

function assemble_matrix!(
    assembler::BlockSparseMatrixAssembler, func::F, Uu, p
) where F <: Function
    @assert length(assembler.dof) == 2 "Only two spaces supported currently"
    storage = assembler.residual_storage
    map(x -> fill!(x, zero(eltype(x))), storage)
    fspace = map(function_space, assembler.dof)
    X = map(coordinates, p)
    # should we do a check that all times, and time steps are consistent?
    t = current_time(p[1])
    Δt = time_step(p[1])
    U = map(x -> x.field, p)
    U_old = map(x -> x.field_old, p)

    for sol_id in 1:length(fspace)
        _update_for_assembly!(p[sol_id], assembler.dof[sol_id], Uu[BlockArrays.Block(sol_id)])
    end

    return_type = AssembledMatrix()
    conns = map(x -> x.elem_conns.data, fspace)
    coffsets = map(x -> x.elem_conns.offsets, fspace)
    physics = p[1].physics
    props = p[1].properties
    for b in 1:num_blocks(fspace[1])
        block_physics = values(physics)[b]
        ref_fe = map(x -> block_reference_element(x, b), fspace)
        num_q_pts = map(num_cell_quadrature_points, ref_fe)
        @assert all(==(num_q_pts[1]), num_q_pts)
        num_q_pts = num_q_pts[1]
        state_old = block_view(p[1].state_old, b)
        state_new = block_view(p[1].state_new, b)
        for e in 1:block_entity_size(fspace[1], b)[2]
            conn = map((r, c, co) -> connectivity(r, c, e, co[b]), ref_fe, conns, coffsets)
            out = map((r, c, x, u, u_old) -> element_level_fields(r, c, e, x, u, u_old), ref_fe, conn, X, U, U_old)
            x_el = map(x -> x[1], out)
            u_el = map(x -> x[2], out)
            u_el_old = map(x -> x[3], out)
            props_el = _element_level_properties(values(props)[b], e)
            # val_el = map((r, u) -> _element_scratch(return_type, r, u), ref_fe, U)
            nfields = length(U)

            val_el = ntuple(i -> ntuple(j->begin
                _element_scratch(
                    return_type,
                    ref_fe[i], U[i],
                    ref_fe[j], U[j]
                )
            end, nfields), nfields)
            for q in 1:num_q_pts
                interps = map(r -> _cell_interpolants(r, q), ref_fe)
                state_old_q = _quadrature_level_state(state_old, q, e)
                state_new_q = _quadrature_level_state(state_new, q, e)
                val_q = func(block_physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, state_new_q, props_el)
                # val_el = map((f, vq, ve) -> _accumulate_q_value(return_type, f, vq, ve, q, e), U, val_q, val_el)
                val_el = map(
                    (vq1, ve1) -> 
                    map((f, vq, ve) -> _accumulate_q_value(return_type, f, vq, ve, q, e), U, vq1, ve1),
                    val_q, val_el
                )
            end
            # map((f, v, c) -> _assemble_element!(f, v, c, e), U, val_el, conn, e)
            for i in 1:nfields
                for j in 1:nfields
                    _assemble_element!(assembler.stiffness_storage[i, j], val_el[i][j], conn[i], e)
                end
            end
        end
    end
end

function assemble_vector!(
    assembler::BlockSparseMatrixAssembler, func::F, Uu, p
) where F <: Function
    @assert length(assembler.dof) == 2 "Only two spaces supported currently"
    storage = assembler.residual_storage
    map(x -> fill!(x, zero(eltype(x))), storage)
    fspace = map(function_space, assembler.dof)
    X = map(coordinates, p)
    # should we do a check that all times, and time steps are consistent?
    t = current_time(p[1])
    display(t)
    Δt = time_step(p[1])
    U = map(x -> x.field, p)
    U_old = map(x -> x.field_old, p)

    for sol_id in 1:length(fspace)
        _update_for_assembly!(p[sol_id], assembler.dof[sol_id], Uu[BlockArrays.Block(sol_id)])
    end

    return_type = AssembledVector()
    conns = map(x -> x.elem_conns.data, fspace)
    coffsets = map(x -> x.elem_conns.offsets, fspace)
    physics = p[1].physics
    props = p[1].properties
    for b in 1:num_blocks(fspace[1])
        block_physics = values(physics)[b]
        ref_fe = map(x -> block_reference_element(x, b), fspace)
        num_q_pts = map(num_cell_quadrature_points, ref_fe)
        @assert all(==(num_q_pts[1]), num_q_pts)
        num_q_pts = num_q_pts[1]
        state_old = block_view(p[1].state_old, b)
        state_new = block_view(p[1].state_new, b)
        for e in 1:block_entity_size(fspace[1], b)[2]
            conn = map((r, c, co) -> connectivity(r, c, e, co[b]), ref_fe, conns, coffsets)
            out = map((r, c, x, u, u_old) -> element_level_fields(r, c, e, x, u, u_old), ref_fe, conn, X, U, U_old)
            x_el = map(x -> x[1], out)
            u_el = map(x -> x[2], out)
            u_el_old = map(x -> x[3], out)
            props_el = _element_level_properties(values(props)[b], e)
            val_el = map((r, u) -> _element_scratch(return_type, r, u), ref_fe, U)
            for q in 1:num_q_pts
                interps = map(r -> _cell_interpolants(r, q), ref_fe)
                state_old_q = _quadrature_level_state(state_old, q, e)
                state_new_q = _quadrature_level_state(state_new, q, e)
                val_q = func(block_physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, state_new_q, props_el)
                val_el = map((f, vq, ve) -> _accumulate_q_value(return_type, f, vq, ve, q, e), U, val_q, val_el)
            end
            # out = map((f, v, c) -> _assemble_element!(f, v, c, e), U, val_el, conn)
            for sol_id in 1:length(U)
                _assemble_element!(U[sol_id], val_el[sol_id], conn[sol_id], e)
            end
        end
    end
end

# this won't work with condensed right now
function update_dofs!(
    assembler::BlockSparseMatrixAssembler, dirichlet_bcs, periodic_bcs
)
    ddofs = map(dirichlet_dofs, dirichlet_bcs)
    pdofs = map(periodic_dofs, periodic_bcs)
    pdofs_side_a = map(x -> x[1], pdofs)
    pdofs_side_b = map(x -> x[2], pdofs)

    # update dof managers first
    for (n, dof) in enumerate(assembler.dof)
        update_dofs!(dof, ddofs[n], pdofs_side_a[n], pdofs_side_b[n])
    end

    # now update sparsity patterns
    for i in axes(assembler.matrix_patterns, 1)
        for j in axes(assembler.matrix_patterns, 2)
            _update_dofs!(
                assembler.matrix_patterns[i, j],
                assembler.dof[i], ddofs[i], pdofs_side_b[i],
                assembler.dof[j], ddofs[j], pdofs_side_b[j]
            )
        end
    end
end
