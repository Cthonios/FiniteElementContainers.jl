abstract type AbstractBlockAssembler <: AbstractAssembler end

function create_field(asm::AbstractBlockAssembler)
    return create_field(asm.dof)
end

function create_unknowns(asm::AbstractBlockAssembler)
    return create_unknowns(asm.dof)
end

mutable struct BlockSparseMatrixAssembler{
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
            matrix_patterns[i, j] = SparseMatrixPattern(dof[i], dof[j])
        end
        vector_patterns[i] = SparseVectorPattern(dof[i])
    end
    n_matrix_entries = map(num_entries, matrix_patterns)
    residual = create_field(dof)
    residual_unknowns = create_unknowns(dof)
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

function assemble_stiffness!(
    assembler::BlockSparseMatrixAssembler, func::F, Uu, p
) where F <: Function
    @assert length(assembler.dof) == 2 "Only two spaces supported currently"
    # storage = assembler.residual_storage
    storage = assembler.stiffness_storage
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
    # conns = map(block_conns, fspace)
    coffsets = map(x -> x.elem_conns.offsets, fspace)
    physics = p[1].physics
    props = p[1].properties
    for b in 1:num_blocks(fspace[1])
        block_physics = values(physics)[b]
        ref_fe = map(x -> block_reference_element(x, b), fspace)
        num_q_pts = map(num_cell_quadrature_points, ref_fe)
        @assert all(==(num_q_pts[1]), num_q_pts)
        num_q_pts = num_q_pts[1]
        for e in 1:block_entity_size(fspace[1], b)[2]
            conn = map((r, c, co) -> connectivity(r, c, e, co[b]), ref_fe, conns, coffsets)
            out = map((r, c, x, u, u_old) -> element_level_fields(r, c, x, u, u_old), ref_fe, conn, X, U, U_old)
            x_el = map(x -> x[1], out)
            u_el = map(x -> x[2], out)
            u_el_old = map(x -> x[3], out)
            props_el = properties(props, e, b)
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
                state_old_q = state_variables(p[1].state_old, q, e, b)
                state_new_q = state_variables(p[1].state_new, q, e, b)
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
    Δt = time_step(p[1])
    U = map(x -> x.field, p)
    U_old = map(x -> x.field_old, p)

    # this allocates a bit
    for sol_id in 1:length(fspace)
        _update_for_assembly!(p[sol_id], assembler.dof[sol_id], view(Uu, BlockArrays.Block(sol_id)))
    end

    return_type = AssembledVector()
    conns = map(x -> x.elem_conns.data, fspace)
    coffsets = map(x -> x.elem_conns.offsets, fspace)
    physics = p[1].physics
    props = p[1].properties
    state_old = p[1].state_old
    state_new = p[1].state_new
    for b in 1:num_blocks(fspace[1])
        block_physics = values(physics)[b]
        ref_fe = map(x -> block_reference_element(x, b), fspace)
        num_q_pts = map(num_cell_quadrature_points, ref_fe)
        @assert all(==(num_q_pts[1]), num_q_pts)
        num_q_pts = num_q_pts[1]
        for e in 1:block_entity_size(fspace[1], b)[2]
            conn = map((r, c, co) -> connectivity(r, c, e, co[b]), ref_fe, conns, coffsets)
            out = map((r, c, x, u, u_old) -> element_level_fields(r, c, x, u, u_old), ref_fe, conn, X, U, U_old)
            x_el = map(x -> x[1], out)
            u_el = map(x -> x[2], out)
            u_el_old = map(x -> x[3], out)
            props_el = properties(props, e, b)
            val_el = map((r, u) -> _element_scratch(return_type, r, u), ref_fe, U)
            for q in 1:num_q_pts
                interps = map(r -> _cell_interpolants(r, q), ref_fe)
                state_old_q = state_variables(state_old, q, e, b)
                state_new_q = state_variables(state_new, q, e, b)
                val_q = func(block_physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, state_new_q, props_el)
                val_el = map((f, vq, ve) -> _accumulate_q_value(return_type, f, vq, ve, q, e), U, val_q, val_el)
            end
            for sol_id in 1:length(U)
                _assemble_element!(assembler.residual_storage[sol_id], val_el[sol_id], conn[sol_id], e)
            end
        end
    end
end

function residual(asm::BlockSparseMatrixAssembler)
    for (b, (d, s)) in enumerate(zip(asm.dof, asm.residual_storage))
        extract_field_unknowns!(view(asm.residual_unknowns, BlockArrays.Block(b)), d, s)
    end
    return asm.residual_unknowns
end

function stiffness(asm::BlockSparseMatrixAssembler)
    ndofs_each = map(x -> length(x.unknown_dofs), asm.dof) |> collect
    ndofs = reduce(+, ndofs_each)
    K = BlockArray(spzeros(ndofs, ndofs), ndofs_each, ndofs_each)
    for i in axes(asm.stiffness_storage, 1)
        for j in axes(asm.stiffness_storage, 1)
            pat = asm.matrix_patterns[i, j]
            vals = asm.stiffness_storage[i, j][pat.unknown_dofs]
            temp = sparse(pat.Is, pat.Js, vals)
            # display(temp)
            # K[BlockArrays.Block(i, j)] = _sparse_matrix_stiffness()
            K[BlockArrays.Block(i, j)] = temp
        end
    end
    return K
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

    # update size of residual unknowns
    assembler.residual_unknowns = create_unknowns(assembler.dof)
    return nothing
end

_use_inplace_methods(::BlockSparseMatrixAssembler) = false
