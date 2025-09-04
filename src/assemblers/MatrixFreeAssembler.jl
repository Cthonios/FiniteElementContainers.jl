struct MatrixFreeAssembler{
    Dof <: DofManager,
    Storage1 <: AbstractField,
    Storage2 <: AbstractArray{<:Number, 1}
} <: AbstractAssembler{Dof}
    dof::Dof
    residual_storage::Storage1
    residual_unknowns::Storage2
    stiffness_action_storage::Storage1
    stiffness_action_unknowns::Storage2
end

function MatrixFreeAssembler(dof::DofManager)

    residual_storage = create_field(dof)
    residual_unknowns = create_unknowns(dof)
    stiffness_action_storage = create_field(dof)
    stiffness_action_unknowns = create_unknowns(dof)
    return MatrixFreeAssembler(
        dof,
        residual_storage, residual_unknowns,
        stiffness_action_storage, stiffness_action_unknowns
    )
end

function MatrixFreeAssembler(u::AbstractFunction)
    dof = DofManager(u)
    return MatrixFreeAssembler(dof)
end

function update_dofs!(assembler::MatrixFreeAssembler, dirichlet_bcs)
    # collect dbcs
    if length(dirichlet_bcs) > 0
        dirichlet_dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, dirichlet_bcs)
        dirichlet_dofs = unique(sort(dirichlet_dofs))
    else
        dirichlet_dofs = Vector{Int}(undef, 0)
    end

    # update dof manager
    update_dofs!(assembler.dof, dirichlet_dofs)

    # update cached arrays in assembler
    resize!(assembler.residual_unknowns, length(assembler.dof.unknown_dofs))
    resize!(assembler.stiffness_action_unknowns, length(assembler.dof.unknown_dofs))

    return nothing
end

Base.eltype(asm::MatrixFreeAssembler) = eltype(asm.residual_storage)
Base.size(asm::MatrixFreeAssembler) = (length(asm.residual_unknowns), length(asm.residual_unknowns))

function LinearAlgebra.mul!(y::T, asm::MatrixFreeAssembler, Uu::T, p) where T
    assemble!(asm.stiffness_action_storage, hvp, Uu, p)
    copyto!(y, hvp(asm))
    return nothing
end
