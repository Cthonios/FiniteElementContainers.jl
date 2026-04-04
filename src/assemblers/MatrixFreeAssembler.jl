struct MatrixFreeAssembler{
    Condensed,
    NumArrDims,
    IV           <: AbstractArray{Int, 1},
    RV           <: AbstractArray{Float64, 1},
    Var          <: AbstractFunction,
    FieldStorage <: AbstractField{Float64, NumArrDims, RV}
} <: AbstractAssembler{DofManager{Condensed, Int, IV, Var}}
    dof::DofManager{Condensed, Int, IV, Var}
    vector_pattern::SparseVectorPattern{IV}
    constraint_storage::RV
    residual_storage::FieldStorage
    residual_unknowns::RV
    stiffness_action_storage::FieldStorage
    stiffness_action_unknowns::RV
end

function MatrixFreeAssembler(dof::DofManager)
    vector_pattern = SparseVectorPattern(dof)

    ND, NN = size(dof)
    n_total_dofs = ND * NN
    constraint_storage = zeros(n_total_dofs)
    constraint_storage[dof.dirichlet_dofs] .= 1.
  
    residual_storage = create_field(dof)
    residual_unknowns = create_unknowns(dof)
    stiffness_action_storage = create_field(dof)
    stiffness_action_unknowns = create_unknowns(dof)
    return MatrixFreeAssembler(
        dof, vector_pattern,
        constraint_storage,
        residual_storage, residual_unknowns,
        stiffness_action_storage, stiffness_action_unknowns
    )
end

function MatrixFreeAssembler(u::AbstractFunction)
    dof = DofManager(u)
    return MatrixFreeAssembler(dof)
end

function Adapt.adapt_structure(to, asm::MatrixFreeAssembler)
    return MatrixFreeAssembler(
        adapt(to, asm.dof),
        adapt(to, asm.vector_pattern),
        adapt(to, asm.constraint_storage),
        adapt(to, asm.residual_storage),
        adapt(to, asm.residual_unknowns),
        adapt(to, asm.stiffness_action_storage),
        adapt(to, asm.stiffness_action_unknowns)
    )
end

function update_dofs!(assembler::MatrixFreeAssembler, dirichlet_bcs::DirichletBCs)
    # collect dbcs
    if length(dirichlet_bcs) > 0
        ddofs = dirichlet_dofs(dirichlet_bcs)
    else
        ddofs = Vector{Int}(undef, 0)
    end

    # update dof manager
    update_dofs!(assembler.dof, ddofs)

    # update cached arrays in assembler
    if use_condensed
        assembler.constraint_storage[assembler.dof.unknown_dofs] .= 0.
        assembler.constraint_storage[assembler.dof.dirichlet_dofs] .= 1.
    else
        resize!(assembler.residual_unknowns, length(assembler.dof.unknown_dofs))
        resize!(assembler.stiffness_action_unknowns, length(assembler.dof.unknown_dofs))
        _update_dofs!(assembler.vector_pattern, assembler.dof, ddofs)
    end

    return nothing
end

Base.eltype(asm::MatrixFreeAssembler) = eltype(asm.residual_storage)
Base.size(asm::MatrixFreeAssembler) = (length(asm.residual_unknowns), length(asm.residual_unknowns))

function LinearAlgebra.mul!(y::T, asm::MatrixFreeAssembler, Uu::T, p) where T
    assemble!(asm.stiffness_action_storage, hvp, Uu, p)
    copyto!(y, hvp(asm))
    return nothing
end
