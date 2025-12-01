struct SparseMatrixAssembler{
    Condensed,
    IV  <: AbstractArray{Int, 1},
    RV  <: AbstractArray{Float64, 1},
    Var <: AbstractFunction
} <: AbstractAssembler{DofManager{Condensed, Int, IV, Var}}
    constraint_storage::RV
    dof::DofManager{Condensed, Int, IV, Var}
    matrix_pattern::SparseMatrixPattern{IV, BlockPattern, RV}
    vector_pattern::SparseVectorPattern{IV}
end