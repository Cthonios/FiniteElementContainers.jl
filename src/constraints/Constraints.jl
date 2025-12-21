abstract type AbstractConstraint end
abstract type AbstractConstraintContainer end

# """
# Input will be formulated as

# c_1 * u_1 + c_2 * u_2 + ... + c_n * u_n = 0

# """
struct LinearConstraint{
    RV <: AbstractVector{<:Number},
    RI <: AbstractVector{<:Integer}
} <: AbstractConstraint
    coefficients::RV
    dofs::RI

    function LinearConstraint(coefficients, dofs)
        @assert length(coefficients) == length(dofs)
        new{typeof(coefficients), typeof(dofs)}(coefficients, dofs)
    end
end

function evaluate(cons::LinearConstraint, u)
    val = zero(eltype(u))
    for n in axes(cons.dofs, 1)
        val += cons.coefficients[n] * u[cons.dofs[n]]
    end
    return val
end
