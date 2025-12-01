abstract type AbstractIntegral{
    A         <: AbstractAssembler,
    Cache     <: Union{<:NamedTuple, <:AbstractField},
    Integrand <: Function
} end

function integrate end

struct ScalarIntegral{A, C, I} <: AbstractIntegral{A, C, I}
    assembler::A
    cache::C
    integrand::I
end

function ScalarIntegral(asm, integrand)
    fspace = function_space(asm.dof)
    cache = Matrix{Float64}[]
    for (key, val) in pairs(fspace.ref_fes)
        NQ = ReferenceFiniteElements.num_quadrature_points(val)
        NE = size(getfield(fspace.elem_conns, key), 2)
        field = L2ElementField(zeros(Float64, NQ, NE))
        push!(cache, field)
    end
    cache = NamedTuple{keys(fspace.ref_fes)}(tuple(scalar_quadarature_storage...))
    return ScalarIntegral(asm, cache, integrand)
end

function integrate(integral::ScalarIntegral, U, p)
    cache, dof = integral.cache, integral.assembler.dof
    func = integral.integrand
    assemble_quadrature_quantity!(cache, dof, func, U, p)
    return mapreduce(sum, sum, values(cache))
end
