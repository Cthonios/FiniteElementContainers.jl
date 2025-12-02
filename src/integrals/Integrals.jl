abstract type AbstractIntegral{
    A         <: AbstractAssembler,
    Integrand <: Function
} end

function integrate end

struct ScalarIntegral{A, I, C <: NamedTuple} <: AbstractIntegral{A, I}
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
    cache = NamedTuple{keys(fspace.ref_fes)}(tuple(cache...))
    return ScalarIntegral(asm, cache, integrand)
end

function gradient(integral::ScalarIntegral)
    func(physics, interps, x, t, dt, u, u_n, state_old, state_new, props) = ForwardDiff.gradient(
        z -> integral.integrand(physics, interps, x, t, dt, z, u_n, state_old, state_new, props),
        u
    )
    cache = create_field(integral.assembler)
    return VectorIntegral(integral.assembler, cache, func)
end

function integrate(integral::ScalarIntegral, U, p)
    cache, dof = integral.cache, integral.assembler.dof
    func = integral.integrand
    assemble_quadrature_quantity!(cache, dof, func, U, p)
    return mapreduce(sum, sum, values(cache))
end

struct VectorIntegral{A, I, C <: AbstractField} <: AbstractIntegral{A, I}
    assembler::A
    cache::C
    integrand::I
end

function VectorIntegral(asm, integrand)
    cache = create_field(asm)
    return VectorIntegral(asm, cache, integrand)
end

function integrate(integral::VectorIntegral, U, p)
    cache, dof = integral.cache, integral.assembler.dof
    func = integral.integrand
    assemble_vector!(cache, dof, func, U, p)
    return cache
end
