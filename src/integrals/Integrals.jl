abstract type AbstractIntegral{
    A <: AbstractAssembler,
    C,
    I <: Function
} end

function Adapt.adapt_storage(to, int::AbstractIntegral)
    return eval(typeof(int).name.name)(
        adapt(to, int.assembler),
        adapt(to, int.cache),
        int.integrand
    )
end

KA.get_backend(int::AbstractIntegral) = KA.get_backend(int.cache)

function integrate end

function (integral::AbstractIntegral)(U, p)
    return integrate(integral, U, p)
end

struct ScalarIntegral{A, C <: NamedTuple, I} <: AbstractIntegral{A, C, I}
    assembler::A
    cache::C
    integrand::I
end

function ScalarIntegral(asm, integrand)
    cache = create_assembler_cache(asm, AssembledScalar())
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
    assemble_quadrature_quantity!(cache, dof, nothing, func, U, p)
    return mapreduce(sum, sum, values(cache))
end

struct MatrixIntegral{A, C <: AbstractVector, I} <: AbstractIntegral{A, C, I}
    assembler::A
    cache::C
    integrand::I
end

function MatrixIntegral(asm, integrand)
    cache = create_assembler_cache(asm, AssembledMatrix())
    return MatrixIntegral(asm, cache, integrand)
end

function integrate(integral::MatrixIntegral, U, p)
    assemble_matrix!(
        integral.cache, 
        integral.assembler.matrix_pattern,
        integral.assembler.dof,
        integral.integrand,
        U, p
    )
    return SparseArrays.sparse!(
        integral.assembler.matrix_pattern,
        integral.cache
    )
end

function remove_fixed_dofs!(integral::MatrixIntegral)
    backend = KA.get_backend(integral)

    # TODO change API so we don't need to
    # bring this guy to life
    A = SparseArrays.sparse!(
        integral.assembler.matrix_pattern,
        integral.cache
    )
    _adjust_matrix_entries_for_constraints!(
        A, integral.assembler.constraint_storage, backend
    )
    return nothing
end

struct VectorIntegral{A, C <: AbstractField, I} <: AbstractIntegral{A, C, I}
    assembler::A
    cache::C
    integrand::I
end

function VectorIntegral(asm, integrand)
    cache = create_assembler_cache(asm, AssembledVector())
    return VectorIntegral(asm, cache, integrand)
end

function integrate(integral::VectorIntegral, U, p)
    assemble_vector!(
        integral.cache, 
        integral.assembler.vector_pattern,
        integral.assembler.dof,
        integral.integrand,
        U, p
    )
    return integral.cache
end

function remove_fixed_dofs!(integral::VectorIntegral)
    backend = KA.get_backend(integral)
    _adjust_vector_entries_for_constraints!(
        integral.cache, integral.assembler.constraint_storage, backend
    )
    return nothing
end
