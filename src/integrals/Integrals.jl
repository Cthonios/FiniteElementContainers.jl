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

function Base.show(io::IO, int::AbstractIntegral)
    println(io, typeof(int).name.name, ":")
    println(io, "  Assembler type     = $(typeof(int.assembler))")
    println(io, "  Cache type         = $(typeof(int.cache))")
    println(io, "  Integrand function = $(int.integrand)")
end

KA.get_backend(int::AbstractIntegral) = KA.get_backend(int.cache)

function create_field(int::AbstractIntegral)
    return create_field(int.assembler)
end

function create_unknowns(int::AbstractIntegral)
    return create_unknowns(int.assembler)
end

function integrate end

function (integral::AbstractIntegral)(U, p)
    return integrate(integral, U, p)
end

# abstract type AbstractIntegral{A, C, I} <: AbstractIntegral{A, C, I} end
# abstract type AbstractBoundaryIntegral{A, C, I} <: AbstractIntegral{A, C, I} end

abstract type AbstractMatrixIntegral{A, C, I} <: AbstractIntegral{A, C, I} end
abstract type AbstractScalarIntegral{A, C, I} <: AbstractIntegral{A, C, I} end
abstract type AbstractVectorIntegral{A, C, I} <: AbstractIntegral{A, C, I} end

struct ScalarCellIntegral{A, C <: NamedTuple, I} <: AbstractIntegral{A, C, I}
    assembler::A
    cache::C
    integrand::I
end

function ScalarCellIntegral(asm, integrand)
    cache = create_assembler_cache(asm, AssembledScalar())
    return ScalarCellIntegral(asm, cache, integrand)
end

function gradient(integral::ScalarCellIntegral)
    func(physics, interps, x, t, dt, u, u_n, state_old, state_new, props) = 
        ForwardDiff.gradient(
            z -> integral.integrand(physics, interps, x, t, dt, z, u_n, state_old, state_new, props),
            u
        )
    cache = create_field(integral.assembler)
    return VectorCellIntegral(integral.assembler, cache, func)
end

function hessian(integral::ScalarCellIntegral)
    func(physics, interps, x, t, dt, u, u_n, state_old, state_new, props) = 
        ForwardDiff.hessian(
            z -> integral.integrand(physics, interps, x, t, dt, z, u_n, state_old, state_new, props),
            u
        )
    cache = create_assembler_cache(asm, AssembledMatrix())
    return MatrixCellIntegral(integral.assembler, cache, func)
end

function integrate(integral::ScalarCellIntegral, U, p)
    cache, dof = integral.cache, integral.assembler.dof
    func = integral.integrand
    assemble_quadrature_quantity!(cache, dof, nothing, func, U, p)
    return mapreduce(sum, sum, values(cache))
end

struct MatrixCellIntegral{A, C <: AbstractVector, I} <: AbstractIntegral{A, C, I}
    assembler::A
    cache::C
    integrand::I
end

function MatrixCellIntegral(asm, integrand)
    cache = create_assembler_cache(asm, AssembledMatrix())
    return MatrixCellIntegral(asm, cache, integrand)
end

function integrate(integral::MatrixCellIntegral, U, p)
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

function remove_fixed_dofs!(integral::MatrixCellIntegral)
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

struct VectorCellIntegral{A, C <: AbstractField, I} <: AbstractIntegral{A, C, I}
    assembler::A
    cache::C
    integrand::I
end

function VectorCellIntegral(asm, integrand)
    cache = create_assembler_cache(asm, AssembledVector())
    return VectorCellIntegral(asm, cache, integrand)
end

function integrate(integral::VectorCellIntegral, U, p)
    assemble_vector!(
        integral.cache, 
        integral.assembler.vector_pattern,
        integral.assembler.dof,
        integral.integrand,
        U, p
    )
    return integral.cache
end

function remove_fixed_dofs!(integral::VectorCellIntegral)
    backend = KA.get_backend(integral)
    _adjust_vector_entries_for_constraints!(
        integral.cache, integral.assembler.constraint_storage, backend
    )
    return nothing
end
