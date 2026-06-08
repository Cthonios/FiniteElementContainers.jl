function _default_residual_method(use_inplace_methods::Bool)
    if _use_inplace_methods(solver.assembler)
        return residual
    else
        return residual!
    end
end

function _default_stiffness_method(solver::AbstractLinearSolver)
    if _use_inplace_methods(solver.assembler)
        return stiffness
    else
        return stiffness!
    end
end

abstract type AbstractSolverSettings end
abstract type AbstractSolver end
# TODO
# abstract type AbstractPreconditioner end

abstract type AbstractLinearSolverSettings <: AbstractSolverSettings end
abstract type AbstractLinearSolver{
    A <: AbstractAssembler,
    P,
    S <: AbstractLinearSolverSettings,
    U <: AbstractVector{<:Number}
} <: AbstractSolver
end

# interface to define
function solve! end

function KA.get_backend(solver::AbstractLinearSolver)
    return KA.get_backend(solver.assembler)
end

struct DirectLinearSolverSettings <: AbstractLinearSolverSettings
end

struct DirectLinearSolver{
    A, P, U
} <: AbstractLinearSolver{A, P, DirectLinearSolverSettings, U}
    assembler::A
    preconditioner::P
    settings::DirectLinearSolverSettings
    timer::TimerOutput
    # TODO add some tolerances
    # what's the best way to do this with general solvers?
    ΔUu::Inc

    function DirectLinearSolver(assembler::SparseMatrixAssembler)
        preconditioner = I
        ΔUu = similar(assembler.residual_unknowns)
        fill!(ΔUu, zero(eltype(ΔUu)))
        new{typeof(assembler), typeof(preconditioner), typeof(ΔUu)}(
            assembler, preconditioner, 
            DirectLinearSolverSettings(), 
            TimerOutput(), ΔUu
        )
    end 
end
  
function solve!(solver::DirectLinearSolver, Uu, p)
    # if _use_inplace_methods(solver.assembler)
    #     residual_method = residual!
    #     stiffness_method = stiffness!
    # else
    #     residual_method = residual
    #     stiffness_method = stiffness
    # end
    residual_method = _default_residual_method(solver)
    stiffness_method = _default_stiffness_method(solver)
    assemble_vector!(solver.assembler, residual_method, Uu, p)
    assemble_vector_source!(solver.assembler, Uu, p)
    assemble_vector_neumann_bc!(solver.assembler, Uu, p)
    # assemble_vector_robin_bc!(solver.assembler, Uu, p)
    assemble_stiffness!(solver.assembler, stiffness_method, Uu, p)
    R = residual(solver.assembler)
    K = stiffness(solver.assembler)
    # TODO specialize to backend solvers if they exists
    # solver.ΔUu .= -K \ R
    copyto!(solver.ΔUu, -K \ R) # currently doesn't work on GPU
    # update_field_unknowns!(p.field, solver.assembler.dof, solver.ΔUu, +)
    map!((x, y) -> x + y, Uu, Uu, solver.ΔUu)
    return nothing
end

struct IterativeLinearSolverSettings
    # TODO add tolerances and what not.
end

struct IterativeLinearSolver{
    A, P, U, W
} <: AbstractLinearSolver{A, P, IterativeLinearSolverSettings, U}
    assembler::A
    preconditioner::P
    settings::IterativeLinearSolverSettings
    timer::TimerOutput
    ΔUu::U
    workspace::W

    function IterativeLinearSolver(assembler::SparseMatrixAssembler, solver_sym)
        # TODO
        preconditioner = I
        K = stiffness(assembler)
        R = residual(assembler)
        ΔUu = similar(R, axes(K, 1))
        fill!(ΔUu, zero(eltype(ΔUu)))
        workspace = krylov_workspace(Val(solver_sym), K, R)
        return new{typeof(assembler), typeof(preconditioner), typeof(ΔUu), typeof(workspace)}(
            assembler, preconditioner, 
            IterativeLinearSolverSettings(),
            TimerOutput(), ΔUu, workspace
        )
    end
end
  
  # TODO specialize for operator like assemblers
function solve!(solver::IterativeLinearSolver, Uu, p)
    asm = solver.assembler
    # if _use_inplace_methods(asm)
    #   residual_method = residual!
    #   stiffness_method = stiffness!
    # else
    #   residual_method = residual
    #   stiffness_method = stiffness
    # end
    residual_method = _default_residual_method(solver)
    stiffness_method = _default_stiffness_method(solver)
    # assemble relevant fields
    @timeit solver.timer "residual assembly" begin
        assemble_vector!(asm, residual_method, Uu, p)
        assemble_vector_source!(asm, Uu, p)
        assemble_vector_neumann_bc!(asm, Uu, p)
        # assemble_vector_robin_bc!(asm, Uu, p)
    end
    @timeit solver.timer "stiffness assembly" begin
        assemble_stiffness!(asm, stiffness_method, Uu, p)
    end
    # solve and fetch solution
    @timeit solver.timer "solve" begin
        krylov_solve!(solver.workspace, stiffness(asm), residual(asm))
    end
    @timeit solver.timer "update solution" begin
        ΔUu = -Krylov.solution(solver.workspace)
        # make necessary copies and updates
        copyto!(solver.ΔUu, ΔUu)
        map!((x, y) -> x + y, Uu, Uu, ΔUu)
    end
    return nothing
end

abstract type AbstractNonLinearSolverSettings <: AbstractSolverSettings end
abstract type AbstractNonLinearSolver{
    L <: AbstractLinearSolver,
    S <: AbstractNonLinearSolverSettings
} <: AbstractSolver
end

# interface to define
function solve! end

struct NewtonSolverSettings{T <: Number, CB}
    max_iters::Int
    abs_increment_tol::T
    abs_residual_tol::T
    rel_residual_tol::T
    # Optional callback: (iter, norm_ΔUu, norm_R, rel_R, converged) -> nothing
    # Called after each Newton iteration. Set to `nothing` to disable.
    log_callback::CB

    function NewtonSolverSettings()
        new{Float64, Nothing}(10, 1e-12, 1e-12, 1e-12, nothing)
    end
end

struct NewtonSolver{
    T, L
} <: AbstractNonLinearSolver{L, NewtonSolverSettings{T}}
    linear_solver::L
    settings::NewtonSolverSettings{T}
    timer::TimerOutput

    function NewtonSolver(linear_solver::AbstractLinearSolver)
        settings = NewtonSolverSettings()
        timer = TimerOutput()
        new{Float64, typeof(linear_solver)}(linear_solver, settings, timer)
    end
end

function solve!(solver::NewtonSolver, Uu, p)
    settings = solver.settings
    @timeit solver.timer "Nonlinear solver" begin
        initial_norm = Ref(0.0)
        for n in 1:settings.max_iters
            @timeit solver.timer "Linear solver" begin
                solve!(solver.linear_solver, Uu, p)
            end

            @timeit solver.timer "convergence check" begin
                norm_ΔUu = sqrt(sum(abs2, solver.linear_solver.ΔUu))
                norm_R   = sqrt(sum(abs2, residual(solver.linear_solver.assembler)))
                if n == 1
                    initial_norm[] = norm_R
                end
                rel_R     = initial_norm[] > 0.0 ? norm_R / initial_norm[] : norm_R
                converged = norm_ΔUu < settings.abs_increment_tol ||
                            norm_R   < settings.abs_residual_tol   ||
                            rel_R    < settings.rel_residual_tol
                @debug "Newton" n norm_ΔUu norm_R rel_R
                if !isnothing(solver.log_callback)
                    solver.log_callback(n, norm_ΔUu, norm_R, rel_R, converged)
                end
                converged && break
            end
        end
    end
end
