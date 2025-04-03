abstract type AbstractSolver end
abstract type AbstractLinearSolver <: AbstractSolver end
abstract type AbstractNonLinearSolver <: AbstractSolver end
abstract type AbstractScalarSolver <: AbstractSolver end

# traditional solver interface
function preconditioner end
function residual end
function tangent end

# optimization based solver interface
function gradient end
function hessian end
function value end
