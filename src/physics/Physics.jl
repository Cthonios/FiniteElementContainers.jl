abstract type AbstractPhysics{NF, NP, NS} end
num_fields(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NF
num_properties(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NP
num_states(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NS

# physics like methods
function damping end
function energy end
function mass end
function residual end
function stiffness end

# optimization like methods
function gradient end
function hessian end
function value end
