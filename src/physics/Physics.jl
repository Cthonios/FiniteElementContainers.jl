abstract type AbstractPhysics{NF, NP, NS} end
num_fields(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NF
num_properties(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NP
num_states(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NS

# better define this interface
function damping end
function energy end
function gradient end
function hessian end
function mass end
function residual end
function stiffness end
