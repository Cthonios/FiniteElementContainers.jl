abstract type AbstractPhysics{NP, NS} end
num_properties(::AbstractPhysics{NP, NS}) where {NP, NS} = NP
num_states(::AbstractPhysics{NP, NS}) where {NP, NS} = NS

# better define this interface
# function damping end
function damping end
function energy end
function mass end
function residual end
function stiffness end
