module BlockArraysExt

using BlockArrays
using FiniteElementContainers

function FiniteElementContainers.create_unknowns(assemblers::NamedTuple)
    Uus = map(create_unknowns, collect(values(assemblers)))
    return mortar(Uus)
end

end # module
