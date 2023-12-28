module FiniteElementContainersAdaptExt

using Adapt
using FiniteElementContainers
using StructArrays

# Nodal fields
function Adapt.adapt_structure(to, field::FiniteElementContainers.SimpleNodalField)
  NF, NN = num_fields(field), num_nodes(field)
  return FiniteElementContainers.SimpleNodalField{NF, NN}(Adapt.adapt_structure(to, field.vals))
end

function Adapt.adapt_structure(to, field::FiniteElementContainers.VectorizedNodalField)
  NF, NN = num_fields(field), num_nodes(field)
  return FiniteElementContainers.VectorizedNodalField{NF, NN}(Adapt.adapt_structure(to, field.vals))
end

# Element fields
function Adapt.adapt_structure(to, field::FiniteElementContainers.SimpleElementField)
  NN, NE = num_nodes_per_element(field), num_elements(field)
  return FiniteElementContainers.SimpleElementField{NN, NE}(Adapt.adapt_structure(to, field.vals))
end

function Adapt.adapt_structure(to, field::FiniteElementContainers.VectorizedElementField)
  NN, NE = num_nodes_per_element(field), num_elements(field)
  return FiniteElementContainers.VectorizedElementField{NN, NE}(Adapt.adapt_structure(to, field.vals))
end

# Quadrature fields
function Adapt.adapt_structure(to, field::FiniteElementContainers.SimpleQuadratureField)
  NF, NQ, NE = num_fields(field), num_q_points(field), num_elements(field)
  return FiniteElementContainers.SimpleQuadratureField{NF, NQ, NE}(Adapt.adapt_structure(to, field.vals))
end

function Adapt.adapt_structure(to, field::FiniteElementContainers.VectorizedQuadratureField)
  NF, NQ, NE = num_fields(field), num_q_points(field), num_elements(field)
  return FiniteElementContainers.VectorizedQuadratureField{NF, NQ, NE}(Adapt.adapt_structure(to, field.vals))
end

end # module