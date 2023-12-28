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

# no need for adapt_structure for connectivity fields since these are aliases anyway

# DofManagers
function Adapt.adapt_structure(to, dof::FiniteElementContainers.SimpleDofManager)
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  is_unknown      = Adapt.adapt_structure(to, dof.is_unknown)
  unknown_indices = Adapt.adapt_structure(to, dof.unknown_indices)
  return FiniteElementContainers.SimpleDofManager{Bool, 2, ND, NN, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

function Adapt.adapt_structure(to, dof::FiniteElementContainers.VectorizedDofManager)
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  is_unknown      = Adapt.adapt_structure(to, dof.is_unknown)
  unknown_indices = Adapt.adapt_structure(to, dof.unknown_indices)
  return FiniteElementContainers.VectorizedDofManager{Bool, 2, ND, NN, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

# Function spaces
function Adapt.adapt_structure(to, fspace::FiniteElementContainers.NonAllocatedFunctionSpace)
  ND       = num_dofs_per_node(fspace)
  conn     = Adapt.adapt_structure(to, fspace.conn)
  dof_conn = Adapt.adapt_structure(to, fspace.dof_conn)
  ref_fe   = Adapt.adapt_structure(to, fspace.ref_fe)
  return FiniteElementContainers.NonAllocatedFunctionSpace{ND, typeof(conn), typeof(dof_conn), typeof(ref_fe)}(
    conn, dof_conn, ref_fe
  )
end

# Assemblers
"""
Need to use SparseArrays.allowscalar(false)
"""
function Adapt.adapt_structure(to, assembler::FiniteElementContainers.StaticAssembler)
  I = FiniteElementContainers.int_type(assembler)
  F = FiniteElementContainers.float_type(assembler)
  R = Adapt.adapt_structure(to, assembler.R)
  K = Adapt.adapt_structure(to, assembler.K)
  return FiniteElementContainers.StaticAssembler{I, F, typeof(R), typeof(K)}(R, K)
end

end # module