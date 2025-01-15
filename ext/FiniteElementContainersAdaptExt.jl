module FiniteElementContainersAdaptExt

using Adapt
using FiniteElementContainers

function Adapt.adapt_structure(to, field::FiniteElementContainersAdaptExt.NodalField)
  NF, NN = num_fields(field), num_nodes(field)
  return FiniteElementContainers.NodalField{NF, NN}(Adapt.adapt_structure(to, field.vals))
end

# Element fields
function Adapt.adapt_structure(to, field::FiniteElementContainers.ElementField)
  NN, NE = num_nodes_per_element(field), num_elements(field)
  return FiniteElementContainers.ElementField{NN, NE}(Adapt.adapt_structure(to, field.vals))
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
function Adapt.adapt_structure(to, dof::FiniteElementContainers.DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V}
  unknown_dofs = Adapt.adapt_structure(to, dof.unknown_dofs)
  return FiniteElementContainers.DofManager{T, ND, NN, A, typeof(unknown_dofs)}(unknown_dofs)
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
function Adapt.adapt_structure(to, asm::DynamicAssembler)
  I = FiniteElementContainers.int_type(assembler)
  F = FiniteElementContainers.float_type(assembler)

  Is = Adapt.adapt_structure(to, asm.Is)
  Js = Adapt.adapt_structure(to, asm.Js)
  unknown_dofs = Adapt.adapt_structure(to, asm.unknown_dofs)
  block_sizes = Adapt.adapt_structure(to, asm.block_sizes)
  block_offsets = Adapt.adapt_structure(to, asm.block_offsets)
  residuals = Adapt.adapt_structure(to, asm.residuals)
  stiffnesses = Adapt.adapt_structure(to, asm.stiffnesses)
  masses = Adapt.adapt_structure(to, asm.masses)

  # cache arrays
  klasttouch = Adapt.adapt_structure(to, asm.klasttouch)
  csrrowptr  = Adapt.adapt_structure(to, asm.csrrowptr)
  csrcolval  = Adapt.adapt_structure(to, asm.csrcolval)
  csrnzval   = Adapt.adapt_structure(to, asm.csrnzval)

  # additional cache arrays
  csccolptr  = Adapt.adapt_structure(to, asm.csccolptr)
  cscrowval  = Adapt.adapt_structure(to, asm.cscrowval)
  cscnzval   = Adapt.adapt_structure(to, asm.cscnzval)

  return DynamicAssembler{
    F, I,
    typeof(Is), typeof(Js), 
    typeof(unknown_dofs), typeof(block_sizes), typeof(block_offsets),
    typeof(residuals), tyepof(stiffnesses), typeof(masses),
    # cache arrays
    typeof(klasttouch), typeof(csrrowptr), typeof(csrcolval), typeof(csrnzval),
    # additional cache arrays
    typeof(csccolptr), typeof(cscrowval), typeof(cscnzval)
  }(
    Is, Js, unknown_dofs, block_sizes, block_offsets, residuals, stiffnesses, masses,
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval
  )
end

function Adapt.adapt_structure(to, asm::StaticAssembler)
  I = FiniteElementContainers.int_type(asm)
  F = FiniteElementContainers.float_type(asm)

  Is = Adapt.adapt_structure(to, asm.Is)
  Js = Adapt.adapt_structure(to, asm.Js)
  unknown_dofs = Adapt.adapt_structure(to, asm.unknown_dofs)
  block_sizes = Adapt.adapt_structure(to, asm.block_sizes)
  block_offsets = Adapt.adapt_structure(to, asm.block_offsets)
  residuals = Adapt.adapt_structure(to, asm.residuals)
  stiffnesses = Adapt.adapt_structure(to, asm.stiffnesses)

  # cache arrays
  klasttouch = Adapt.adapt_structure(to, asm.klasttouch)
  csrrowptr  = Adapt.adapt_structure(to, asm.csrrowptr)
  csrcolval  = Adapt.adapt_structure(to, asm.csrcolval)
  csrnzval   = Adapt.adapt_structure(to, asm.csrnzval)

  # additional cache arrays
  csccolptr  = Adapt.adapt_structure(to, asm.csccolptr)
  cscrowval  = Adapt.adapt_structure(to, asm.cscrowval)
  cscnzval   = Adapt.adapt_structure(to, asm.cscnzval)

  return StaticAssembler{
    F, I,
    typeof(Is), typeof(Js), 
    typeof(unknown_dofs), typeof(block_sizes), typeof(block_offsets),
    typeof(residuals), typeof(stiffnesses),
    # cache arrays
    typeof(klasttouch), typeof(csrrowptr), typeof(csrcolval), typeof(csrnzval),
    # additional cache arrays
    typeof(csccolptr), typeof(cscrowval), typeof(cscnzval)
  }(
    Is, Js, unknown_dofs, block_sizes, block_offsets, residuals, stiffnesses,
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval
  )
end 

 
# # """
# # Need to use SparseArrays.allowscalar(false)
# # """
# # function Adapt.adapt_structure(to, assembler::FiniteElementContainers.StaticAssembler)
# #   I = FiniteElementContainers.int_type(assembler)
# #   F = FiniteElementContainers.float_type(assembler)
# #   R = Adapt.adapt_structure(to, assembler.R)
# #   K = Adapt.adapt_structure(to, assembler.K)
# #   return FiniteElementContainers.StaticAssembler{I, F, typeof(R), typeof(K)}(R, K)
# # end

end # module