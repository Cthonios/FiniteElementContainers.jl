module FiniteElementContainersAdaptExt

using Adapt
using FiniteElementContainers

# Assemblers
function Adapt.adapt_structure(to, asm::SparseMatrixAssembler)
  dof = Adapt.adapt_structure(to, asm.dof)
  pattern = Adapt.adapt_structure(to, asm.pattern)
  constraint_storage = Adapt.adapt_structure(to, asm.constraint_storage)
  stiffness_storage = Adapt.adapt_structure(to, asm.stiffness_storage)
  return SparseMatrixAssembler(dof, pattern, constraint_storage, stiffness_storage)
end

function Adapt.adapt_structure(to, asm::FiniteElementContainers.SparsityPattern)
  Is = Adapt.adapt_structure(to, asm.Is)
  Js = Adapt.adapt_structure(to, asm.Js)
  unknown_dofs = Adapt.adapt_structure(to, asm.unknown_dofs)
  block_sizes = Adapt.adapt_structure(to, asm.block_sizes)
  block_offsets = Adapt.adapt_structure(to, asm.block_offsets)
  #
  klasttouch = Adapt.adapt_structure(to, asm.klasttouch)
  csrrowptr = Adapt.adapt_structure(to, asm.csrrowptr)
  csrcolval = Adapt.adapt_structure(to, asm.csrcolval)
  csrnzval = Adapt.adapt_structure(to, asm.csrnzval)
  #
  csccolptr = Adapt.adapt_structure(to, asm.csccolptr)
  cscrowval = Adapt.adapt_structure(to, asm.cscrowval)
  cscnzval = Adapt.adapt_structure(to, asm.cscnzval)
  return FiniteElementContainers.SparsityPattern(
    Is, Js,
    unknown_dofs, block_sizes, block_offsets,
    klasttouch, csrrowptr, csrcolval, csrnzval,
    csccolptr, cscrowval, cscnzval
  )
end

# DofManagers
function Adapt.adapt_structure(to, dof::NewDofManager{T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars}) where {T, IDs, H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars}
  H1_bc_dofs = Adapt.adapt_structure(to, dof.H1_bc_dofs)
  H1_unknown_dofs = Adapt.adapt_structure(to, dof.H1_unknown_dofs)
  Hcurl_bc_dofs = Adapt.adapt_structure(to, dof.Hcurl_bc_dofs)
  Hcurl_unknown_dofs = Adapt.adapt_structure(to, dof.Hcurl_unknown_dofs)
  Hdiv_bc_dofs = Adapt.adapt_structure(to, dof.Hdiv_bc_dofs)
  Hdiv_unknown_dofs = Adapt.adapt_structure(to, dof.Hdiv_unknown_dofs)
  L2_element_dofs = Adapt.adapt_structure(to, dof.L2_element_dofs)
  L2_quadrature_dofs = Adapt.adapt_structure(to, dof.L2_quadrature_dofs)
  H1_vars = Adapt.adapt_structure(to, dof.H1_vars)
  Hcurl_vars = Adapt.adapt_structure(to, dof.Hcurl_vars)
  Hdiv_vars = Adapt.adapt_structure(to, dof.Hdiv_vars)
  L2_element_vars = Adapt.adapt_structure(to, dof.L2_element_vars)
  L2_quadrature_vars = Adapt.adapt_structure(to, dof.L2_quadrature_vars)
  return NewDofManager{T, NH1, NHcurl, NHdiv, typeof(H1_bc_dofs), typeof(vars)}(
    H1_bc_dofs, H1_unknown_dofs,
    Hcurl_bc_dofs, Hcurl_unknown_dofs,
    Hdiv_bc_dofs, Hdiv_unknown_dofs,
    L2_element_dofs, L2_quadrature_dofs,
    H1_vars, Hcurl_vars, Hdiv_vars, L2_element_vars, L2_quadrature_vars
  )
end

# Fields
function Adapt.adapt_structure(to, field::L2ElementField{T, NF, V, S}) where {T, NF, V, S}
  vals = Adapt.adapt_structure(to, field.vals)
  return L2ElementField{T, NF, typeof(vals), S}(vals)
end

function Adapt.adapt_structure(to, field::H1Field{T, NF, V, S}) where {T, NF, V, S}
  vals = Adapt.adapt_structure(to, field.vals)
  return H1Field{T, NF, typeof(vals), S}(vals)
end

# Function spaces
function Adapt.adapt_structure(to, fspace::FunctionSpace)
  coords = Adapt.adapt_structure(to, fspace.coords)
  elem_conns = Adapt.adapt_structure(to, fspace.elem_conns)
  elem_id_maps = Adapt.adapt_structure(to, fspace.elem_id_maps)
  fspace_type = Adapt.adapt_structure(to, fspace.fspace_type)
  ref_fes = Adapt.adapt_structure(to, fspace.ref_fes)
  sideset_elems = Adapt.adapt_structure(to, fspace.sideset_elems)
  sideset_nodes = Adapt.adapt_structure(to, fspace.sideset_nodes)
  sideset_sides = Adapt.adapt_structure(to, fspace.sideset_sides)
  return FunctionSpace(
    coords, 
    elem_conns, elem_id_maps, 
    fspace_type, 
    ref_fes,
    sideset_elems, sideset_nodes, sideset_sides
  )
end

# # Mesh - won't work, symbols are mutable
# # function Adapt.adapt_structure(to, mesh::UnstructuredMesh)
# #   nodal_coords = Adapt.adapt_structure(to, mesh.nodal_coords)
# #   element_block_names = Adapt.adapt_structure(to, mesh.element_block_names)
# #   element_types = Adapt.adapt_structure(to, mesh.element_types)
# #   element_conns = Adapt.adapt_structure(to, mesh.element_conns)
# #   element_id_maps = Adapt.adapt_structure(to, mesh.element_id_maps)
# #   nodeset_nodes = Adapt.adapt_structure(to, mesh.nodeset_nodes)
# #   return UnstructuredMesh(
# #     nodal_coords, element_block_names,
# #     element_types, element_conns,
# #     element_id_maps, nodeset_nodes
# #   )
# # end

# # Variables
function Adapt.adapt_structure(to, var::ScalarFunction)
  syms = names(var)
  fspace = Adapt.adapt_structure(to, var.fspace)
  return ScalarFunction{syms, typeof(fspace)}(fspace)
end

function Adapt.adapt_structure(to, var::SymmetricTensorFunction)
  syms = names(var)
  fspace = Adapt.adapt_structure(to, var.fspace)
  return SymmetricTensorFunction{syms, typeof(fspace)}(fspace)
end

function Adapt.adapt_structure(to, var::TensorFunction)
  syms = names(var)
  fspace = Adapt.adapt_structure(to, var.fspace)
  return TensorFunction{syms, typeof(fspace)}(fspace)
end

function Adapt.adapt_structure(to, var::VectorFunction)
  syms = names(var)
  fspace = Adapt.adapt_structure(to, var.fspace)
  return VectorFunction{syms, typeof(fspace)}(fspace)
end
 
# # # # """
# # # # Need to use SparseArrays.allowscalar(false)
# # # # """
# # # # function Adapt.adapt_structure(to, assembler::FiniteElementContainers.StaticAssembler)
# # # #   I = FiniteElementContainers.int_type(assembler)
# # # #   F = FiniteElementContainers.float_type(assembler)
# # # #   R = Adapt.adapt_structure(to, assembler.R)
# # # #   K = Adapt.adapt_structure(to, assembler.K)
# # # #   return FiniteElementContainers.StaticAssembler{I, F, typeof(R), typeof(K)}(R, K)
# # # # end

end # module
