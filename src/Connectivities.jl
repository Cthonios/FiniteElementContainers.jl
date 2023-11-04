struct Connectivity{V <: AbstractVector{<:Integer}}
  n_elem::Int
  n_nodes_per_elem::Int
  n_dofs::Int
  conn::V
  dof_conn::V
end

function Connectivity(block::M, n_nodes::Int, n_dofs::Int) where M <: MeshBlock
  ids = reshape(Base.OneTo(eltype(block.conn)(n_nodes * n_dofs)), n_dofs, n_nodes)
  dof_conn = vec(ids[:, block.conn])
  Connectivity(size(block.conn, 2), size(block.conn, 1), n_dofs, block.conn[:], dof_conn)
end

Base.length(conn::Connectivity) = conn.n_elem
Base.axes(conn::Connectivity) = Base.OneTo(length(conn))

function element_connectivity(conn::Connectivity, e::Int)
  start = (e - 1) * conn.n_nodes_per_elem + 1
  finish = e * conn.n_nodes_per_elem
  @inbounds @views conn.conn[start:finish]
end

dof_connectivity(conn::Connectivity) = conn.dof_conn

function dof_connectivity(conn::Connectivity, e::Int)
  start = (e - 1) * conn.n_dofs * conn.n_nodes_per_elem + 1
  finish = e * conn.n_dofs * conn.n_nodes_per_elem
  @inbounds @views conn.dof_conn[start:finish]
end

