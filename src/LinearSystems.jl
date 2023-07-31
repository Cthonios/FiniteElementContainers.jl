struct LinearSystem{Itype, Rtype}
  R::Vector{Rtype}
  K::SparseMatrixCSC{Rtype, Itype}
end

Base.show(io::IO, l::LinearSystem) = print(io,
  display(l.R),
  display(l.K)
)

function LinearSystem(d::DofManager{Itype, D, Rtype}) where {Itype, D, Rtype}
  R = Vector{Rtype}(undef, length(d.row_coords))
  K = sparse(d.row_coords, d.col_coords, zeros(Rtype, length(d.row_coords)))
  return LinearSystem{Itype, Rtype}(R, K)
end
