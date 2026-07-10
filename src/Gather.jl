# gathers and interpolates field u
# at quadrature points on block b,
# element e, quadrature point q.
# TODO call in shape_function_values
# and write method like shape_function_values(ref_fe, n, q)
# to get a specific basis function at a specific q point
@generated function interpolate_values(
    u::H1Field{T, D, NF}, x::H1Field{T, D, ND},
    conns, ref_fe,
    q, e, b,
) where {T, D, NF, ND}
    # create NF accumulator values
    sums = [Symbol(:u_, i) for i = 1:NF]
    # set them all to zero with identical type to u
    init = [:($(s) = zero(T)) for s in sums]
    # create update methods
    updates = [
        :($(sums[i]) += Ni * u[$i, idx])
        for i = 1:NF
    ]
    ret = :(SVector{$NF, T}($(sums...)))

    quote
        $(init...)
        N = ref_fe.cell_interps.values
        @inbounds for n in axes(N, 2)
            idx = connectivity(conns, n, e, b)
            Ni = N[n, q]
            $(updates...)
        end
        $ret
    end
end

@generated function mapping_jacobian(
    x::H1Field{T, D, ND},
    conns, ref_fe,
    q, e, b,
) where {T, D, ND}

    entries = Expr[]
    for i in 1:ND, j in 1:ND
        push!(entries,
            quote
                let s = zero(T)
                    @inbounds for n in axes(dN, 2)
                        idx = connectivity(conns, n, e, b)
                        s += x[$i, idx] * dN[$j, n, q]
                    end
                    s
                end
            end)
    end

    quote
        dN = ref_fe.cell_interps.gradients
        SMatrix{$ND, $ND, T, $(ND * ND)}(
            $(entries...)
        )
    end
end

# TODO not finished...
# need to write jacobian mapping operator
@generated function interpolate_gradients(
    u::H1Field{T, D, NF}, x::H1Field{T, D, NF},
    conns, ref_fe,
    q, e, b
) where {T, D, NF}
    dim = ReferenceFiniteElements.dimension(ref_fe)
    sums = [Symbol(:u_, j, :_, i) for j = 1:dim, i = 1:NF]
    init = [:($(s) = zero(T)) for s in sums]
    updates = [
        :($(sums[j, i]) += ∇N_ji * u[$i, idx])
        for j = 1:dim, i = 1:NF
    ]
    ret = :(SMatrix{$dim, $NF, T $dim * $NF})($(sums...))
    quote
        $(init...)
        ∇N_ξ
    end
end

@inline function interpolate_gradient(
    u::H1Field{T, D, NF}, x::H1Field{T, D, ND},
    conns, ref_fe,
    q, e, b,
) where {T, D, NF, ND}
    dN = ref_fe.cell_interps.gradients
    J = mapping_jacobian(x, conns, ref_fe, q, e, b)
    # inverse Jacobian (computed once)
    invJ = inv(J)

    # result: ND x NF matrix
    return SMatrix{ND, NF, T}(ntuple(Val(ND * NF)) do k
        i = (k - 1) ÷ NF + 1   # spatial direction
        f = (k - 1) % NF + 1   # field component

        # compute reference gradient component ∂u_f / ∂ξ_i
        du_dξ = zero(T)

        @inbounds for n in axes(dN, 2)
            idx = connectivity(conns, n, e, b)
            du_dξ += dN[i, n, q] * u[f, idx]
        end

        # map to physical space: (J^{-1})_{i,j} ∂u/∂ξ_j
        du_dx = zero(T)
        @inbounds for j in 1:ND
            du_dx += invJ[j, i] * du_dξ
        end

        du_dx
    end)
end
