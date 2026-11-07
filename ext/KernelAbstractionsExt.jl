module KernelAbstractionsExt

import FiniteElementContainers as FEC
import KernelAbstractions as KA
import KernelAbstractions as KI

# general stuff
FEC.get_host() = KA.CPU()
FEC.to_backend(::KA.CPU, x) = x

# DofManager specific
# maybe we can remove this after testing against Enzyme
# COV_EXCL_START
KA.@kernel function _update_field_unknowns_kernel_1!(
    U::FEC.AbstractField, 
    unknown_dofs::IDs,
    Uu::V
) where {V <: AbstractVector{<:Number}, IDs}
    N = KA.@index(Global)
    @inbounds U.data[unknown_dofs[N]] = Uu[unknown_dofs[N]]
end
# COV_EXCL_STOP
  
# COV_EXCL_START
KA.@kernel function _update_field_unknowns_kernel_2!(
    U::FEC.AbstractField, 
    unknown_dofs::IDs,
    Uu::V
) where {V <: AbstractVector{<:Number}, IDs}
    N = KA.@index(Global)
    @inbounds U.data[unknown_dofs[N]] = Uu[N]
end
# COV_EXCL_STOP

function FEC._update_field_unknowns!(
    exec_mode::FEC.GPUExec,
    U::FEC.AbstractField, 
    dof::FEC.DofManager{flag, IT, IDs, Var}, 
    Uu::T
) where {
    T <: AbstractVector{<:Number},
    flag, IT, IDs, Var
}
    backend = KI.get_backend(U)
    if FEC._is_condensed(dof)
        kernel! = _update_field_unknowns_kernel_1!(backend)
    else
        kernel! = _update_field_unknowns_kernel_2!(backend)
    end
    kernel!(U, dof.unknown_dofs, Uu, ndrange = length(dof.unknown_dofs))
    return nothing
end

end # module
