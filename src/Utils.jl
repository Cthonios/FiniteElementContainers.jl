function fec_atomic_add!(
    field::AbstractField{T, N, D},
    index::Int, val::T
) where {T <: Number, N, D <: AbstractArray{T, 1}}
    Atomix.@atomic field.data[index] += val
    return nothing
end

# TODO find a way to further dispatch on thread number
# so we don't suffer this runtime overhead on CPU backends
function fec_atomic_add!(
    field::AbstractField{T, N, D},
    index::Int, val::T
) where {T <: Number, N, D <: Array{T, 1}}
    if Threads.nthreads() > 1
        Atomix.@atomic field.data[index] += val
    else
        field.data[index] += val
    end
    return nothing
end

function _forindices_serial(f, indices)
    for i in indices
        @inline f(i)
    end
end

function fec_foreach(
    f, itr,
    backend::KA.Backend = KA.get_backend(itr);
    # CPU settings
    max_tasks = Threads.nthreads(),
    min_elems = 1,
    prefer_threads::Bool = true,
    # GPU settings
    block_size = 256,
)
    if AK.use_gpu_algorithm(backend, prefer_threads)
        AK._forindices_gpu(f, eachindex(itr), backend; block_size)
    elseif max_tasks == 1
        _forindices_serial(f, eachindex(itr))
    else
        AK._forindices_threads(
            f, eachindex(itr);
            max_tasks = max_tasks,
            min_elems = min_elems,
            # num_tasks = num_tasks
        )
    end
end

function fec_foraxes(
    f, itr,
    dims::Union{Nothing, <:Integer} = nothing,
    backend::KA.Backend = KA.get_backend(itr);
    # CPU settings
    max_tasks = Threads.nthreads(),
    min_elems = 1,
    prefer_threads::Bool = true,
    # GPU settings
    block_size = 256,
)
    if isnothing(dims)
        return fec_foreach(
            f, itr, backend;
            max_tasks, min_elems,
            prefer_threads, block_size,
        )
    end

    if AK.use_gpu_algorithm(backend, prefer_threads)
        AK._forindices_gpu(f, axes(itr, dims), backend; block_size)
    elseif max_tasks == 1
        _forindices_serial(f, axes(itr, dims))
    else
        AK._forindices_threads(
            f, axes(itr, dims);
            max_tasks = max_tasks,
            min_elems = min_elems
        )
    end
end

# @generated function foreach_block(
#     f, 
#     ::Connectivity{D, T},
#     physics::NamedTuple,
#     properties::NamedTuple,
#     # ref_fes::NamedTuple
#     ref_fes
# ) where {D, T}
#     stmts = Expr(:block)
#     for k in 1:fieldcount(physics)
#         push!(stmts.args, quote
#             f(
#                 values(physics)[$k], values(properties)[$k],
#                 # values(ref_fes)[$k], $k
#                 ref_fes[$k], $k
#             )
#         end)
#     end
#     return stmts
# end

@generated function foreach_block(
    f,
    fspace::FunctionSpace{IT, IV, C, R},
    p::Parameters
) where {IT, IV, C, R}
    stmts = Expr(:block)
    # NOTE need to grab one of the NamedTuple types to get the field count
    physics_type = fieldtype(p, :physics)
    N = fieldcount(physics_type)
    for k in 1:N
        push!(stmts.args, quote
            f(
                values(p.physics)[$k], values(p.properties)[$k],
                # values(fspace.ref_fes)[$k], $k
                fspace.ref_fes[$k], $k
                # block_reference_element(fspace, $k), $k
            )
        end)
    end
    return stmts
end
