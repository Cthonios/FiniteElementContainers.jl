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

# NOTE the _task_partition stuff below
# is a blatant copy-paste job of AcceleratedKernels.jl
# ther @archeck in that method makes it not trimmable.
# leaving this implementation here until we get time to open up
# a PR over there
function _task_partitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    # Simple correctness checks
    @assert num_elems >= 0
    @assert max_tasks > 0
    @assert min_elems > 0

    # Number of tasks needed to have at least `min_elems` per task
    num_tasks = min(max_tasks, num_elems ÷ min_elems)
    if num_tasks <= 1
        num_tasks = 1
        return AK.TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks, Int[], Task[])
    end

    # Each task gets at least (num_elems ÷ num_tasks) elements; the remaining are redistributed
    # among the first (num_elems % num_tasks) tasks, i.e. they get one extra element
    per_task, remaining = divrem(num_elems, num_tasks)

    # Store starting index of each task
    task_istarts = Vector{Int}(undef, num_tasks)
    istart = 1
    @inbounds for i in 1:num_tasks
        task_istarts[i] = istart
        istart += i <= remaining ? per_task + 1 : per_task
    end

    # Pre-allocate tasks array; this can be reused
    tasks = Vector{Task}(undef, num_tasks)

    AK.TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks, task_istarts, tasks)
end

function _task_partition(f, num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    num_elems >= 0 || throw(ArgumentError("num_elems must be >= 0"))
    max_tasks > 0 || throw(ArgumentError("max_tasks must be > 0"))
    min_elems > 0 || throw(ArgumentError("min_elems must be > 0"))

    if min(max_tasks, num_elems ÷ min_elems) <= 1
        f(1:num_elems)
    else
        # Compiler should decide if this should be inlined; threading adds quite a bit of code, it
        # is faster (as seen in Cthulhu) to keep it in a separate self-contained function
        tp = _task_partitioner(num_elems, max_tasks, min_elems)
        _task_partition(f, tp)
    end
    nothing
end

function _task_partition(f, tp::AK.TaskPartitioner)
    for i in 1:tp.num_tasks
        tp.tasks[i] = Threads.@spawn f(@inbounds(tp[i]))
    end
    @inbounds for i in 1:tp.num_tasks
        wait(tp.tasks[i])
    end
end

function _forindices_threads(f, indices; max_tasks, min_elems)
    _task_partition(length(indices), max_tasks, min_elems) do irange
        for i in irange
            @inbounds index = indices[i]
            @inline f(index)
        end
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
        _forindices_threads(
            f, eachindex(itr),
            max_tasks = max_tasks,
            min_elems = min_elems
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
        _forindices_threads(
            f, axes(itr, dims);
            max_tasks = max_tasks,
            min_elems = min_elems
        )
    end
end

# this one only does the right thing when physics is a namedtuple
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
                block_reference_element(fspace, $k), $k
            )
        end)
    end
    return stmts
end

@generated function foreach_block(
    f,
    fspace::FunctionSpace{B, IT, IV, BTRE, C, R},
    p::TypeStableParameters
) where {B, IT, IV, BTRE, C, R}
    exprs = map(1:MAX_BLOCKS) do i
        # For each slot i, generate a branch over ALL possible ref_fe indices
        # block_to_ref_fe_id[i] == -1 means inactive block
        # Otherwise it's 1..length(REFS) — enumerate them all statically
        n_refs = length(BTRE.parameters)
        ref_dispatches = map(1:n_refs) do j
            quote
                if fspace.block_to_ref_fe_id[$i] == $j
                    f(p.physics[$i], p.properties[$i], fspace.ref_fes[$j], $i)
                end
            end
        end
        quote
            if fspace.block_to_ref_fe_id[$i] != -1
                $(ref_dispatches...)
            end
        end
    end
    quote $(exprs...) end
end