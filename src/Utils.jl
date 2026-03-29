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

function fec_axes(
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
