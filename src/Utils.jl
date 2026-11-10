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

"""
$(TYPEDSIGNATURES)
Create a dense array on ``backend`` of default type.
"""
function fec_dense_array(backend::KA.Backend, size...)
    return fec_dense_array(backend, Float64, size...)
end

"""
$(TYPEDSIGNATURES)
Create a dense array on the host of arbitrary type ``RT``.
"""
function fec_dense_array(::KA.CPU, ::Type{RT}, size...) where RT <: Number
    return zeros(RT, size...)
end

# boosted from AcceleratedKernels.jl
# The `i`-th element of `indices`, counting from 1 whatever the first index of `indices` is
@inline nth(indices, i) = @inbounds indices[firstindex(indices) - 1 + i]
KA.@kernel inbounds=true cpu=false unsafe_indices=true function foreachindex_kernel!(f, indices)
    N = KA.@groupsize()[1]
    iblock = KA.@index(Group, Linear)
    ithread = KA.@index(Local, Linear)
    i = ithread + (iblock - 0x1) * N

    # Avoid an out-of-line device call that can copy captured state to local memory.
    if i <= length(indices)
        @inline f(nth(indices, i))
    end
end

# end code boosted from AcceleratedKernels.jl

function _forindices_gpu(f, indices, backend::KA.Backend, block_size::Int)
    blocks = cld(length(indices), block_size)
    foreachindex_kernel!(backend, block_size)(f, indices, ndrange=(block_size * blocks,))
end

function _forindices_serial(f, indices)
    for i in indices
        @inline f(i)
    end
end

function _forindices_threads(f, indices; max_tasks, min_elems)
    task_partition(length(indices), max_tasks, min_elems) do irange
        for i in irange
            @inbounds index = indices[i]
            @inline f(index)
        end
    end
end

# cpu implementation for both serial and multi-threading.
# one method name at the cost of a branch on CPU runs.
# no such branch in the GPU code path
function _fec_forindices(
    ::KA.CPU, f, indices,
    block_size::Int, max_tasks::Int, min_elems::Int
)
    if max_tasks == 1
        _forindices_serial(f, indices)
    else
        _forindices_threads(
            f, indices;
            max_tasks = max_tasks,
            min_elems = min_elems
        )
    end
end

# gpu implementation
function _fec_forindices(
    backend::KA.Backend, f, indices,
    block_size::Int, max_tasks::Int, min_elems::Int
)
    _forindices_gpu(f, indices, backend, block_size)
end

function fec_foreach(
    f, itr,
    backend::KA.Backend = KA.get_backend(itr);
    # CPU settings
    max_tasks = Threads.nthreads(),
    min_elems = 1,
    # GPU settingss
    block_size = 256,
)
    _fec_forindices(backend, f, eachindex(itr), block_size, max_tasks, min_elems)
end

function fec_foraxes(
    f, itr,
    dims::Union{Nothing, <:Integer} = nothing,
    backend::KA.Backend = KA.get_backend(itr);
    # CPU settings
    max_tasks = Threads.nthreads(),
    min_elems = 1,
    # GPU settings
    block_size = 256,
)
    if isnothing(dims)
        return fec_foreach(
            f, itr, backend;
            block_size, max_tasks, min_elems
        )
    end

    _fec_forindices(backend, f, axes(itr, dims), block_size, max_tasks, min_elems)
end

# this one only does the right thing when physics is a namedtuple
@generated function foreach_block(
    f,
    fspace::FunctionSpace,
    p::Parameters
)
    stmts = Expr(:block)
    # NOTE need to grab one of the NamedTuple types to get the field count
    physics_type = fieldtype(p, :physics)
    N = fieldcount(physics_type)
    for k in 1:N
        push!(stmts.args, quote
            f(
                values(p.physics)[$k],
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
                    f(p.physics[$i], fspace.ref_fes[$j], $i)
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
# function foreach_block(
#     f,
#     fspace,
#     p::TypeStableParameters
# )
#     for b in 1:num_blocks(fspace)
#         f(p.physics[b], p.properties[b], block_reference_element(fspace, b), b)
#     end
# end

# assumes ref_fes is a namedtuple
@generated function foreach_block(
    f,
    fspace::FunctionSpace{false, IT, IV, BTRE, C, R}
) where {IT, IV, BTRE, C, R}
    stmts = Expr(:block)
    # NOTE need to grab one of the NamedTuple types to get the field count
    ref_fes_type = fieldtype(fspace, :ref_fes)
    N = fieldcount(ref_fes_type)
    for k in 1:N
        push!(stmts.args, quote
            f(
                block_reference_element(fspace, $k), $k
            )
        end)
    end
    return stmts
end

@generated function foreach_block(
    f,
    fspace::FunctionSpace{true, IT, IV, BTRE, C, R}
) where {IT, IV, BTRE, C, R}
    exprs = map(1:MAX_BLOCKS) do i
        # For each slot i, generate a branch over ALL possible ref_fe indices
        # block_to_ref_fe_id[i] == -1 means inactive block
        # Otherwise it's 1..length(REFS) — enumerate them all statically
        n_refs = length(BTRE.parameters)
        ref_dispatches = map(1:n_refs) do j
            quote
                if fspace.block_to_ref_fe_id[$i] == $j
                    f(fspace.ref_fes[$j], $i)
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

"""
takes in connectivity and element id
"""
function foreach_element(
    f, conns, block_id,
    backend = KA.get_backend(conns.data);
    max_tasks = Threads.nthreads(),
    min_elems = 1,
    # GPU settings
    block_size = 256
)
    nelems = conns.nelems[block_id]
    _fec_forindices(backend, f, 1:nelems, block_size, max_tasks, min_elems)
end

# below is taken directly from AcceleratedKernels.jl
# so that we don't need to depend on AK  and the whole
# julia GPU stack for cpu multithreading
#
# made a few changes to remove the @archeck macros

struct TaskPartitioner
    num_elems::Int
    max_tasks::Int
    min_elems::Int

    # Computed
    num_tasks::Int
    task_istarts::Vector{Int}
    tasks::Vector{Task}
end

# The number of tasks of a `TaskPartitioner(num_elems, max_tasks, min_elems)`: at most
# `max_tasks`, each with at least `min_elems` elements, and one if there are fewer
task_count(num_elems, max_tasks, min_elems) = max(1, min(max_tasks, num_elems ÷ min_elems))

function TaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    num_tasks = task_count(num_elems, max_tasks, min_elems)
    if num_tasks == 1
        return TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks, Int[], Task[])
    end

    # Each task gets at least (num_elems ÷ num_tasks) elements; the remaining are redistributed
    # among the first (num_elems % num_tasks) tasks, i.e. they get one extra element
    per_task, remaining = divrem(num_elems, num_tasks)

    task_istarts = Vector{Int}(undef, num_tasks)
    istart = 1
    @inbounds for i in 1:num_tasks
        task_istarts[i] = istart
        istart += i <= remaining ? per_task + 1 : per_task
    end

    # (reused by every call with this partitioner)
    tasks = Vector{Task}(undef, num_tasks)

    TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks, task_istarts, tasks)
end

function Base.getindex(tp::TaskPartitioner, itask::Integer)

    @boundscheck 1 <= itask <= tp.num_tasks || throw(BoundsError(tp, itask))

    # Special-cased for single task, in which case tp.task_istarts is empty
    if tp.num_tasks == 1
        return 1:tp.num_elems
    end

    task_istart = @inbounds tp.task_istarts[itask]
    if itask == tp.num_tasks
        return task_istart:tp.num_elems
    else
        task_istop = @inbounds tp.task_istarts[itask + 1] - 1
        return task_istart:task_istop
    end
end

Base.firstindex(::TaskPartitioner) = 1
Base.lastindex(tp::TaskPartitioner) = tp.num_tasks
Base.length(tp::TaskPartitioner) = tp.num_tasks

task_partition(f, args...) = itask_partition((_, irange) -> f(irange), args...)

function itask_partition(f, num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    num_elems >= 0 || throw(ArgumentError("num_elems must be >= 0"))
    max_tasks > 0 || throw(ArgumentError("max_tasks must be > 0"))
    min_elems > 0 || throw(ArgumentError("min_elems must be > 0"))

    if task_count(num_elems, max_tasks, min_elems) == 1
        f(1, 1:num_elems)
    else
        # Compiler should decide if this should be inlined; threading adds quite a bit of code, it
        # is faster (as seen in Cthulhu) to keep it in a separate self-contained function
        tp = TaskPartitioner(num_elems, max_tasks, min_elems)
        itask_partition(f, tp)
    end
    nothing
end

function itask_partition(f, tp::TaskPartitioner)
    # A single task has no `tasks` entry
    tp.num_tasks == 1 && return (f(1, 1:tp.num_elems); nothing)
    for i in 1:tp.num_tasks
        tp.tasks[i] = Threads.@spawn f(i, @inbounds(tp[i]))
    end
    @inbounds for i in 1:tp.num_tasks
        wait(tp.tasks[i])
    end
end

#########################################
# hooks for extensions
#########################################
# device stuff
# function cpu end
cpu(x) = to_host(x)
function cuda end
function rocm end

"""
$(TYPEDSIGNATURES)
Move `x` onto the given KernelAbstractions backend.  CPU is identity —
CPU-built data already lives on the CPU backend.  GPU backends (CUDABackend,
ROCBackend) are provided by the CUDA / AMDGPU package extensions.
"""
to_backend(b::KA.Backend, x) = error(
  "to_backend is not implemented for backend $(typeof(b)); load the " *
  "corresponding GPU package (CUDA.jl or AMDGPU.jl) so its extension activates."
)

# to_backend(::KA.CPU, x) = x

"""
$(TYPEDSIGNATURES)
Move `x` onto the host.
"""
to_host(x) = to_backend(KA.CPU(), x)

function _coo_matrix end
function _csc_matrix end
function _csr_matrix end
