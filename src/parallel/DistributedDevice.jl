abstract type AbstractDevice end

struct DistributedDevice <: AbstractDevice
    backend::KA.Backend
    num_ranks::Int32
    rank::Int32
    # maybe other settings later, e.g. 
    # threads on a local rank
    # gpu settings
    # etc.
end
