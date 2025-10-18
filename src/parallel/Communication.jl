# The assumptions of this struct are as follows
# the dege points from the current comm
# to rank. 
# This struct can be used to build
# send/receive data exchangerss

struct CommunicationGraphEdge{
    TV <: AbstractVector,
    IV <: AbstractVector{<:Integer}
}
    data_recv::TV
    data_send::TV
    indices::IV
    is_owned_recv::IV
    is_owned_send::IV
    rank::Int32
end

# TODO cpu only method currently
function setdatasend!(edge::CommunicationGraphEdge, part)
    for n in axes(edge.indices, 1)
        index = edge.indices[n]
        edge.data_send[n] = part[index]
    end
end

struct CommunicationGraph{
    TV <: AbstractVector,
    IV <: AbstractVector{<:Integer}
}
    edges::Vector{CommunicationGraphEdge{TV, IV}}
    global_to_color::IV
    n_local::Int
    n_owned::Int
end
