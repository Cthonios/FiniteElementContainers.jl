"""
$(TYPEDEF)
"""
const Connectivity{T, D, NF} = L2ElementField{T, D, NF}

"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::Connectivity) = conn.data
"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::Connectivity, e::Int) = @views conn[:, e] # TODO maybe a duplicate view here in some cases
