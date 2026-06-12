# using SimpleYAML

include("SimpleYAML.jl")


function print_dict(io::IO, d::Dict{String, T}) where T
    isempty(d) && return

    # Compute max key length (type-stable)
    maxlen = 0
    for k in Base.keys(d)
        len = ncodeunits(k)  # safer for strings than length in some contexts
        if len > maxlen
            maxlen = len
        end
    end

    # Optional: deterministic order without Pair allocations
    ks = collect(Base.keys(d))  # Vector{String}, concrete
    sort!(ks)

    for k in ks
        v = d[k]

        print(io, '"')
        print(io, k)
        print(io, '"')

        # manual padding (avoids rpad allocation)
        pad = maxlen - ncodeunits(k)
        for _ in 1:pad
            print(io, ' ')
        end

        print(io, " => ")

        print(io, '"')
        print(io, v)
        println(io, '"')
    end
end

function app_main(ARGS)
    input_file = ARGS[1]
    println(Core.stdout, input_file)

    io = open(input_file, "r")
    str = read(io, String)
    close(io)

    println(Core.stdout, str)
    d = SimpleYAML.loads(str)
    d = SimpleYAML.as_dict(d)

    solver = SimpleYAML.get_value(d, "solver", Dict{String, Dict})
    print_dict(Core.stdout, d)
    print_dict(Core.stdout, solver)

    d = SimpleYAML.to_dict(d)
    # print_dict(Core.stdout, d)
    
    return d
end

function @main(ARGS::Vector{String})
    app_main(ARGS)
    return 0
end