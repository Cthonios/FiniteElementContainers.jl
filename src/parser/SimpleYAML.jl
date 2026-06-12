module SimpleYAML

export YAMLValue, YAMLNull, YAMLBool, YAMLInt, YAMLFloat, YAMLString,
       YAMLArray, YAMLDict,
       load, loads,
       as_dict, as_array, as_string, as_int, as_float, as_bool, is_null,
       get_value, to_dict

# ─────────────────────────────────────────────────────────────────────────────
# Value type  (concrete tagged union — trim-safe, fully type-stable)
# ─────────────────────────────────────────────────────────────────────────────

@enum YAMLTag begin
    TAG_NULL; TAG_BOOL; TAG_INT; TAG_FLOAT; TAG_STRING; TAG_ARRAY; TAG_DICT
end

struct YAMLValue
    tag  :: YAMLTag
    bval :: Bool
    ival :: Int64
    fval :: Float64
    sval :: String
    arr  :: Vector{YAMLValue}
    dict :: Dict{String, YAMLValue}
end

const YAMLNull   = YAMLValue(TAG_NULL,  false, 0,   0.0, "", YAMLValue[], Dict{String,YAMLValue}())
YAMLBool(b::Bool)      = YAMLValue(TAG_BOOL,  b,     0,   0.0, "", YAMLValue[], Dict{String,YAMLValue}())
YAMLInt(i::Int64)      = YAMLValue(TAG_INT,   false, i,   0.0, "", YAMLValue[], Dict{String,YAMLValue}())
YAMLFloat(f::Float64)  = YAMLValue(TAG_FLOAT, false, 0,   f,   "", YAMLValue[], Dict{String,YAMLValue}())
YAMLString(s::String)  = YAMLValue(TAG_STRING,false, 0,   0.0, s,  YAMLValue[], Dict{String,YAMLValue}())
YAMLArray(a::Vector{YAMLValue})     = YAMLValue(TAG_ARRAY, false, 0, 0.0, "", a, Dict{String,YAMLValue}())
YAMLDict(d::Dict{String,YAMLValue}) = YAMLValue(TAG_DICT,  false, 0, 0.0, "", YAMLValue[], d)

is_null(v::YAMLValue) = v.tag === TAG_NULL

function as_bool(v::YAMLValue)::Bool
    v.tag === TAG_BOOL || error("YAMLValue is not a bool (tag=$(v.tag))"); v.bval
end
function as_int(v::YAMLValue)::Int64
    v.tag === TAG_INT || error("YAMLValue is not an int (tag=$(v.tag))"); v.ival
end
function as_float(v::YAMLValue)::Float64
    v.tag === TAG_FLOAT || error("YAMLValue is not a float (tag=$(v.tag))"); v.fval
end
function as_string(v::YAMLValue)::String
    v.tag === TAG_STRING || error("YAMLValue is not a string (tag=$(v.tag))"); v.sval
end
function as_array(v::YAMLValue)::Vector{YAMLValue}
    v.tag === TAG_ARRAY || error("YAMLValue is not an array (tag=$(v.tag))"); v.arr
end
function as_dict(v::YAMLValue)::Dict{String,YAMLValue}
    v.tag === TAG_DICT || error("YAMLValue is not a dict (tag=$(v.tag))"); v.dict
end

function Base.show(io::IO, v::YAMLValue)
    if v.tag === TAG_NULL;        print(io, "null")
    elseif v.tag === TAG_BOOL;    print(io, v.bval ? "true" : "false")
    elseif v.tag === TAG_INT;     print(io, v.ival)
    elseif v.tag === TAG_FLOAT;   print(io, v.fval)
    elseif v.tag === TAG_STRING;  print(io, repr(v.sval))
    elseif v.tag === TAG_ARRAY
        print(io, "[")
        for (i, x) in enumerate(v.arr); i > 1 && print(io, ", "); show(io, x); end
        print(io, "]")
    else
        print(io, "{")
        first = true
        for (k, val) in v.dict
            first || print(io, ", "); first = false
            print(io, repr(k), ": "); show(io, val)
        end
        print(io, "}")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Parser state
# ─────────────────────────────────────────────────────────────────────────────

mutable struct ParseState
    lines   :: Vector{String}
    lineno  :: Int
    anchors :: Dict{String, YAMLValue}
end

function ParseState(src::String)::ParseState
    # split() returns SubStrings — convert immediately so all downstream
    # functions only ever handle concrete String.
    raw   = split(src, '\n')
    lines = Vector{String}(undef, length(raw))
    for i in eachindex(raw); lines[i] = String(raw[i]); end
    return ParseState(lines, 1, Dict{String,YAMLValue}())
end

@inline at_end(ps::ParseState)       = ps.lineno > length(ps.lines)
@inline current_line(ps::ParseState) = ps.lineno <= length(ps.lines) ? ps.lines[ps.lineno] : ""

# ─────────────────────────────────────────────────────────────────────────────
# String helpers — ALL use only codeunit / nextind (never s[byte_index]).
#
# Design rule: the only place character indexing s[i] is allowed is when i
# comes from firstindex(s) or nextind(s, prev) — i.e. a proper character walk.
# Every other access uses codeunit(s, i) for byte inspection or byte-range
# slices s[a:b] where a/b are validated character boundaries.
# ─────────────────────────────────────────────────────────────────────────────

"""Count leading ASCII space bytes."""
function line_indent(l::String)::Int
    n = ncodeunits(l); i = 1
    while i <= n && codeunit(l, i) == UInt8(' '); i += 1; end
    return i - 1
end

"""
Return (indent::Int, content::String) for a raw line.
content is the line with leading spaces removed, as a proper String.
"""
function split_indent(l::String)::Tuple{Int,String}
    n   = ncodeunits(l)
    ind = line_indent(l)
    # ind is a count of ASCII space bytes, so ind+1 is always a valid char start.
    return ind, ind < n ? l[ind+1:end] : ""
end

"""Find the indent of the next non-blank non-comment line without advancing."""
function peek_indent(ps::ParseState)::Int
    i = ps.lineno
    while i <= length(ps.lines)
        ind, rest = split_indent(ps.lines[i])
        if !isempty(rest) && codeunit(rest, 1) != UInt8('#')
            return ind
        end
        i += 1
    end
    return -1
end

function skip_empty!(ps::ParseState)
    while !at_end(ps)
        _, rest = split_indent(current_line(ps))
        if isempty(rest) || codeunit(rest, 1) == UInt8('#')
            ps.lineno += 1
        else
            break
        end
    end
end

"""
Strip trailing inline comment (' # ...') and trailing whitespace.
Always returns a plain String; uses only byte-level ops.
"""
function strip_inline_comment(s::String)::String
    n = ncodeunits(s)
    # Need at least 2 bytes: char before '#' + '#' itself
    i = 2
    while i <= n
        if codeunit(s, i) == UInt8('#') && codeunit(s, i-1) == UInt8(' ')
            j = i - 1   # byte just before the space-hash
            while j >= 1 && (codeunit(s, j) == UInt8(' ') || codeunit(s, j) == UInt8('\t'))
                j -= 1
            end
            return j >= 1 ? s[1:j] : ""
        end
        i += 1
    end
    # no comment — rstrip trailing whitespace
    j = n
    while j >= 1 && (codeunit(s, j) == UInt8(' ')  || codeunit(s, j) == UInt8('\t') ||
                      codeunit(s, j) == UInt8('\r') || codeunit(s, j) == UInt8('\n'))
        j -= 1
    end
    return j >= 1 ? (j == n ? s : s[1:j]) : ""
end

"""
lstrip ASCII whitespace, returning a plain String.
Only advances through single-byte ASCII space/tab characters, so the
first non-whitespace byte is always a valid character boundary.
"""
function lstrip_ascii(s::String)::String
    n = ncodeunits(s); i = 1
    while i <= n && (codeunit(s, i) == UInt8(' ') || codeunit(s, i) == UInt8('\t'))
        i += 1
    end
    return i <= n ? s[i:end] : ""
end

"""
Return the suffix of s starting one character after byte-position `byte_pos`.
`byte_pos` must point to the start of a character (i.e. be a valid index).
Uses nextind to step one character, then slices to end.
"""
function suffix_after_char(s::String, byte_pos::Int)::String
    n = ncodeunits(s)
    ni = nextind(s, byte_pos)
    return ni <= n ? s[ni:end] : ""
end

# ─────────────────────────────────────────────────────────────────────────────
# Scalar coercion
# ─────────────────────────────────────────────────────────────────────────────

function parse_scalar(s::String)::YAMLValue
    isempty(s) && return YAMLNull
    (s == "null" || s == "Null" || s == "NULL" || s == "~") && return YAMLNull
    (s == "true"  || s == "True"  || s == "TRUE")  && return YAMLBool(true)
    (s == "false" || s == "False" || s == "FALSE")  && return YAMLBool(false)
    (s == ".inf"  || s == ".Inf"  || s == ".INF"  ||
     s == "+.inf" || s == "+.Inf" || s == "+.INF") && return YAMLFloat(Inf)
    (s == "-.inf" || s == "-.Inf" || s == "-.INF") && return YAMLFloat(-Inf)
    (s == ".nan"  || s == ".NaN"  || s == ".NAN")  && return YAMLFloat(NaN)

    n = ncodeunits(s)
    if n >= 3 && codeunit(s,1) == UInt8('0') && codeunit(s,2) == UInt8('x')
        v = tryparse(Int64, s[3:end]; base=16); v !== nothing && return YAMLInt(v)
    elseif n >= 3 && codeunit(s,1) == UInt8('0') && codeunit(s,2) == UInt8('o')
        v = tryparse(Int64, s[3:end]; base=8);  v !== nothing && return YAMLInt(v)
    elseif n >= 3 && codeunit(s,1) == UInt8('0') && codeunit(s,2) == UInt8('b')
        v = tryparse(Int64, s[3:end]; base=2);  v !== nothing && return YAMLInt(v)
    else
        v = tryparse(Int64, s);    v !== nothing && return YAMLInt(v)
    end
    fv = tryparse(Float64, s); fv !== nothing && return YAMLFloat(fv)
    return YAMLString(s)
end

# ─────────────────────────────────────────────────────────────────────────────
# Quoted-string parsers
# Take a String + byte start index (just after opening quote).
# Return (parsed String, byte index after closing quote).
# Use proper character walks (nextind), so Unicode-safe throughout.
# ─────────────────────────────────────────────────────────────────────────────

function parse_single_quoted(s::String, start::Int)::Tuple{String,Int}
    buf = IOBuffer(); i = start; n = ncodeunits(s)
    while i <= n
        # Walk one character at a time using nextind
        ci  = i
        i   = nextind(s, ci)   # advance past current char
        ch  = s[ci]            # safe: ci from nextind chain
        if ch == '\''
            if i <= n && codeunit(s, i) == UInt8('\'')
                write(buf, '\''); i = nextind(s, i)
            else
                break   # closing quote; i already past it
            end
        else
            write(buf, ch)
        end
    end
    return String(take!(buf)), i
end

function parse_double_quoted(s::String, start::Int)::Tuple{String,Int}
    buf = IOBuffer(); i = start; n = ncodeunits(s)
    while i <= n
        ci = i; i = nextind(s, ci); ch = s[ci]
        if ch == '"'
            break
        elseif ch == '\\'
            i > n && break
            ei = i; i = nextind(s, ei); esc = s[ei]
            if     esc == '0';  write(buf, '\0')
            elseif esc == 'a';  write(buf, '\a')
            elseif esc == 'b';  write(buf, '\b')
            elseif esc == 't' || esc == '\t'; write(buf, '\t')
            elseif esc == 'n';  write(buf, '\n')
            elseif esc == 'v';  write(buf, '\v')
            elseif esc == 'f';  write(buf, '\f')
            elseif esc == 'r';  write(buf, '\r')
            elseif esc == 'e';  write(buf, '\e')
            elseif esc == ' ';  write(buf, ' ')
            elseif esc == '"';  write(buf, '"')
            elseif esc == '/';  write(buf, '/')
            elseif esc == '\\'; write(buf, '\\')
            elseif esc == 'N';  write(buf, '\u0085')
            elseif esc == '_';  write(buf, '\u00a0')
            elseif esc == 'L';  write(buf, '\u2028')
            elseif esc == 'P';  write(buf, '\u2029')
            elseif esc == 'x'
                hex = i+1 <= n ? s[i:i+1] : "00"
                cp  = tryparse(UInt32, hex; base=16)
                write(buf, cp !== nothing ? Char(cp) : '?')
                i <= n && (i = nextind(s, i))
                i <= n && (i = nextind(s, i))
            elseif esc == 'u'
                hex = i+3 <= n ? s[i:i+3] : "0000"
                cp  = tryparse(UInt32, hex; base=16)
                write(buf, cp !== nothing ? Char(cp) : '?')
                for _ in 1:4; i <= n && (i = nextind(s, i)); end
            elseif esc == 'U'
                hex = i+7 <= n ? s[i:i+7] : "00000000"
                cp  = tryparse(UInt32, hex; base=16)
                write(buf, cp !== nothing ? Char(cp) : '?')
                for _ in 1:8; i <= n && (i = nextind(s, i)); end
            else
                write(buf, esc)
            end
        else
            write(buf, ch)
        end
    end
    return String(take!(buf)), i
end

# ─────────────────────────────────────────────────────────────────────────────
# Block scalar  (| and >)
# Caller has already consumed the header line; ps.lineno → first content line.
# ─────────────────────────────────────────────────────────────────────────────

function parse_block_scalar(ps::ParseState, style::Char, header::String)::String
    chomping = :clip; explicit_indent = 0
    h = header
    # strip trailing comment
    hi = firstindex(h)
    while hi <= lastindex(h)
        if codeunit(h, hi) == UInt8('#') && hi > 1 && codeunit(h, hi-1) == UInt8(' ')
            h = String(rstrip(h[1:hi-2])); break
        end
        hi = nextind(h, hi)
    end
    h = String(strip(h))
    # parse chomping / explicit-indent chars (all ASCII, safe with nextind walk)
    hi = firstindex(h)
    while hi <= lastindex(h)
        ch = h[hi]   # safe: hi from nextind chain starting at firstindex
        if     ch == '-';   chomping = :strip; hi = nextind(h, hi)
        elseif ch == '+';   chomping = :keep;  hi = nextind(h, hi)
        elseif isdigit(ch); explicit_indent = Int(ch - '0'); hi = nextind(h, hi)
        else; break
        end
    end

    block_indent = explicit_indent
    if block_indent == 0
        j = ps.lineno
        while j <= length(ps.lines)
            ind, rest = split_indent(ps.lines[j])
            if !isempty(rest); block_indent = ind; break; end
            j += 1
        end
    end

    lines_out = String[]
    while !at_end(ps)
        l = current_line(ps)
        ind, rest = split_indent(l)
        if isempty(rest)
            push!(lines_out, ""); ps.lineno += 1; continue
        end
        ind < block_indent && break
        push!(lines_out, l[block_indent+1:end]); ps.lineno += 1
    end

    result::String = if style == '|'
        join(lines_out, "\n")
    else
        buf = IOBuffer(); prev_empty = false
        for (idx, ln) in enumerate(lines_out)
            if isempty(ln); prev_empty = true
            else
                if idx > 1 && !prev_empty; write(buf, ' ')
                elseif prev_empty;          write(buf, '\n')
                end
                write(buf, ln); prev_empty = false
            end
        end
        String(take!(buf))
    end

    if     chomping === :strip; result = String(rstrip(result, '\n'))
    elseif chomping === :clip;  result = String(rstrip(result, '\n')) * "\n"
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# Flow (inline) parsers
# All use codeunit for structural character checks and nextind for walking.
# ─────────────────────────────────────────────────────────────────────────────

function parse_flow_value(ps::ParseState, s::String, i::Int)::Tuple{YAMLValue,Int}
    n = ncodeunits(s)
    while i <= n && (codeunit(s,i) == UInt8(' ') || codeunit(s,i) == UInt8('\t'))
        i += 1
    end
    i > n && return YAMLNull, i

    # Peek at first byte — all YAML flow structural characters are single-byte ASCII.
    b = codeunit(s, i)
    if b == UInt8('{')
        return parse_flow_mapping(ps, s, i+1)
    elseif b == UInt8('[')
        return parse_flow_sequence(ps, s, i+1)
    elseif b == UInt8('\'')
        str, i2 = parse_single_quoted(s, i+1)
        return YAMLString(str), i2
    elseif b == UInt8('"')
        str, i2 = parse_double_quoted(s, i+1)
        return YAMLString(str), i2
    elseif b == UInt8('&')
        j = i + 1
        while j <= n && codeunit(s,j) != UInt8(' ') && codeunit(s,j) != UInt8(',') &&
                        codeunit(s,j) != UInt8('}') && codeunit(s,j) != UInt8(']')
            j += 1
        end
        aname = s[i+1:j-1]   # anchor names are ASCII by convention
        val, i2 = parse_flow_value(ps, s, j)
        ps.anchors[aname] = val
        return val, i2
    elseif b == UInt8('*')
        j = i + 1
        while j <= n && codeunit(s,j) != UInt8(' ') && codeunit(s,j) != UInt8(',') &&
                        codeunit(s,j) != UInt8('}') && codeunit(s,j) != UInt8(']')
            j += 1
        end
        return get(ps.anchors, s[i+1:j-1], YAMLNull), j
    else
        j = i
        while j <= n
            bj = codeunit(s, j)
            (bj == UInt8(',') || bj == UInt8('}') || bj == UInt8(']') || bj == UInt8('#')) && break
            j += 1
        end
        raw = String(strip(s[i:j-1]))
        return parse_scalar(raw), j
    end
end

function parse_flow_sequence(ps::ParseState, s::String, i::Int)::Tuple{YAMLValue,Int}
    items = YAMLValue[]; n = ncodeunits(s)
    while i <= n
        while i <= n && (codeunit(s,i) == UInt8(' ') || codeunit(s,i) == UInt8('\t') ||
                          codeunit(s,i) == UInt8('\n'))
            i += 1
        end
        i > n && break
        codeunit(s,i) == UInt8(']') && (i += 1; break)
        codeunit(s,i) == UInt8(',') && (i += 1; continue)
        val, i = parse_flow_value(ps, s, i)
        push!(items, val)
        while i <= n && (codeunit(s,i) == UInt8(' ') || codeunit(s,i) == UInt8('\t')); i += 1; end
        i <= n && codeunit(s,i) == UInt8(',') && (i += 1)
    end
    return YAMLArray(items), i
end

function parse_flow_mapping(ps::ParseState, s::String, i::Int)::Tuple{YAMLValue,Int}
    d = Dict{String,YAMLValue}(); n = ncodeunits(s)
    while i <= n
        while i <= n && (codeunit(s,i) == UInt8(' ') || codeunit(s,i) == UInt8('\t') ||
                          codeunit(s,i) == UInt8('\n'))
            i += 1
        end
        i > n && break
        codeunit(s,i) == UInt8('}') && (i += 1; break)
        codeunit(s,i) == UInt8(',') && (i += 1; continue)
        key_val, i = parse_flow_value(ps, s, i)
        key = yaml_value_to_key(key_val)
        while i <= n && (codeunit(s,i) == UInt8(' ') || codeunit(s,i) == UInt8('\t')); i += 1; end
        i <= n && codeunit(s,i) == UInt8(':') && (i += 1)
        val, i = parse_flow_value(ps, s, i)
        d[key] = val
        while i <= n && (codeunit(s,i) == UInt8(' ') || codeunit(s,i) == UInt8('\t')); i += 1; end
        i <= n && codeunit(s,i) == UInt8(',') && (i += 1)
    end
    return YAMLDict(d), i
end

function yaml_value_to_key(v::YAMLValue)::String
    v.tag === TAG_STRING && return v.sval
    v.tag === TAG_INT    && return string(v.ival)
    v.tag === TAG_FLOAT  && return string(v.fval)
    v.tag === TAG_BOOL   && return v.bval ? "true" : "false"
    return "null"
end

# ─────────────────────────────────────────────────────────────────────────────
# Key detection — fully Unicode-safe via character walks
#
# Returns (key::String, after_colon_byte_index::Int) or (nothing, 0).
# after_colon_byte_index points one past the ':' byte (i.e. the start of the
# value portion); it is always a valid character boundary because ':' is ASCII
# and the next byte starts a new character.
# ─────────────────────────────────────────────────────────────────────────────

function find_mapping_key(content::String)::Tuple{Union{Nothing,String},Int}
    isempty(content) && return nothing, 0
    n = ncodeunits(content)
    b1 = codeunit(content, 1)

    if b1 == UInt8('"')
        key, i = parse_double_quoted(content, 2)
        # skip spaces (all ASCII, so += 1 safe)
        while i <= n && codeunit(content, i) == UInt8(' '); i += 1; end
        if i <= n && codeunit(content, i) == UInt8(':')
            ni = i + 1   # safe: ':' is 1 byte, next byte starts a new char
            if ni > n || codeunit(content, ni) == UInt8(' ') ||
               codeunit(content, ni) == UInt8('\t') || codeunit(content, ni) == UInt8('#')
                return key, ni
            end
        end
        return nothing, 0

    elseif b1 == UInt8('\'')
        key, i = parse_single_quoted(content, 2)
        while i <= n && codeunit(content, i) == UInt8(' '); i += 1; end
        if i <= n && codeunit(content, i) == UInt8(':')
            ni = i + 1
            if ni > n || codeunit(content, ni) == UInt8(' ') ||
               codeunit(content, ni) == UInt8('\t') || codeunit(content, ni) == UInt8('#')
                return key, ni
            end
        end
        return nothing, 0

    else
        # Bare key: walk character by character using nextind.
        # We are looking for a ':' byte followed by space/tab/EOF/comment.
        # ':' is ASCII so codeunit comparison is correct; nextind handles
        # multi-byte characters safely.
        i = firstindex(content)   # always 1, but semantically correct
        while i <= lastindex(content)
            if codeunit(content, i) == UInt8(':')
                ni = i + 1   # ':' is ASCII (1 byte), so ni is a valid index
                if ni > n || codeunit(content, ni) == UInt8(' ') ||
                   codeunit(content, ni) == UInt8('\t') || codeunit(content, ni) == UInt8('#')
                    # Collect key as content[1 .. prevind(content, i)]
                    # rstrip ASCII whitespace from the right
                    ke = i - 1   # byte before ':'
                    while ke >= 1 && (codeunit(content, ke) == UInt8(' ') ||
                                      codeunit(content, ke) == UInt8('\t'))
                        ke -= 1
                    end
                    ke < 1 && return nothing, 0
                    # content[1:ke] is a valid byte-range slice: ke is either
                    # the last byte of a multi-byte char (fine for range end)
                    # or a single-byte char.  Julia range slices are byte-based
                    # and only require the START to be a valid char boundary;
                    # the end can be any byte position >= the last byte of a char.
                    return content[1:ke], ni
                end
            end
            i = nextind(content, i)
        end
        return nothing, 0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Block mapping
# ─────────────────────────────────────────────────────────────────────────────

function parse_block_mapping(ps::ParseState, map_indent::Int)::YAMLValue
    d = Dict{String,YAMLValue}()
    while !at_end(ps)
        skip_empty!(ps); at_end(ps) && break
        l = current_line(ps)
        indent, content = split_indent(l)
        indent != map_indent && break
        isempty(content) && (ps.lineno += 1; continue)
        (startswith(content, "---") || startswith(content, "...")) && break

        key, after_colon = find_mapping_key(content)
        key === nothing && break
        ps.lineno += 1   # consume key line

        # Value portion on the same line: content[after_colon:end], lstripped.
        # after_colon is always a valid character boundary (one past ASCII ':').
        rest = after_colon <= n_cu(content) ? lstrip_ascii(content[after_colon:end]) : ""
        rest = strip_inline_comment(rest)

        val::YAMLValue = _parse_value_after_colon(ps, rest, map_indent)
        d[key] = val
    end
    return YAMLDict(d)
end

# ncodeunits shorthand (avoids repeating the call name)
@inline n_cu(s::String) = ncodeunits(s)

"""
Determine the value given the stripped inline remainder after a mapping colon
(or after a sequence dash), plus the current block's indent level.
"""
function _parse_value_after_colon(ps::ParseState, rest::String, block_indent::Int)::YAMLValue
    if isempty(rest) || codeunit(rest, 1) == UInt8('#')
        # value on subsequent line(s)
        next_ind = peek_indent(ps)
        if next_ind < 0
            return YAMLNull
        elseif next_ind > block_indent
            return parse_node(ps, next_ind)
        elseif next_ind == block_indent
            _, nc = split_indent(current_line(ps))
            if !isempty(nc) && codeunit(nc, 1) == UInt8('-') &&
               (n_cu(nc) == 1 || codeunit(nc, 2) == UInt8(' ') || codeunit(nc, 2) == UInt8('\t'))
                return parse_block_sequence(ps, next_ind)
            end
        end
        return YAMLNull

    elseif codeunit(rest, 1) == UInt8('|') || codeunit(rest, 1) == UInt8('>')
        style = Char(codeunit(rest, 1))
        hdr   = n_cu(rest) > 1 ? suffix_after_char(rest, 1) : ""
        ps.lineno += 1   # consume block-scalar header (current line already consumed by caller)
        return YAMLString(parse_block_scalar(ps, style, hdr))

    elseif codeunit(rest, 1) == UInt8('{')
        v, _ = parse_flow_mapping(ps, rest, 2); return v

    elseif codeunit(rest, 1) == UInt8('[')
        v, _ = parse_flow_sequence(ps, rest, 2); return v

    elseif codeunit(rest, 1) == UInt8('&')
        # &anchorname <value>
        j = 2
        while j <= n_cu(rest) && codeunit(rest,j) != UInt8(' ') && codeunit(rest,j) != UInt8('\t')
            j += 1
        end
        aname = rest[2:j-1]   # anchor names are plain ASCII
        inner = lstrip_ascii(j <= n_cu(rest) ? rest[j:end] : "")
        v = if isempty(inner) || codeunit(inner, 1) == UInt8('#')
            ni = peek_indent(ps)
            ni > block_indent ? parse_node(ps, ni) : YAMLNull
        else
            parse_scalar(strip_inline_comment(inner))
        end
        ps.anchors[aname] = v; return v

    elseif codeunit(rest, 1) == UInt8('*')
        j = 2
        while j <= n_cu(rest) && codeunit(rest,j) != UInt8(' ') && codeunit(rest,j) != UInt8('#')
            j += 1
        end
        return get(ps.anchors, rest[2:j-1], YAMLNull)

    else
        return parse_scalar(rest)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Block sequence
# ─────────────────────────────────────────────────────────────────────────────

function parse_block_sequence(ps::ParseState, seq_indent::Int)::YAMLValue
    items = YAMLValue[]
    while !at_end(ps)
        skip_empty!(ps); at_end(ps) && break
        l = current_line(ps)
        indent, content = split_indent(l)
        indent != seq_indent && break
        isempty(content) && (ps.lineno += 1; continue)
        codeunit(content, 1) != UInt8('-') && break
        if n_cu(content) >= 2
            b2 = codeunit(content, 2)
            b2 != UInt8(' ') && b2 != UInt8('\t') && break
        end

        ps.lineno += 1   # consume '-' line

        # Content after '- ' (suffix_after_char handles multi-byte-safe skip of '-')
        inline = lstrip_ascii(n_cu(content) > 1 ? suffix_after_char(content, 1) : "")
        inline = strip_inline_comment(inline)

        val::YAMLValue = if isempty(inline) || codeunit(inline, 1) == UInt8('#')
            next_ind = peek_indent(ps)
            next_ind > seq_indent ? parse_node(ps, next_ind) : YAMLNull

        elseif codeunit(inline, 1) == UInt8('-') &&
               (n_cu(inline) == 1 || codeunit(inline, 2) == UInt8(' '))
            # fake = " "^(seq_indent + 2) * inline
            new_ind = seq_indent ÷ 2
            buf = IOBuffer()
            for i in 1:new_ind
                write(buf, ' ')
            end
            write(buf, inline)
            fake = String(take!(buf))
            insert!(ps.lines, ps.lineno, fake)
            parse_block_sequence(ps, seq_indent + 2)

        else
            # Check for inline mapping key first, then delegate to _parse_value_after_colon
            k, _ = find_mapping_key(inline)
            if k !== nothing
                new_ind   = seq_indent + 2
                # fake_line = " "^new_ind * inline
                buf = IOBuffer()
                for i in 1:new_ind
                    write(buf, ' ')
                end
                write(buf, inline)
                fake_line = String(take!(buf))

                insert!(ps.lines, ps.lineno, fake_line)
                parse_node(ps, new_ind)
            else
                _parse_value_after_colon(ps, inline, seq_indent)
            end
        end

        push!(items, val)
    end
    return YAMLArray(items)
end

# ─────────────────────────────────────────────────────────────────────────────
# Node dispatcher
# ─────────────────────────────────────────────────────────────────────────────

function parse_node(ps::ParseState, min_indent::Int)::YAMLValue
    skip_empty!(ps); at_end(ps) && return YAMLNull
    l = current_line(ps)
    indent, content = split_indent(l)
    indent < min_indent && return YAMLNull
    isempty(content) && return YAMLNull

    if startswith(content, "---") || startswith(content, "...")
        nc = n_cu(content)
        ok = nc == 3 || codeunit(content,4) == UInt8(' ') ||
                        codeunit(content,4) == UInt8('\t') || codeunit(content,4) == UInt8('#')
        if ok; ps.lineno += 1; return parse_node(ps, min_indent); end
    end

    b1 = codeunit(content, 1)
    if b1 == UInt8('-') && (n_cu(content) == 1 || codeunit(content,2) == UInt8(' ') ||
                                                   codeunit(content,2) == UInt8('\t'))
        return parse_block_sequence(ps, indent)
    end

    key, _ = find_mapping_key(content)
    key !== nothing && return parse_block_mapping(ps, indent)

    return parse_line_scalar(ps, min_indent)
end

# ─────────────────────────────────────────────────────────────────────────────
# Scalar line parser
# ─────────────────────────────────────────────────────────────────────────────

function parse_line_scalar(ps::ParseState, min_indent::Int)::YAMLValue
    skip_empty!(ps); at_end(ps) && return YAMLNull
    l = current_line(ps)
    indent, content = split_indent(l)
    indent < min_indent && return YAMLNull
    isempty(content) && return YAMLNull

    b1 = codeunit(content, 1)

    if b1 == UInt8('&')
        ps.lineno += 1
        j = 2; nn = n_cu(content)
        while j <= nn && codeunit(content,j) != UInt8(' ') && codeunit(content,j) != UInt8('\t')
            j += 1
        end
        aname = content[2:j-1]
        inner = lstrip_ascii(j <= nn ? content[j:end] : "")
        val = isempty(inner) || codeunit(inner,1) == UInt8('#') ?
              parse_node(ps, min_indent) :
              parse_scalar(strip_inline_comment(inner))
        ps.anchors[aname] = val; return val
    end

    if b1 == UInt8('*')
        ps.lineno += 1; j = 2; nn = n_cu(content)
        while j <= nn && codeunit(content,j) != UInt8(' ') && codeunit(content,j) != UInt8('#')
            j += 1
        end
        return get(ps.anchors, content[2:j-1], YAMLNull)
    end

    if b1 == UInt8('"')
        ps.lineno += 1; str, _ = parse_double_quoted(content, 2); return YAMLString(str)
    end
    if b1 == UInt8('\'')
        ps.lineno += 1; str, _ = parse_single_quoted(content, 2); return YAMLString(str)
    end
    if b1 == UInt8('|') || b1 == UInt8('>')
        style = Char(b1)
        hdr   = n_cu(content) > 1 ? suffix_after_char(content, 1) : ""
        ps.lineno += 1
        return YAMLString(parse_block_scalar(ps, style, hdr))
    end
    if b1 == UInt8('{')
        ps.lineno += 1; v, _ = parse_flow_mapping(ps, content, 2); return v
    end
    if b1 == UInt8('[')
        ps.lineno += 1; v, _ = parse_flow_sequence(ps, content, 2); return v
    end

    # Plain (possibly multi-line) scalar
    ps.lineno += 1
    parts = String[strip_inline_comment(content)]
    while !at_end(ps)
        skip_empty!(ps); at_end(ps) && break
        l2 = current_line(ps)
        ind2, c2 = split_indent(l2)
        ind2 <= min_indent && break
        isempty(c2) && break
        codeunit(c2,1) == UInt8('#') && break
        k2, _ = find_mapping_key(c2); k2 !== nothing && break
        if codeunit(c2,1) == UInt8('-') && (n_cu(c2) == 1 || codeunit(c2,2) == UInt8(' '))
            break
        end
        push!(parts, strip_inline_comment(c2)); ps.lineno += 1
    end
    return parse_scalar(join(parts, " "))
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

function loads(yaml::String)::YAMLValue
    ps = ParseState(yaml); skip_empty!(ps)
    at_end(ps) && return YAMLNull
    return parse_node(ps, 0)
end

function parsefile(path::String)::YAMLValue
    return loads(read(path, String))
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

function get_value(d::Dict{String,YAMLValue}, key, ::Type{Array});   return as_array(d[key]); end
function get_value(d::Dict{String,YAMLValue}, key, ::Type{<:Dict});  return as_dict(d[key]);  end
function get_value(d::Dict{String,YAMLValue}, key, ::Type{Float64}); return as_float(d[key]); end
function get_value(d::Dict{String,YAMLValue}, key, ::Type{Int});     return as_int(d[key]);   end
function get_value(d::Dict{String,YAMLValue}, key, ::Type{String});  return as_string(d[key]);end

function to_dict(v::String)
    return v
end

function to_dict(v::YAMLValue)
    if     v.tag === TAG_NULL;   return nothing
    elseif v.tag === TAG_BOOL;   return v.bval
    elseif v.tag === TAG_INT;    return v.ival
    elseif v.tag === TAG_FLOAT;  return v.fval
    elseif v.tag === TAG_STRING; return v.sval
    elseif v.tag === TAG_ARRAY
        vals = Any[]
        for item in v.arr; push!(vals, to_dict(item)); end
        return vals
    else
        d = Dict{String,Any}()
        for (k, val) in v.dict; d[k] = to_dict(val); end
        return d
    end
end

function to_dict(d::Dict{String,YAMLValue})
    out = Dict{String,Any}()
    for (k, v) in d; out[k] = to_dict(v); end
    return out
end

end # module SimpleYAML
