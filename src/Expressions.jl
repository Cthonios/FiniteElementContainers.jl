module Expressions

import DynamicExpressions.NodeModule: DEFAULT_MAX_DEGREE
using DocStringExtensions
using DynamicExpressions
using StaticArrays

########################################################
# Helper types
########################################################
const Name     = Union{Nothing, Symbol}
const NameID   = Union{Nothing, Int}
const Operator = Union{Nothing, Int}
const Value{T} = Union{Nothing, T}

########################################################
# Enums, TODO should we make actual enums or cenum?
########################################################
# expression tokens
const ASSIGNMENT  = 1
const BINARY      = 2
const CALL        = 3
const COMMA       = 4
const END         = 5
const EXPRESSION  = 6
const IDENTIFIER  = 7
const LEFT_PAREN  = 8
const NUMBER      = 9
const OPERATOR    = 10
const RIGHT_PAREN = 11
const SEMICOLON   = 12
const UNARY       = 13
const VARIABLE    = 14

# internally supported analytic functions
const FUNC_COS   = 1
const FUNC_COSH  = 2
const FUNC_EXP   = 3
const FUNC_LOG   = 4
const FUNC_MINUS = 5
const FUNC_SIN   = 6
const FUNC_SINH  = 7
const FUNC_SQRT  = 8
const FUNC_TAN   = 9
const FUNC_TANH  = 10

_minus(x) = -x

# all supported operators
const operators = OperatorEnum(
    1 => (cos, cosh, exp, log, _minus, sin, sinh, sqrt, tan, tanh),
    2 => (+, -, *, /, ^)
)
const operators_type = typeof(operators)
const ntuple_type = NamedTuple{(:operators, :var_names), Tuple{operators_type, Vector{String}}}

const NULL_OPERATOR = 1

const UNARY_MINUS = FUNC_MINUS

const BINARY_PLUS     = 1
const BINARY_MINUS    = 2
const BINARY_MULTIPLY = 3
const BINARY_DIVIDE   = 4
const BINARY_POWER    = 5

########################################################
# Tokens
########################################################
struct Token{T <: Number}
    id::Int
    name::Name
    op::Operator
    value::Value{T}

    function Token{T}(id::Int) where T <: Number
        new{T}(id, nothing, nothing, -1.0)
    end

    function Token{T}(value::T) where T <: Number
        new{T}(NUMBER, nothing, nothing, value)
    end

    function Token{T}(id::Int, name::Symbol, op::Int) where T <: Number
        new{T}(id, name, op, nothing)
    end
end

########################################################
# Lexer
########################################################
struct InvalidScientificNotationError <: Exception
    input::String
end

function Base.showerror(io::IO, e::InvalidScientificNotationError) 
    print(io, "Invalid conversion to scientific notation for input $(e.input)")
end

function invalid_scientific_notation_error(num)
    err = InvalidScientificNotationError(string(num))
    throw(err)
end

mutable struct Lexer
    string::String
    index::Int

    function Lexer(string::String)
        new(string, firstindex(string))
    end
end

function _advance!(l::Lexer)
    c = _peek(l)
    l.index += 1
    return c
end

function _next_token(l::Lexer, ::Type{T} = Float64) where T <: Number
    _skip_ws!(l)

    c = _peek(l)

    c == '\0' && return Token{T}(END)

    if isdigit(c)
        return Token{T}(_read_number(l))
    elseif isletter(c)
        return Token{T}(IDENTIFIER, _read_identifier(l), NULL_OPERATOR)
    elseif c == '+'
        _advance!(l); return Token{T}(OPERATOR, :null, BINARY_PLUS)
    elseif c == '-'
        _advance!(l); return Token{T}(OPERATOR, :null, BINARY_MINUS)
    elseif c == '*'
        _advance!(l); return Token{T}(OPERATOR, :null, BINARY_MULTIPLY)
    elseif c == '/'
        _advance!(l); return Token{T}(OPERATOR, :null, BINARY_DIVIDE)
    elseif c == '^'
        _advance!(l); return Token{T}(OPERATOR, :null, BINARY_POWER)
    elseif c == '('
        _advance!(l); return Token{T}(LEFT_PAREN)
    elseif c == ')'
        _advance!(l); return Token{T}(RIGHT_PAREN)
    elseif c == ','
        _advance!(l); return Token{T}(COMMA)
    elseif c == ';'
        _advance!(l); return Token{T}(SEMICOLON)
    else
        error("Unknown character: $c")
    end
end

function _peek(l::Lexer)
    l.index > lastindex(l.string) && return '\0'
    return l.string[l.index]
end

function _read_number(l::Lexer, ::Type{T} = Float64) where T
    start = l.index

    # integer / decimal part
    while isdigit(_peek(l)) || _peek(l) == '.'
        _advance!(l)
    end

    # exponent part
    if _peek(l) == 'e' || _peek(l) == 'E'
        _advance!(l)

        # optional sign
        if _peek(l) == '+' || _peek(l) == '-'
            _advance!(l)
        end

        # must have at least one digit
        if !isdigit(_peek(l))
            invalid_scientific_notation_error(_peek(l))
        end

        while isdigit(_peek(l))
            _advance!(l)
        end
    end

    return parse(T, SubString(l.string, start, l.index - 1))
end

function _read_identifier(l::Lexer)
    start = l.index
    while isletter(_peek(l))
        _advance!(l)
    end
    return Symbol(SubString(l.string, start, l.index - 1))
end

function _skip_ws!(l)
    while isspace(_peek(l))
        _advance!(l)
    end
end

########################################################
# PRATT Parser
########################################################
mutable struct Parser{T}
    current::Token{T}
    lexer::Lexer
    parameter_names::Vector{String}
    var_names::Vector{String}

    function Parser{T}(string::String, var_names::Vector{String}) where T <: Number
        lexer = Lexer(string)
        current = _next_token(lexer)
        parameter_names = String[]
        new{T}(current, lexer, parameter_names, var_names)
    end
end

function _advance!(p::Parser)
    p.current = _next_token(p.lexer)
end

function _find_parameters(p::Parser)
    while p.current.id != END
        _advance!(p)
        if p.current.id == IDENTIFIER
            if string(p.current.name) in p.var_names
                continue
            else
                @info "Found a parameter"
                push!(p.parameter_names, string(p.current.name))
            end
        end
    end
end

function _func_id(func_name::String)
    if func_name == "cos"
        return FUNC_COS
    elseif func_name == "cosh"
        return FUNC_COSH
    elseif func_name == "exp"
        return FUNC_EXP
    elseif func_name == "log"
        return FUNC_LOG
    elseif func_name == "sin"
        return FUNC_SIN
    elseif func_name == "sinh"
        return FUNC_SINH
    elseif func_name == "sqrt"
        return FUNC_SQRT
    elseif func_name == "tan"
        return FUNC_TAN
    elseif func_name == "tanh"
        return FUNC_TANH
    else
        @assert false "Function $func_name not supported internally"
    end
end

function _lbp(token::Token)
    if token.id == END
        return 0
    elseif token.id == RIGHT_PAREN
        return 0
    elseif token.id == OPERATOR
        if token.op == BINARY_PLUS || token.op == BINARY_MINUS
            return 10
        elseif token.op == BINARY_MULTIPLY || token.op == BINARY_DIVIDE
            return 20
        elseif token.op == BINARY_POWER
            return 30
        else    
            error("Unknown op")
        end
    end
end

function _led(p::Parser, token::Token, left, ::Type{T}) where T <: Number
    right = _parse_statement(p, _lbp(token))
    return Node{T}(; op = token.op, l = left, r = right)
end

function _nud(p::Parser, t::Token, ::Type{T}) where T <: Number
    if t.id == IDENTIFIER
        if p.current.id == LEFT_PAREN
            _advance!(p) # consume '('
            args = Node{T, DEFAULT_MAX_DEGREE}[]
            if !(p.current.id == RIGHT_PAREN)
                while true
                    push!(args, _parse_statement(p, 0))
                    if p.current.id == COMMA
                        _advance!(p)
                    else
                        break
                    end
                end
            end
            p.current.id == RIGHT_PAREN || error("Expected )")
            @assert length(args) == 1 "We're likely only supporting single argument calls right now"
            _advance!(p)
            # TODO if we support multi args later drop the [1]
            func_name = string(t.name)
            func_id = _func_id(func_name)
            return Node{T}(; op = func_id, l = args[1])
        else
            if string(t.name) in p.var_names
                name_id = findfirst(x -> x == string(t.name), p.var_names)
                return Node{T}(; feature = name_id)
            elseif string(t.name) == "pi"
                return Node{T}(; val = π)
            # elseif string(t.name) in p.parameter_names
            #     name_id = findfirst(x -> x == string(t.name), p.parameter_names)
            #     return ASTNode{T}(nothing, PARAMETER, nothing, name_id, nothing, nothing, )
            else
                @assert false "Need to support parameters"
            end
        end
    elseif t.id == LEFT_PAREN
        expr = _parse_statement(p, 0)
        if p.current.id != RIGHT_PAREN
            error("Expected )")
        end
        _advance!(p)
        return expr
    elseif t.id == NUMBER
        return Node{T}(; val = t.value)
    elseif t.id == OPERATOR
        if t.op == BINARY_MINUS
            # Unary minus.  Right-binding-power 25 sits between `*`/`/` (20)
            # and `^` (30) — matches standard math precedence so `-t^2` parses
            # as `-(t^2)` rather than `(-t)^2`.
            val = _parse_statement(p, 25)
            return Node{T}(; op = UNARY_MINUS, l = val)
        else
            error("Unexpected operator in _nud. Found operator $(t.op)")
        end
    else
        error("Should this happen?")
    end
end

function _parse_statement(p::Parser, rbp::Int, ::Type{T} = Float64) where T <: Number
    t = p.current
    _advance!(p)

    left = _nud(p, t, T)

    while rbp < _lbp(p.current)
        t = p.current
        _advance!(p)
        left = _led(p, t, left, T)
    end

    return left
end

function _reset!(p::Parser{T}) where T <: Number
    p.lexer.index = 1
    p.current = _next_token(p.lexer, T)
    return nothing
end

########################################################
# Front facing APIinc
########################################################
"""
$(TYPEDEF)
"""
abstract type AbstractExpressionFunction{T, N, D} <: Function end

########################################################
# Flat expression-tree representation
#
# `ScalarExpressionFunction` stores its expression tree as a fixed-size
# `NTuple` of `FlatNode` records — a struct of plain integer fields and a
# value of type `T`.  The whole thing is `isbits`, which means:
#
#   * It passes through KernelAbstractions kernels as a value argument and
#     can be called on the GPU directly — no host↔device round-trip per
#     time step for BC evaluation.
#   * It survives `juliac --trim`: no closures, no `@eval`, no
#     `RuntimeGeneratedFunctions`, just data.
#
# `FEC_EXPR_MAX_NODES` is the maximum tree size.  Carina's largest
# inlined expression today is ~25 nodes, but second-derivative trees from
# the symbolic differentiator can grow to ~100 nodes for Gaussian-pulse
# BCs; 256 gives 2-3× headroom at ~6 KB per function — negligible memory
# for typical BC/IC counts.  Trees that exceed the cap raise at
# construction time.
########################################################

const FEC_EXPR_MAX_NODES = 256

"""
$(TYPEDEF)
$(TYPEDFIELDS)

One node of a flattened expression tree.  Children are referenced by
1-based index into the parent array; index 0 marks "no child".  Leaves
carry either a constant `val` or a `feature` (1-based variable index).
"""
struct FlatNode{T <: Number}
    degree::UInt8     # 0 = leaf, 1 = unary, 2 = binary
    op::UInt8         # FUNC_* or BINARY_* op code; 0 for leaves
    constant::Bool    # leaf form: true ⇒ use `val`, false ⇒ use `feature`
    val::T            # leaf value when `constant`
    feature::UInt16   # leaf variable index (1-based) when !`constant`
    l_idx::UInt16     # index of left  child (0 if leaf)
    r_idx::UInt16     # index of right child (0 if leaf or unary)
end

@inline FlatNode{T}() where T <: Number =
    FlatNode{T}(UInt8(0), UInt8(0), false, zero(T), UInt16(0), UInt16(0), UInt16(0))

# Flatten a recursive `Node{T, D}` (produced by FEC's Pratt parser) into a
# preorder NTuple of `FlatNode{T}` records.  The first entry is the root;
# children follow in DFS order.
function _flatten_visit!(buf::Vector{FlatNode{T}}, node::Node{T, D})::UInt16 where {T, D}
    if node.degree == 0
        if node.constant
            push!(buf, FlatNode{T}(UInt8(0), UInt8(0), true,
                                   T(node.val), UInt16(0),
                                   UInt16(0), UInt16(0)))
        else
            push!(buf, FlatNode{T}(UInt8(0), UInt8(0), false,
                                   zero(T), UInt16(node.feature),
                                   UInt16(0), UInt16(0)))
        end
        return UInt16(length(buf))
    elseif node.degree == 1
        push!(buf, FlatNode{T}())                   # reserve own slot
        my_idx = length(buf)
        l = _flatten_visit!(buf, node.l)
        buf[my_idx] = FlatNode{T}(UInt8(1), UInt8(node.op), false,
                                  zero(T), UInt16(0), l, UInt16(0))
        return UInt16(my_idx)
    else                                            # degree == 2
        push!(buf, FlatNode{T}())                   # reserve own slot
        my_idx = length(buf)
        l = _flatten_visit!(buf, node.l)
        r = _flatten_visit!(buf, node.r)
        buf[my_idx] = FlatNode{T}(UInt8(2), UInt8(node.op), false,
                                  zero(T), UInt16(0), l, r)
        return UInt16(my_idx)
    end
end

@generated function _vector_to_ntuple(v::Vector{T}) where {T}
    Expr(:tuple, [:(@inbounds(v[$i])) for i=1:FEC_EXPR_MAX_NODES]...)
end

function _flatten(root::Node{T, D}) where {T, D}
    buf = FlatNode{T}[]
    sizehint!(buf, FEC_EXPR_MAX_NODES)
    _flatten_visit!(buf, root)
    n_active = length(buf)
    n_active <= FEC_EXPR_MAX_NODES || error(
        "expression too large: $n_active nodes (max $FEC_EXPR_MAX_NODES)"
    )
    # Pad to fixed length with default nodes so the resulting NTuple type
    # has a constant size at the type level.
    while length(buf) < FEC_EXPR_MAX_NODES
        push!(buf, FlatNode{T}())
    end
    # nodes = NTuple{FEC_EXPR_MAX_NODES, FlatNode{T}}(buf)
    nodes = _vector_to_ntuple(buf)
    return nodes, UInt16(n_active)
end

# Inverse used by `differentiate`: rebuild a recursive `Node{T, D}` from a
# flat NTuple so the existing recursive symbolic differentiator can run on
# it.  Called once per `differentiate` invocation, off the GPU.
function _unflatten(nodes::NTuple{N, FlatNode{T}}, idx::Integer = 1) where {N, T}
    n = nodes[idx]
    if n.degree == 0
        if n.constant
            return Node{T, DEFAULT_MAX_DEGREE}(; val = n.val)
        else
            return Node{T, DEFAULT_MAX_DEGREE}(; feature = Int(n.feature))
        end
    elseif n.degree == 1
        l = _unflatten(nodes, n.l_idx)
        return Node{T, DEFAULT_MAX_DEGREE}(; op = Int(n.op), l = l)
    else
        l = _unflatten(nodes, n.l_idx)
        r = _unflatten(nodes, n.r_idx)
        return Node{T, DEFAULT_MAX_DEGREE}(; op = Int(n.op), l = l, r = r)
    end
end

# Op-code dispatch — open-coded if/elseif so the compiler can inline
# branches inside KA kernels without resorting to a function table.
@inline function _apply_unary_op(::Type{T}, op::UInt8, u::T) where T <: Number
    if     op == UInt8(FUNC_MINUS); return -u
    elseif op == UInt8(FUNC_COS);   return cos(u)
    elseif op == UInt8(FUNC_COSH);  return cosh(u)
    elseif op == UInt8(FUNC_EXP);   return exp(u)
    elseif op == UInt8(FUNC_LOG);   return log(u)
    elseif op == UInt8(FUNC_SIN);   return sin(u)
    elseif op == UInt8(FUNC_SINH);  return sinh(u)
    elseif op == UInt8(FUNC_SQRT);  return sqrt(u)
    elseif op == UInt8(FUNC_TAN);   return tan(u)
    elseif op == UInt8(FUNC_TANH);  return tanh(u)
    end
    return T(NaN)
end

@inline function _apply_binary_op(::Type{T}, op::UInt8, u::T, v::T) where T <: Number
    if     op == UInt8(BINARY_PLUS);     return u + v
    elseif op == UInt8(BINARY_MINUS);    return u - v
    elseif op == UInt8(BINARY_MULTIPLY); return u * v
    elseif op == UInt8(BINARY_DIVIDE);   return u / v
    elseif op == UInt8(BINARY_POWER);    return u ^ v
    end
    return T(NaN)
end

# Recursive evaluator over the flat NTuple.  Depth is bounded by the
# expression tree height (≤ ~10 for the expressions Carina uses today),
# so GPUCompiler handles the recursion without stack pressure.
function _eval_node(nodes::NTuple{N, FlatNode{T}}, idx::UInt16,
                    vars) where {N, T}
    n = nodes[idx]
    if n.degree == 0
        if n.constant
            return n.val
        else
            return T(vars[n.feature])
        end
    elseif n.degree == 1
        u = _eval_node(nodes, n.l_idx, vars)
        return _apply_unary_op(T, n.op, u)
    else
        u = _eval_node(nodes, n.l_idx, vars)
        v = _eval_node(nodes, n.r_idx, vars)
        return _apply_binary_op(T, n.op, u, v)
    end
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Scalar expression function as a flat, `isbits` value — usable as a
KernelAbstractions kernel argument and trim-mode safe under `juliac`.
The trailing variable is conventionally time; FEC's juliac-safe
`DirichletBCs` constructor uses `num_vars` as the time-derivative index.
"""
struct ScalarExpressionFunction{T <: Number} <: AbstractExpressionFunction{T, FlatNode{T}, ntuple_type}
    nodes::NTuple{FEC_EXPR_MAX_NODES, FlatNode{T}}
    n_active::UInt16
    num_vars::UInt8

    """
    $(TYPEDSIGNATURES)

    Parse `string` as an expression in the variable namespace `var_names`
    and store the resulting tree in flat form.  `var_names` is consumed by
    the parser to bind identifiers to feature indices; it is not retained
    on the resulting function.
    """
    function ScalarExpressionFunction{T}(string::String, var_names::Vector{String}) where T <: Number
        p = Parser{T}(string, var_names)
        _reset!(p)
        ast = _parse_statement(p, 0)
        nodes, n_active = _flatten(ast)
        new{T}(nodes, n_active, UInt8(length(var_names)))
    end

    """
    $(TYPEDSIGNATURES)

    Build a `ScalarExpressionFunction` from a prebuilt flat NTuple — used
    internally by [`differentiate`](@ref) to wrap the result of a tree
    rewrite without round-tripping through the parser.
    """
    function ScalarExpressionFunction{T}(
        nodes::NTuple{FEC_EXPR_MAX_NODES, FlatNode{T}},
        n_active::UInt16,
        num_vars::Integer
    ) where T <: Number
        new{T}(nodes, n_active, UInt8(num_vars))
    end
end

Base.eltype(::ScalarExpressionFunction{T}) where T <: Number = T

# Single scalar
function (f::ScalarExpressionFunction{T})(var::T) where T <: Number
    @assert f.num_vars == 1
    return _eval_node(f.nodes, UInt16(1), SVector{1, T}(var))
end

# Vector of variable values (no `t` overload)
function (f::ScalarExpressionFunction{T})(vars::AbstractVector{T}) where T <: Number
    @assert length(vars) == Int(f.num_vars) "expected $(Int(f.num_vars)) variables, got $(length(vars))"
    return _eval_node(f.nodes, UInt16(1), vars)
end

# IC-style call: ND spatial coords (no time variable in the expression).
function (f::ScalarExpressionFunction{T})(X::SVector{ND, T}) where {ND, T <: Number}
    @assert Int(f.num_vars) == ND "You need $ND variables for this function"
    return _eval_node(f.nodes, UInt16(1), X)
end

# BC-style call: ND spatial coords + scalar time, packed into a stack-
# allocated SVector so the call survives KA kernels.
function (f::ScalarExpressionFunction{T})(X::SVector{ND, T}, t::T) where {ND, T <: Number}
    @assert Int(f.num_vars) == ND + 1 "You need $(ND + 1) variables for this function"
    if ND == 1
        vars = SVector{2, T}(X[1], t)
    elseif ND == 2
        vars = SVector{3, T}(X[1], X[2], t)
    elseif ND == 3
        vars = SVector{4, T}(X[1], X[2], X[3], t)
    else
        # Generic path (very unlikely; covered for completeness).  Allocates.
        vars = T[X...; t]
    end
    return _eval_node(f.nodes, UInt16(1), vars)
end

"""
$(METHODLIST)
$(TYPEDFIELDS)
$(TYPEDEF)
"""
struct VectorExpressionFunction{N, T <: Number} <: AbstractExpressionFunction{T, Node{T, DEFAULT_MAX_DEGREE}, ntuple_type}
    exprs::SVector{N, ScalarExpressionFunction{T}}
    num_vars::Int

    function VectorExpressionFunction{N, T}(
        strings::Vector{String},
        var_names::Vector{String},
    ) where {N, T<:Number}
        @assert length(strings) == N
        funcs = ntuple(
            i -> ScalarExpressionFunction{T}(strings[i], var_names),
            Val(N),
        )
        return new{N, T}(SVector{N}(funcs), length(var_names))
    end
end

function (f::VectorExpressionFunction)(var::T) where T <: Number
    return map(func -> func(var), f.exprs)
end

function (f::VectorExpressionFunction)(vars::AbstractVector{T}) where T <: Number
    return map(func -> func(vars), f.exprs)
end

function (f::VectorExpressionFunction)(X::SVector{ND, T}) where {ND, T <: Number}
    return map(func -> func(X), f.exprs)
end

function (f::VectorExpressionFunction)(X::SVector{ND, T}, t::T) where {ND, T <: Number}
    return map(func -> func(X, t), f.exprs)
end

########################################################
# Symbolic differentiation on the recursive Node form.
#
# The differentiator is a pure tree-rewrite over FEC's closed grammar
# (10 unary + 5 binary ops).  It operates on `Node{T, D}` because the
# constant-folding helpers compose recursively; the boundary with
# `ScalarExpressionFunction` is `_unflatten` / `_flatten`, called once
# per `differentiate` invocation.
########################################################

@inline _is_const(n::Node) = n.degree == 0 && n.constant
@inline _is_zero(n::Node)  = _is_const(n) && iszero(n.val)
@inline _is_one(n::Node)   = _is_const(n) && isone(n.val)

# Smart constructors that fold trivial constants so derivative trees stay
# proportional in size to the input.
function _add(a::Node{T, D}, b::Node{T, D}) where {T, D}
    _is_zero(a) && return b
    _is_zero(b) && return a
    return Node{T, D}(; op = BINARY_PLUS, l = a, r = b)
end

function _sub(a::Node{T, D}, b::Node{T, D}) where {T, D}
    _is_zero(b) && return a
    _is_zero(a) && return _neg(b)
    return Node{T, D}(; op = BINARY_MINUS, l = a, r = b)
end

function _mul(a::Node{T, D}, b::Node{T, D}) where {T, D}
    (_is_zero(a) || _is_zero(b)) && return Node{T, D}(; val = zero(T))
    _is_one(a) && return b
    _is_one(b) && return a
    return Node{T, D}(; op = BINARY_MULTIPLY, l = a, r = b)
end

function _div(a::Node{T, D}, b::Node{T, D}) where {T, D}
    _is_zero(a) && return Node{T, D}(; val = zero(T))
    _is_one(b) && return a
    return Node{T, D}(; op = BINARY_DIVIDE, l = a, r = b)
end

function _neg(a::Node{T, D}) where {T, D}
    _is_zero(a) && return a
    return Node{T, D}(; op = UNARY_MINUS, l = a)
end

function _pow_int(a::Node{T, D}, k::Int) where {T, D}
    k == 1 && return a
    return Node{T, D}(; op = BINARY_POWER, l = a, r = Node{T, D}(; val = T(k)))
end

# Recursive tree-rewrite differentiator.  Returns a new tree representing
# ∂node/∂x_{var_idx}.
function _differentiate(node::Node{T, D}, var_idx::Int) where {T, D}
    if node.degree == 0
        if node.constant
            return Node{T, D}(; val = zero(T))
        else
            return Node{T, D}(;
                val = node.feature == var_idx ? one(T) : zero(T)
            )
        end
    elseif node.degree == 1
        u  = node.l
        du = _differentiate(u, var_idx)
        op = node.op
        if op == UNARY_MINUS
            return _neg(du)
        elseif op == FUNC_COS
            sin_u = Node{T, D}(; op = FUNC_SIN, l = u)
            return _mul(_neg(sin_u), du)
        elseif op == FUNC_COSH
            sinh_u = Node{T, D}(; op = FUNC_SINH, l = u)
            return _mul(sinh_u, du)
        elseif op == FUNC_EXP
            return _mul(node, du)
        elseif op == FUNC_LOG
            return _div(du, u)
        elseif op == FUNC_SIN
            cos_u = Node{T, D}(; op = FUNC_COS, l = u)
            return _mul(cos_u, du)
        elseif op == FUNC_SINH
            cosh_u = Node{T, D}(; op = FUNC_COSH, l = u)
            return _mul(cosh_u, du)
        elseif op == FUNC_SQRT
            two_sqrt = _mul(Node{T, D}(; val = T(2)), node)
            return _div(du, two_sqrt)
        elseif op == FUNC_TAN
            cos_u = Node{T, D}(; op = FUNC_COS, l = u)
            return _div(du, _pow_int(cos_u, 2))
        elseif op == FUNC_TANH
            cosh_u = Node{T, D}(; op = FUNC_COSH, l = u)
            return _div(du, _pow_int(cosh_u, 2))
        end
        error("differentiate: unhandled unary op $op")
    elseif node.degree == 2
        u, v = node.l, node.r
        op   = node.op
        if op == BINARY_PLUS
            return _add(_differentiate(u, var_idx), _differentiate(v, var_idx))
        elseif op == BINARY_MINUS
            return _sub(_differentiate(u, var_idx), _differentiate(v, var_idx))
        elseif op == BINARY_MULTIPLY
            du, dv = _differentiate(u, var_idx), _differentiate(v, var_idx)
            return _add(_mul(du, v), _mul(u, dv))
        elseif op == BINARY_DIVIDE
            du, dv = _differentiate(u, var_idx), _differentiate(v, var_idx)
            num = _sub(_mul(du, v), _mul(u, dv))
            return _div(num, _pow_int(v, 2))
        elseif op == BINARY_POWER
            if _is_const(v)
                # d/dx(u^c) = c · u^(c-1) · u'
                du     = _differentiate(u, var_idx)
                c      = v
                c_minus_1 = Node{T, D}(; val = c.val - one(T))
                u_pow  = Node{T, D}(; op = BINARY_POWER, l = u, r = c_minus_1)
                return _mul(_mul(c, u_pow), du)
            elseif _is_const(u)
                # d/dx(c^v) = c^v · log(c) · v'
                dv    = _differentiate(v, var_idx)
                log_c = Node{T, D}(; val = log(u.val))
                return _mul(_mul(node, log_c), dv)
            else
                # general: d/dx(u^v) = u^v · (v' log u + v u'/u)
                du = _differentiate(u, var_idx)
                dv = _differentiate(v, var_idx)
                log_u = Node{T, D}(; op = FUNC_LOG, l = u)
                term1 = _mul(dv, log_u)
                term2 = _div(_mul(v, du), u)
                return _mul(node, _add(term1, term2))
            end
        end
        error("differentiate: unhandled binary op $op")
    end
    error("differentiate: unhandled degree $(node.degree)")
end

"""
$(TYPEDSIGNATURES)

Return the symbolic derivative of `f` with respect to the variable whose
1-based feature index is `var_idx`.  Differentiation is implemented as a
recursive tree rewrite over FEC's closed grammar (10 unary + 5 binary
operators) — no dependency on ForwardDiff, Zygote, or Symbolics.

The result is a fresh `ScalarExpressionFunction` over the same variable
slots; the trailing variable is conventionally time.
"""
function differentiate(f::ScalarExpressionFunction{T}, var_idx::Integer) where T
    @assert 1 <= Int(var_idx) <= Int(f.num_vars) "var_idx $(var_idx) out of range 1..$(Int(f.num_vars))"
    tree        = _unflatten(f.nodes)
    deriv_tree  = _differentiate(tree, Int(var_idx))
    nodes, n_active = _flatten(deriv_tree)
    return ScalarExpressionFunction{T}(nodes, n_active, f.num_vars)
end

"""
$(TYPEDSIGNATURES)

Convenience overload that resolves `var_name` against an explicit
`var_names` list, then delegates to the integer form.  Useful when the
caller still has the var-name list in scope (typical at TOML parse time);
runtime hot paths should call the integer form directly.
"""
function differentiate(f::ScalarExpressionFunction{T},
                       var_names::AbstractVector{<:AbstractString},
                       var_name::AbstractString) where T
    idx = findfirst(==(var_name), var_names)
    # TODO add error message
    @assert idx !== nothing
    return differentiate(f, idx)
end

end # module
