module Expressions

using StaticArrays

########################################################
# Helper types
########################################################
const Name = Union{Nothing, Symbol}
const NameID = Union{Nothing, Int}
const Operator = Union{Nothing, Symbol}
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
const FUNC_NAMES = String["exp", "sin"]
const FUNC_EXP = 1
const FUNC_SIN = 2

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

    function Token{T}(id::Int, name::Symbol, op::Symbol) where T <: Number
        new{T}(id, name, op, nothing)
    end
end

########################################################
# Lexer
########################################################
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
        return Token{T}(IDENTIFIER, _read_identifier(l), :null)
    elseif c == '+'
        _advance!(l); return Token{T}(OPERATOR, :null, :+)
    elseif c == '-'
        _advance!(l); return Token{T}(OPERATOR, :null, :-)
    elseif c == '*'
        _advance!(l); return Token{T}(OPERATOR, :null, :*)
    elseif c == '/'
        _advance!(l); return Token{T}(OPERATOR, :null, :/)
    elseif c == '^'
        _advance!(l); return Token{T}(OPERATOR, :null, :^)
    elseif c == '='
        _advance!(l); return Token{T}(OPERATOR, :null, :(=))
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
            error("Invalid scientific notation")
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
# AST
########################################################

struct ASTNode{T <: Number}
    # args::Union{Nothing, Vector{ASTNode{T}}}
    arg::Union{Nothing, ASTNode{T}}
    id::Int
    left::Union{Nothing, ASTNode{T}}
    name_id::NameID
    operator::Operator
    right::Union{Nothing, ASTNode{T}}
    value::Value{T}

    function ASTNode{T}(arg, id, left, name_id, operator, right, value) where T <: Number
        new{T}(arg, id, left, name_id, operator, right, value)
    end
end

function __eval_ast_call(ast::ASTNode{T}, vars) where T <: Number
    input = _eval_ast(ast.arg, vars)
    if ast.name_id == FUNC_EXP
        return exp(input)
    elseif ast.name_id == FUNC_SIN
        return sin(input)
    else
        @assert false "Unsupported function with id $(ast.name_id)"
    end
end

function __eval_ast_var(::ASTNode{T}, var::T)::T where T <: Number
    return var
end

function __eval_ast_var(ast::ASTNode{T}, vars::AbstractVector{T})::T where T <: Number
    return vars[ast.name_id]
end

function _eval_ast(ast::ASTNode{T}, vars)::T where T <: Number
    if ast.id == BINARY
        left = _eval_ast(ast.left, vars)
        right = _eval_ast(ast.right, vars)

        if ast.operator == :+
            return left + right
        elseif ast.operator == :-
            return left - right
        elseif ast.operator == :*
            return left * right
        elseif ast.operator == :/
            return left / right
        elseif ast.operator == :^
            return left ^ right
        else
            @assert false "Unsupport operator $(ast.operator)"
        end
    elseif ast.id == CALL
        return __eval_ast_call(ast, vars)
    elseif ast.id == NUMBER
        return ast.value::T
    elseif ast.id == UNARY
        if ast.operator == :-
            val = _eval_ast(ast.arg, vars)
            return -val
        else
            @assert false "Unsupport operator $(ast.operator)"
        end
    elseif ast.id == VARIABLE
        return __eval_ast_var(ast, vars)
    else
        @assert false "Unsupported type id $(ast.id)"
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
    if func_name == "exp"
        return FUNC_EXP
    elseif func_name == "sin"
        return FUNC_SIN
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
        if token.op == :+ || token.op == :-
            return 10
        elseif token.op == :* || token.op == :/
            return 20
        elseif token.op == :^
            return 30
        else    
            error("Unknown op")
        end
    end
end

function _led(p::Parser, token::Token, left, ::Type{T}) where T <: Number
    right = _parse_statement(p, _lbp(token))
    return ASTNode{T}(nothing, BINARY, left, nothing, token.op, right, nothing)
end

function _nud(p::Parser, t::Token, ::Type{T}) where T <: Number
    if t.id == IDENTIFIER
        if p.current.id == LEFT_PAREN
            _advance!(p) # consume '('
            args = ASTNode{T}[]
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
            return ASTNode{T}(args[1], CALL, nothing, func_id, nothing, nothing, nothing)
        else
            if string(t.name) in p.var_names
                name_id = findfirst(x -> x == string(t.name), p.var_names)
                return ASTNode{T}(nothing, VARIABLE, nothing, name_id, nothing, nothing, nothing)
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
        return ASTNode{T}(nothing, NUMBER, nothing, nothing, nothing, nothing, t.value)
    elseif t.id == OPERATOR
        if t.op == :-
            val = _parse_statement(p, 100)
            return ASTNode{T}(val, UNARY, nothing, nothing, :-, nothing, nothing)
        else
            error("Unexpected operator in _nud")
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
# Front facing API
########################################################
struct ExpressionFunction{T <: Number}
    ast::ASTNode{T}
    num_vars::Int

    function ExpressionFunction{T}(string::String, var_names::Vector{String}) where T <: Number
        p = Parser{T}(string, var_names)
        # params = _find_parameters(p)
        _reset!(p)
        ast = _parse_statement(p, 0)
        new{T}(ast, length(var_names))
    end
end

function (f::ExpressionFunction)(var::T) where T <: Number
    @assert f.num_vars == 1
    return _eval_ast(f.ast, var)
end

function (f::ExpressionFunction)(vars::AbstractVector{T}) where T <: Number
    return _eval_ast(f.ast, vars)
end

# for ic type funcs
function (f::ExpressionFunction)(X::SVector{ND, T}) where {ND, T <: Number}
    @assert f.num_vars == ND
    return _eval_ast(f.ast, X)
end

# for bc type funcs
function (f::ExpressionFunction)(X::SVector{ND, T}, t::T) where {ND, T <: Number}
    @assert f.num_vars == ND + 1
    vars = SVector{ND + 1, T}(X..., t)
    return _eval_ast(f.ast, vars)
end

end # module
