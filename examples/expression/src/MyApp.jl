import FiniteElementContainers: Expressions as Exp
import FiniteElementContainers.Expressions: Parser
using DynamicExpressions
using StaticArrays

const operators = OperatorEnum(1 => (sin, cos, exp), 2 => (+, -, *, /))

function test_method()
    # constant
    # println(Core.stdout, "Number")
    # string = "5.0"
    # func = Exp.ScalarExpressionFunction{Float64}(string, String[])
    # val = func(Float64[])
    # println(Core.stdout, "func = $val")

    # # constant + constant
    # println(Core.stdout, "Number + Number")
    # string = "5.0 + 3.0"
    # func = Exp.ScalarExpressionFunction{Float64}(string, String[])
    # val = func(Float64[])
    # println(Core.stdout, "func = $val")

    # println(Core.stdout, "Number + Number")
    # string = "5.0 + (3.0 * 8.0)"
    # func = Exp.ScalarExpressionFunction{Float64}(string, String[])
    # val = func(Float64[])
    # println(Core.stdout, "func = $val")

    println(Core.stdout, "Variable")
    string = "x"
    func = Exp.ScalarExpressionFunction{Float64}(string, ["x"])
    val = func([5.0])
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Variable")
    string = "x * x"
    func = Exp.ScalarExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Variable")
    string = "y + (x * x + 5) * y"
    func = Exp.ScalarExpressionFunction{Float64}(string, ["x", "y"])
    val = func([5.0, 2.0])
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Bc like function")
    string = "x + y + z * t"
    func = Exp.ScalarExpressionFunction{Float64}(string, ["x", "y", "z", "t"])
    X = SVector{3, Float64}(1.0, 2.0, 3.0)
    t = 4.0
    val = func(X, t)
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Function call")
    string = "x * y * exp(x + y)"
    func = Exp.ScalarExpressionFunction{Float64}(string, ["x", "y"])
    val = func([1.0, 2.0])
    println(Core.stdout, "func = $val")

    variable_names = ["x", "y"]
    c = Node{Float64}(; val = 2.0)
    x = Node{Float64}(; feature = 1)
    y = Node{Float64}(; feature = 2)
    complex_node = Node{Float64}(; op=3, l=x, r=Node{Float64}(; op=1, l=y, r=c))
    expr = Expression(complex_node; operators, variable_names)
    vars = SMatrix{2, 1, Float64, 2}(5.0, 6.0)
    out = expr(vars)[1]
    println(Core.stdout, out)
    # x_expr = DynamicExpressions.Expresssion(x; operators, variable_names)
end

function @main(ARGS)
    test_method()
    return 0
end
