import FiniteElementContainers: Expressions as Exp
import FiniteElementContainers.Expressions: Parser
using StaticArrays

function test_method()
    # constant
    println(Core.stdout, "Number")
    string = "5.0"
    func = Exp.ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    println(Core.stdout, "func = $val")

    # constant + constant
    println(Core.stdout, "Number + Number")
    string = "5.0 + 3.0"
    func = Exp.ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Number + Number")
    string = "5.0 + (3.0 * 8.0)"
    func = Exp.ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Variable")
    string = "x"
    func = Exp.ExpressionFunction{Float64}(string, ["x"])
    val = func([5.0])
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Variable")
    string = "x * x"
    func = Exp.ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Variable")
    string = "y + (x * x + 5) * y"
    func = Exp.ExpressionFunction{Float64}(string, ["x", "y"])
    val = func([5.0, 2.0])
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Bc like function")
    string = "x + y + z * t"
    func = Exp.ExpressionFunction{Float64}(string, ["x", "y", "z", "t"])
    X = SVector{3, Float64}(1.0, 2.0, 3.0)
    t = 4.0
    val = func(X, t)
    println(Core.stdout, "func = $val")

    println(Core.stdout, "Function call")
    string = "x * y * exp(x + y)"
    func = Exp.ExpressionFunction{Float64}(string, ["x", "y"])
    val = func([1.0, 2.0])
    println(Core.stdout, "func = $val")
end

function @main(ARGS)
    test_method()
    return 0
end

# test_method()
