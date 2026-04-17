function test_constant_expression_functions()
    string = "5.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 5.0

    string = "5.0 + 3.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 8.0

    string = "5.0 + (4.0 * 2.0) / 4.0 + 2.0"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 9.0
end

function test_different_floats()
    string = "5.64e-12"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ 5.64e-12

    string = "-4.68e-68"
    func = ExpressionFunction{Float64}(string, String[])
    val = func(Float64[])
    @test val ≈ -4.68e-68
end

function test_simple_variable_expressions()
    string = "x"
    func = ExpressionFunction{Float64}(string, String["x"])
    val = func(Float64[53.0])
    @test val ≈ 53.0

    string = "x"
    func = ExpressionFunction{Float64}(string, String["x", "y"])
    val = func(Float64[53.0, 40.0])
    @test val ≈ 53.0

    string = "y"
    func = ExpressionFunction{Float64}(string, String["x", "y"])
    val = func(Float64[53.0, 40.0])
    @test val ≈ 40.0
end

function test_variable_arithmetic()
    string = "x * x"
    func = ExpressionFunction{Float64}(string, ["x"])
    val = func(5.0)
    @test val ≈ 25.0

    string = "y + (x * x + 5) * y"
    func = ExpressionFunction{Float64}(string, ["x", "y"])
    val = func([5.0, 2.0])
    @test val ≈ 62.0

    string = "x + y + z * t"
    func = ExpressionFunction{Float64}(string, ["x", "y", "z", "t"])
    X = SVector{3, Float64}(1.0, 2.0, 3.0)
    t = 15.0
    val = func(X, t)

    string = "x * y * exp(x + y)"
    func = ExpressionFunction{Float64}(string, ["x", "y"])
    val = func([1.0, 2.0])
    @test val ≈ 1.0 * 2.0 * exp(1.0 + 2.0)
end

@testset "Expressions" begin
    test_constant_expression_functions()
    test_different_floats()
    test_simple_variable_expressions()
    test_variable_arithmetic()
end
